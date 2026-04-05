"""
Core environment implementation for the Student Task Manager OpenEnv.

Implements the full OpenEnv API:
  - reset() → Observation
  - step(action) → (Observation, RewardBreakdown, done: bool, info: dict)
  - state() → EnvironmentState
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Optional, Tuple

from env.models import (
    Action,
    ActionType,
    EnvironmentState,
    Observation,
    RewardBreakdown,
    Task,
    TaskSummary,
)
from env.tasks import get_scenario
from env.grader import grade


# ---------------------------------------------------------------------------
# Reward hyper-parameters
# ---------------------------------------------------------------------------

REWARD_TASK_COMPLETE = 0.40          # Base reward for completing a task
REWARD_EARLY_BONUS_MAX = 0.15        # Max extra for completing early
REWARD_PROGRESS_PER_HOUR = 0.025     # Reward per hour of productive work
REWARD_PRIORITY_MULT = 0.10          # Extra for high-importance tasks (scaled by importance/5)

PENALTY_DEADLINE_MISS = -0.30        # Penalty when a deadline passes
PENALTY_IDLE = -0.05                 # Penalty for skipping a day
PENALTY_DEP_VIOLATION = -0.10        # Penalty for touching a blocked task
PENALTY_LOW_PRIORITY = -0.02         # Penalty per hour spent on low-prio while urgent tasks exist

MAX_STEPS_DEFAULT = 120
EPISODE_DONE_OVERDUE_THRESHOLD = 0.5  # End episode if >50% tasks overdue


class StudentTaskManagerEnv:
    """
    OpenEnv-compliant Student Task Manager Environment.

    The agent represents an AI scheduler that helps a student manage
    assignments across multiple days with limited daily work hours.
    """

    def __init__(self, scenario: str = "medium", seed: int = 42) -> None:
        self._scenario_name = scenario.lower()
        self._seed = seed
        self._internal_state: Optional[EnvironmentState] = None
        self.reset()

    # -----------------------------------------------------------------------
    # OpenEnv Core API
    # -----------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset the environment to the initial state and return the first observation."""
        cfg = get_scenario(self._scenario_name)
        tasks_dict = {t.task_id: t for t in cfg["tasks"]}

        self._internal_state = EnvironmentState(
            current_day=1,
            current_hour=0.0,
            remaining_time_today=cfg["max_hours_per_day"],
            max_hours_per_day=cfg["max_hours_per_day"],
            total_days=cfg["total_days"],
            tasks=tasks_dict,
            priority_order=list(tasks_dict.keys()),
            active_task_id=None,
            steps_taken=0,
            max_steps=cfg["max_steps"],
            cumulative_reward=0.0,
            episode_done=False,
            action_history=[],
            reward_history=[],
            day_summaries=[],
            seed=self._seed,
        )
        return self._build_observation(last_action_status="reset", last_action_message="Environment reset.")

    def step(self, action: Action | Dict[str, Any]) -> Tuple[Observation, RewardBreakdown, bool, Dict[str, Any]]:
        """
        Execute one action in the environment.

        Args:
            action: A validated Action model or raw dict.

        Returns:
            observation: Current environment observation.
            reward: RewardBreakdown with total in [0, 1].
            done: Whether the episode has ended.
            info: Additional metadata and error messages.
        """
        s = self._internal_state
        if s is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        if s.episode_done:
            obs = self._build_observation("error", "Episode already done. Call reset().")
            reward = RewardBreakdown(total=0.0)
            return obs, reward, True, {"error": "Episode done"}

        # Parse action
        parsed_action, parse_error = self._parse_action(action)
        if parse_error:
            reward = RewardBreakdown(total=0.0, raw_total=0.0)
            s.steps_taken += 1
            s.action_history.append({"step": s.steps_taken, "action": str(action), "error": parse_error})
            obs = self._build_observation("invalid", parse_error)
            done = self._check_done()
            return obs, reward, done, {"error": parse_error, "last_action_status": "invalid"}

        # Dispatch action
        reward, action_status, action_message, action_meta = self._dispatch(parsed_action)

        # Advance simulation clock
        self._advance_time(parsed_action)

        # Record history
        s.steps_taken += 1
        s.cumulative_reward += reward.total
        s.reward_history.append(reward.total)
        s.action_history.append({
            "step": s.steps_taken,
            "day": s.current_day,
            "action_type": parsed_action.action_type.value,
            "task_id": parsed_action.task_id,
            "hours": parsed_action.hours,
            "reward": round(reward.total, 4),
            "status": action_status,
            "dependency_violation": action_meta.get("dependency_violation", False),
        })

        # Check termination
        done = self._check_done()
        if done:
            s.episode_done = True

        obs = self._build_observation(action_status, action_message)
        info = {
            "last_action_status": action_status,
            "last_action_message": action_message,
            "reward_breakdown": reward.model_dump(),
            "step": s.steps_taken,
            "day": s.current_day,
            **action_meta,
        }

        if done:
            grading = grade(s)
            info["episode_grade"] = grading
            info["final_score"] = grading["score"]

        return obs, reward, done, info

    def state(self) -> Dict[str, Any]:
        """Return the full, serializable internal state."""
        if self._internal_state is None:
            raise RuntimeError("Environment not initialized.")
        return self._internal_state.model_dump()

    # -----------------------------------------------------------------------
    # Action Dispatch
    # -----------------------------------------------------------------------

    def _dispatch(
        self, action: Action
    ) -> Tuple[RewardBreakdown, str, str, Dict[str, Any]]:
        """Route action to the appropriate handler."""
        handlers = {
            ActionType.SELECT_TASK: self._handle_select_task,
            ActionType.ALLOCATE_TIME: self._handle_allocate_time,
            ActionType.MARK_COMPLETE: self._handle_mark_complete,
            ActionType.SKIP_DAY: self._handle_skip_day,
            ActionType.REORDER_PRIORITY: self._handle_reorder_priority,
        }
        handler = handlers[action.action_type]
        return handler(action)

    def _handle_select_task(
        self, action: Action
    ) -> Tuple[RewardBreakdown, str, str, Dict[str, Any]]:
        s = self._internal_state
        task_id = action.task_id

        if task_id not in s.tasks:
            r = RewardBreakdown(total=0.0, raw_total=0.0)
            return r, "error", f"Task '{task_id}' not found.", {}

        task = s.tasks[task_id]
        if task.is_completed:
            r = RewardBreakdown(total=0.0, raw_total=0.0)
            return r, "warning", f"Task '{task_id}' is already completed.", {}

        # Check dependencies
        dep_ok, unmet = self._check_dependencies(task)
        if not dep_ok:
            penalty = PENALTY_DEP_VIOLATION
            raw = penalty
            total = max(0.0, min(1.0, 0.5 + penalty))
            r = RewardBreakdown(total=total, raw_total=raw, dependency_violation_penalty=penalty)
            return r, "dependency_blocked", (
                f"Cannot select '{task_id}': unmet dependencies {unmet}."
            ), {"dependency_violation": True}

        s.active_task_id = task_id
        r = RewardBreakdown(total=0.5, raw_total=0.5)  # neutral positive for valid selection
        return r, "success", f"Task '{task_id}' selected.", {}

    def _handle_allocate_time(
        self, action: Action
    ) -> Tuple[RewardBreakdown, str, str, Dict[str, Any]]:
        s = self._internal_state
        task_id = action.task_id
        hours = action.hours or 1.0

        if task_id not in s.tasks:
            r = RewardBreakdown(total=0.0, raw_total=0.0)
            return r, "error", f"Task '{task_id}' not found.", {}

        task = s.tasks[task_id]

        if task.is_completed:
            r = RewardBreakdown(total=0.0, raw_total=0.0)
            return r, "warning", f"Task '{task_id}' already completed, cannot allocate time.", {}

        # Check dependencies
        dep_ok, unmet = self._check_dependencies(task)
        if not dep_ok:
            penalty = PENALTY_DEP_VIOLATION
            raw = penalty
            total = max(0.0, min(1.0, 0.5 + penalty))
            r = RewardBreakdown(total=total, raw_total=raw, dependency_violation_penalty=penalty)
            return r, "dependency_blocked", (
                f"Task '{task_id}' is blocked by {unmet}."
            ), {"dependency_violation": True}

        # Cap hours to available time today and remaining task hours
        effective_hours = min(hours, s.remaining_time_today, task.remaining_hours + 0.1)
        effective_hours = max(0.01, effective_hours)

        # Update task progress
        progress_delta = min(
            100.0 - task.progress,
            (effective_hours / task.estimated_duration) * 100.0,
        )
        task.progress = min(100.0, task.progress + progress_delta)
        task.hours_worked += effective_hours
        s.remaining_time_today = max(0.0, s.remaining_time_today - effective_hours)

        # Check if task completed via allocation
        just_completed = False
        if task.progress >= 99.0:
            task.progress = 100.0
            task.is_completed = True
            task.completion_day = s.current_day
            just_completed = True

        # Compute reward
        raw, breakdown_kwargs = self._compute_allocate_reward(
            task, effective_hours, just_completed, s
        )
        total = max(0.0, min(1.0, raw))
        r = RewardBreakdown(total=total, raw_total=raw, **breakdown_kwargs)
        msg = (
            f"Allocated {effective_hours:.2f}h to '{task_id}'. "
            f"Progress: {task.progress:.1f}%."
            + (" ✓ Task completed!" if just_completed else "")
        )
        return r, "success", msg, {"effective_hours": effective_hours, "just_completed": just_completed}

    def _handle_mark_complete(
        self, action: Action
    ) -> Tuple[RewardBreakdown, str, str, Dict[str, Any]]:
        s = self._internal_state
        task_id = action.task_id

        if task_id not in s.tasks:
            r = RewardBreakdown(total=0.0, raw_total=0.0)
            return r, "error", f"Task '{task_id}' not found.", {}

        task = s.tasks[task_id]
        if task.is_completed:
            r = RewardBreakdown(total=0.0, raw_total=0.0)
            return r, "warning", f"Task '{task_id}' already marked complete.", {}

        if task.progress < 80.0:
            r = RewardBreakdown(total=0.0, raw_total=0.0)
            return r, "error", (
                f"Task '{task_id}' only {task.progress:.1f}% done. "
                "Must reach ≥80% before marking complete."
            ), {}

        task.progress = 100.0
        task.is_completed = True
        task.completion_day = s.current_day

        on_time = s.current_day <= task.deadline_day
        days_early = task.deadline_day - s.current_day if on_time else 0
        completion_bonus = REWARD_TASK_COMPLETE
        early_bonus = min(REWARD_EARLY_BONUS_MAX, days_early * 0.05) if on_time else 0.0
        raw = completion_bonus + early_bonus
        total = max(0.0, min(1.0, raw))

        r = RewardBreakdown(
            total=total,
            raw_total=raw,
            task_completion_bonus=completion_bonus,
            early_completion_bonus=early_bonus,
        )
        return r, "success", f"Task '{task_id}' marked complete! On-time: {on_time}.", {
            "on_time": on_time,
        }

    def _handle_skip_day(
        self, action: Action
    ) -> Tuple[RewardBreakdown, str, str, Dict[str, Any]]:
        s = self._internal_state

        # Advance to next day
        self._end_day()
        penalty = PENALTY_IDLE
        total = max(0.0, min(1.0, 0.5 + penalty))  # 0.45 – slight penalty
        r = RewardBreakdown(total=total, raw_total=penalty, idle_penalty=penalty)
        return r, "success", f"Skipped to day {s.current_day}.", {}

    def _handle_reorder_priority(
        self, action: Action
    ) -> Tuple[RewardBreakdown, str, str, Dict[str, Any]]:
        s = self._internal_state
        new_order = action.priority_order or []

        # Validate all IDs exist
        unknown = [tid for tid in new_order if tid not in s.tasks]
        if unknown:
            r = RewardBreakdown(total=0.0, raw_total=0.0)
            return r, "error", f"Unknown task IDs in priority order: {unknown}.", {}

        # Check that all tasks are included (warn but don't fail if some missing)
        missing = [tid for tid in s.tasks if tid not in new_order]
        # Append missing tasks at end
        s.priority_order = new_order + missing
        r = RewardBreakdown(total=0.5, raw_total=0.5)
        msg = f"Priority reordered. Top task: {s.priority_order[0] if s.priority_order else 'none'}."
        if missing:
            msg += f" Note: tasks {missing} appended at end."
        return r, "success", msg, {}

    # -----------------------------------------------------------------------
    # Reward Computation
    # -----------------------------------------------------------------------

    def _compute_allocate_reward(
        self,
        task: Task,
        hours: float,
        just_completed: bool,
        s: EnvironmentState,
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute nuanced reward for allocate_time action."""
        kwargs: Dict[str, Any] = {}
        raw = 0.0

        # Progress reward (proportional to hours worked)
        progress_r = REWARD_PROGRESS_PER_HOUR * hours
        kwargs["progress_reward"] = progress_r
        raw += progress_r

        # Completion bonus
        if just_completed:
            comp_r = REWARD_TASK_COMPLETE
            on_time = s.current_day <= task.deadline_day
            early_bonus = (
                min(REWARD_EARLY_BONUS_MAX, (task.deadline_day - s.current_day) * 0.05)
                if on_time else 0.0
            )
            kwargs["task_completion_bonus"] = comp_r
            kwargs["early_completion_bonus"] = early_bonus
            raw += comp_r + early_bonus

        # Priority alignment reward
        is_urgent = (task.deadline_day - s.current_day) <= 2
        is_high_importance = task.importance >= 4
        has_urgent_pending = bool(s.urgent_tasks)

        if is_urgent or is_high_importance:
            prio_r = REWARD_PRIORITY_MULT * (task.importance / 5.0)
            kwargs["priority_alignment_reward"] = prio_r
            raw += prio_r
        elif has_urgent_pending and task.importance < 3:
            # Penalize working on low-priority while urgent tasks exist
            low_prio_pen = PENALTY_LOW_PRIORITY * hours
            kwargs["low_priority_penalty"] = low_prio_pen
            raw += low_prio_pen

        return raw, kwargs

    # -----------------------------------------------------------------------
    # Time Management
    # -----------------------------------------------------------------------

    def _advance_time(self, action: Action) -> None:
        """Advance the simulation clock after an action."""
        s = self._internal_state
        # If time exhausted, advance to next day
        if s.remaining_time_today <= 0.01 and action.action_type != ActionType.SKIP_DAY:
            self._end_day()

    def _end_day(self) -> None:
        """Finalize the current day and advance to the next."""
        s = self._internal_state

        # Mark tasks as overdue if deadline passed
        overdue_penalty = 0.0
        for task in s.tasks.values():
            if not task.is_completed and s.current_day >= task.deadline_day:
                if not task.is_overdue:
                    task.is_overdue = True
                    overdue_penalty += abs(PENALTY_DEADLINE_MISS)

        # Record day summary
        s.day_summaries.append({
            "day": s.current_day,
            "hours_used": s.max_hours_per_day - s.remaining_time_today,
            "tasks_completed_today": [
                t.task_id for t in s.tasks.values()
                if t.is_completed and t.completion_day == s.current_day
            ],
            "new_overdue": [t.task_id for t in s.tasks.values() if t.is_overdue],
            "overdue_penalty": overdue_penalty,
        })

        # Advance day
        s.current_day += 1
        s.remaining_time_today = s.max_hours_per_day
        s.active_task_id = None

    # -----------------------------------------------------------------------
    # Dependency Checking
    # -----------------------------------------------------------------------

    def _check_dependencies(self, task: Task) -> Tuple[bool, list]:
        """Return (all_met, list_of_unmet_dep_ids)."""
        s = self._internal_state
        unmet = [
            dep_id
            for dep_id in task.dependencies
            if dep_id in s.tasks and not s.tasks[dep_id].is_completed
        ]
        return len(unmet) == 0, unmet

    # -----------------------------------------------------------------------
    # Episode Termination
    # -----------------------------------------------------------------------

    def _check_done(self) -> bool:
        s = self._internal_state
        # All tasks completed
        if all(t.is_completed for t in s.tasks.values()):
            return True
        # Max steps reached
        if s.steps_taken >= s.max_steps:
            return True
        # Max days exceeded
        if s.current_day > s.total_days:
            return True
        # Too many overdue tasks
        overdue_count = sum(1 for t in s.tasks.values() if t.is_overdue)
        overdue_fraction = overdue_count / len(s.tasks) if s.tasks else 0
        if overdue_fraction > EPISODE_DONE_OVERDUE_THRESHOLD:
            return True
        return False

    # -----------------------------------------------------------------------
    # Observation Builder
    # -----------------------------------------------------------------------

    def _build_observation(self, last_action_status: str, last_action_message: str) -> Observation:
        s = self._internal_state

        task_summaries = []
        for task in s.tasks.values():
            task_summaries.append(
                TaskSummary(
                    task_id=task.task_id,
                    subject=task.subject,
                    title=task.title,
                    deadline_day=task.deadline_day,
                    difficulty=task.difficulty,
                    importance=task.importance,
                    progress=round(task.progress, 1),
                    remaining_hours=round(task.remaining_hours, 2),
                    is_completed=task.is_completed,
                    is_overdue=task.is_overdue,
                    has_unmet_dependencies=not self._check_dependencies(task)[0],
                    days_until_deadline=max(0, task.deadline_day - s.current_day),
                )
            )

        completed_ids = [t.task_id for t in s.tasks.values() if t.is_completed]
        overdue_ids = [t.task_id for t in s.tasks.values() if t.is_overdue]
        pending_ids = [t.task_id for t in s.tasks.values() if not t.is_completed and not t.is_overdue]
        urgent_ids = [t.task_id for t in s.urgent_tasks]

        episode_score = s.cumulative_reward / max(1, s.steps_taken)

        return Observation(
            current_day=s.current_day,
            total_days=s.total_days,
            remaining_time_today=round(s.remaining_time_today, 2),
            max_hours_per_day=s.max_hours_per_day,
            tasks=task_summaries,
            active_task_id=s.active_task_id,
            completed_task_ids=completed_ids,
            overdue_task_ids=overdue_ids,
            pending_task_ids=pending_ids,
            total_tasks=len(s.tasks),
            completed_count=len(completed_ids),
            overdue_count=len(overdue_ids),
            pending_count=len(pending_ids),
            episode_score_so_far=round(min(1.0, max(0.0, episode_score)), 4),
            steps_taken=s.steps_taken,
            max_steps=s.max_steps,
            cumulative_reward=round(s.cumulative_reward, 4),
            last_action_type=s.action_history[-1]["action_type"] if s.action_history else None,
            last_action_status=last_action_status,
            last_action_message=last_action_message,
            urgent_tasks=urgent_ids,
        )

    # -----------------------------------------------------------------------
    # Helper
    # -----------------------------------------------------------------------

    def _parse_action(self, action: Any) -> Tuple[Optional[Action], Optional[str]]:
        """Parse raw dict or Action object. Returns (parsed, error_message)."""
        if isinstance(action, Action):
            return action, None
        if isinstance(action, dict):
            try:
                return Action(**action), None
            except Exception as e:
                return None, f"Invalid action format: {e}"
        return None, f"Unsupported action type: {type(action)}"

    # -----------------------------------------------------------------------
    # Convenience properties
    # -----------------------------------------------------------------------

    @property
    def scenario(self) -> str:
        return self._scenario_name

    @property
    def current_day(self) -> int:
        return self._internal_state.current_day if self._internal_state else 0
