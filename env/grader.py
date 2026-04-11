"""
Deterministic grader for the Student Task Manager Environment.

Evaluates agent performance at the end of an episode and returns a score in [0.0, 1.0].
The grader measures:
  1. Completion rate       – what fraction of tasks were completed
  2. Deadline adherence    – how promptly tasks were completed relative to deadlines
  3. Scheduling efficiency – hours utilized vs. wasted + priority alignment
  4. Dependency adherence  – no dependency violations
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from env.models import EnvironmentState, Task


# ---------------------------------------------------------------------------
# Weight constants (must sum to 1.0)
# ---------------------------------------------------------------------------

W_COMPLETION: float = 0.40      # Completion rate weight
W_DEADLINE: float = 0.30        # Deadline adherence weight
W_EFFICIENCY: float = 0.20      # Scheduling efficiency weight
W_DEPENDENCY: float = 0.10      # Dependency adherence weight


# ---------------------------------------------------------------------------
# Sub-score calculators
# ---------------------------------------------------------------------------

def _completion_score(tasks: Dict[str, Task]) -> Tuple[float, Dict[str, Any]]:
    """
    Score based on tasks completed (with partial credit for progress).
    Returns value in [0, 1].
    """
    if not tasks:
        return 0.01, {}

    total = len(tasks)
    full_complete = sum(1 for t in tasks.values() if t.is_completed)
    partial_credit = sum(
        t.progress / 100.0
        for t in tasks.values()
        if not t.is_completed
    )

    raw = (full_complete + partial_credit * 0.5) / total
    score = max(0.01, min(0.99, raw))

    details = {
        "total_tasks": total,
        "fully_completed": full_complete,
        "partial_credit_tasks": total - full_complete,
        "partial_hours_credited": round(partial_credit, 3),
        "raw_score": round(raw, 4),
    }
    return round(score, 4), details


def _deadline_score(tasks: Dict[str, Task], total_days: int) -> Tuple[float, Dict[str, Any]]:
    """
    Score based on how well deadlines were met.
    - Completed on time → full score
    - Completed late → partial (decays with lateness)
    - Not completed but not overdue → small partial
    - Overdue → 0 for that task
    Returns value in [0, 1].
    """
    if not tasks:
        return 0.01, {}

    task_scores = []
    details_list = []

    for t in tasks.values():
        if t.is_completed:
            if t.completion_day is not None and t.completion_day <= t.deadline_day:
                # On-time bonus: earlier → higher score
                days_early = t.deadline_day - t.completion_day
                bonus = min(0.2, days_early * 0.05)
                ts = min(0.99, 1.0 + bonus)
            else:
                # Late completion – decay by days late
                days_late = (t.completion_day or total_days) - t.deadline_day
                ts = max(0.01, 1.0 - days_late * 0.2)
            details_list.append({"task_id": t.task_id, "deadline_score": round(ts, 3), "status": "completed"})
        elif t.is_overdue:
            ts = 0.01
            details_list.append({"task_id": t.task_id, "deadline_score": 0.01, "status": "overdue"})
        else:
            # In progress but not overdue – give partial credit proportional to progress
            ts = max(0.01, (t.progress / 100.0) * 0.3)
            details_list.append({"task_id": t.task_id, "deadline_score": round(ts, 3), "status": "incomplete"})

        task_scores.append(ts)

    score = sum(task_scores) / len(task_scores) if task_scores else 0.01
    return round(max(0.01, min(0.99, score)), 4), {"per_task": details_list}


def _efficiency_score(state: EnvironmentState) -> Tuple[float, Dict[str, Any]]:
    """
    Score based on:
    - Total productive hours / total available hours
    - Priority alignment (were high-priority tasks worked on?)
    Returns value in [0, 1].
    """
    total_available = state.total_days * state.max_hours_per_day
    total_worked = sum(t.hours_worked for t in state.tasks.values())

    utilization = min(0.99, total_worked / total_available) if total_available > 0 else 0.01

    # Priority alignment: reward if high-importance tasks received more hours proportionally
    tasks = list(state.tasks.values())
    if not tasks:
        return 0.01, {}

    # Expected hours distribution based on importance
    total_importance = sum(t.importance for t in tasks)
    alignment_score = 0.0
    if total_importance > 0:
        for t in tasks:
            expected_fraction = t.importance / total_importance
            actual_fraction = t.hours_worked / total_worked if total_worked > 0 else 0.0
            # Penalize deviation from ideal allocation
            deviation = abs(actual_fraction - expected_fraction)
            alignment_score += max(0, 1.0 - deviation * 5.0) / len(tasks)

    score = utilization * 0.5 + alignment_score * 0.5
    score = max(0.01, min(0.99, score))

    details = {
        "total_available_hours": round(total_available, 2),
        "total_worked_hours": round(total_worked, 2),
        "utilization_rate": round(utilization, 4),
        "priority_alignment_score": round(alignment_score, 4),
    }
    return round(score, 4), details


def _dependency_score(
    tasks: Dict[str, Task], action_history: List[Dict[str, Any]]
) -> Tuple[float, Dict[str, Any]]:
    """
    Score penalizing dependency violations (working on a task before its prerequisites).
    Returns value in [0, 1] (1 = no violations).
    """
    violations = sum(
        1
        for entry in action_history
        if entry.get("dependency_violation", False)
    )
    total_steps = len(action_history)

    if total_steps == 0:
        return 0.99, {"violations": 0, "total_steps": 0}

    violation_rate = violations / total_steps
    # Sigmoid-style decay: few violations → near 1, many → near 0
    score = math.exp(-violation_rate * 5.0)
    score = max(0.01, min(0.99, score))

    return round(score, 4), {
        "dependency_violations": violations,
        "total_steps": total_steps,
        "violation_rate": round(violation_rate, 4),
    }


# ---------------------------------------------------------------------------
# Public grader function
# ---------------------------------------------------------------------------

def grade(state: EnvironmentState) -> Dict[str, Any]:
    """
    Grade the completed episode.

    Args:
        state: Final EnvironmentState after episode termination.

    Returns:
        Dict with:
            score (float): Final normalized score in [0.0, 1.0]
            breakdown (dict): Per-component scores and metadata
    """
    comp_score, comp_details = _completion_score(state.tasks)
    dead_score, dead_details = _deadline_score(state.tasks, state.total_days)
    eff_score, eff_details = _efficiency_score(state)
    dep_score, dep_details = _dependency_score(state.tasks, state.action_history)

    # Weighted aggregate
    raw_score = (
        W_COMPLETION * comp_score
        + W_DEADLINE * dead_score
        + W_EFFICIENCY * eff_score
        + W_DEPENDENCY * dep_score
    )

    # Clamp to strictly within (0, 1)
    final_score = max(0.01, min(0.99, raw_score))

    return {
        "score": round(final_score, 4),
        "breakdown": {
            "completion": {
                "weight": W_COMPLETION,
                "score": comp_score,
                "details": comp_details,
            },
            "deadline_adherence": {
                "weight": W_DEADLINE,
                "score": dead_score,
                "details": dead_details,
            },
            "scheduling_efficiency": {
                "weight": W_EFFICIENCY,
                "score": eff_score,
                "details": eff_details,
            },
            "dependency_adherence": {
                "weight": W_DEPENDENCY,
                "score": dep_score,
                "details": dep_details,
            },
        },
        "summary": {
            "total_tasks": len(state.tasks),
            "completed": len(state.completed_tasks),
            "overdue": len(state.overdue_tasks),
            "pending": len(state.pending_tasks),
            "days_used": state.current_day,
            "steps_taken": state.steps_taken,
        },
    }
