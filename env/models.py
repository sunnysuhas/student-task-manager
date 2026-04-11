"""
Pydantic models for the Student Task Manager Environment.
Defines all typed data models: Task, Observation, Action, Reward, and State.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    SELECT_TASK = "select_task"
    ALLOCATE_TIME = "allocate_time"
    MARK_COMPLETE = "mark_complete"
    SKIP_DAY = "skip_day"
    REORDER_PRIORITY = "reorder_priority"


class DifficultyLevel(int, Enum):
    VERY_EASY = 1
    EASY = 2
    MEDIUM = 3
    HARD = 4
    VERY_HARD = 5


class ImportanceLevel(int, Enum):
    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5


# ---------------------------------------------------------------------------
# Core Entity Models
# ---------------------------------------------------------------------------

class Task(BaseModel):
    """Represents a single student assignment."""

    task_id: str = Field(..., description="Unique task identifier")
    subject: str = Field(..., description="Subject/course name")
    title: str = Field(..., description="Assignment title")
    deadline_day: int = Field(..., ge=1, description="Day number by which the task must be completed")
    estimated_duration: float = Field(..., gt=0.0, le=24.0, description="Estimated hours to complete")
    difficulty: int = Field(..., ge=1, le=5, description="Difficulty rating 1-5")
    importance: int = Field(..., ge=1, le=5, description="Importance rating 1-5")
    dependencies: List[str] = Field(default_factory=list, description="List of task_ids that must be completed first")
    progress: float = Field(default=0.0, ge=0.0, le=100.0, description="Completion percentage 0-100")
    hours_worked: float = Field(default=0.0, ge=0.0, description="Total hours worked on this task")
    is_completed: bool = Field(default=False, description="Whether the task is fully completed")
    is_overdue: bool = Field(default=False, description="Whether the task missed its deadline")
    completion_day: Optional[int] = Field(default=None, description="Day the task was completed")

    @field_validator("difficulty")
    @classmethod
    def validate_difficulty(cls, v: int) -> int:
        if v not in range(1, 6):
            raise ValueError("difficulty must be between 1 and 5")
        return v

    @field_validator("importance")
    @classmethod
    def validate_importance(cls, v: int) -> int:
        if v not in range(1, 6):
            raise ValueError("importance must be between 1 and 5")
        return v

    @property
    def priority_score(self) -> float:
        """Composite priority score combining importance and difficulty."""
        return (self.importance * 2.0 + self.difficulty) / 15.0  # normalized 0-1

    @property
    def remaining_hours(self) -> float:
        """Hours still needed to complete this task."""
        if self.is_completed:
            return 0.0
        completed_fraction = self.progress / 100.0
        return max(0.0, self.estimated_duration * (1.0 - completed_fraction))

    def model_dump_summary(self) -> Dict[str, Any]:
        """Compact dict for observations."""
        return {
            "task_id": self.task_id,
            "subject": self.subject,
            "title": self.title,
            "deadline_day": self.deadline_day,
            "difficulty": self.difficulty,
            "importance": self.importance,
            "progress": round(self.progress, 1),
            "remaining_hours": round(self.remaining_hours, 2),
            "is_completed": self.is_completed,
            "is_overdue": self.is_overdue,
            "has_unmet_dependencies": len(self.dependencies) > 0,
        }


class TaskSummary(BaseModel):
    """Lightweight task summary for observation space."""
    task_id: str
    subject: str
    title: str
    deadline_day: int
    difficulty: int
    importance: int
    progress: float
    remaining_hours: float
    is_completed: bool
    is_overdue: bool
    has_unmet_dependencies: bool
    days_until_deadline: int


# ---------------------------------------------------------------------------
# Action Model
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """Typed, validated action the agent can take."""

    action_type: ActionType = Field(..., description="Type of action to perform")
    task_id: Optional[str] = Field(default=None, description="Target task ID (for task-specific actions)")
    hours: Optional[float] = Field(default=None, ge=0.1, le=8.0, description="Hours to allocate (for allocate_time)")
    priority_order: Optional[List[str]] = Field(
        default=None, description="Ordered list of task_ids (for reorder_priority)"
    )

    @model_validator(mode="after")
    def validate_action_fields(self) -> "Action":
        if self.action_type == ActionType.SELECT_TASK and not self.task_id:
            raise ValueError("select_task requires a task_id")
        if self.action_type == ActionType.ALLOCATE_TIME:
            if not self.task_id:
                raise ValueError("allocate_time requires a task_id")
            if self.hours is None:
                raise ValueError("allocate_time requires hours")
        if self.action_type == ActionType.MARK_COMPLETE and not self.task_id:
            raise ValueError("mark_complete requires a task_id")
        if self.action_type == ActionType.REORDER_PRIORITY and not self.priority_order:
            raise ValueError("reorder_priority requires priority_order list")
        return self


# ---------------------------------------------------------------------------
# Observation Model
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """Complete observation returned to the agent after each step."""

    current_day: int = Field(..., description="Current simulation day (1-indexed)")
    total_days: int = Field(..., description="Total days in the episode")
    remaining_time_today: float = Field(..., ge=0.0, description="Hours remaining today")
    max_hours_per_day: float = Field(..., description="Max working hours per day")

    tasks: List[TaskSummary] = Field(default_factory=list, description="All tasks with their current state")
    active_task_id: Optional[str] = Field(default=None, description="Currently selected task")

    completed_task_ids: List[str] = Field(default_factory=list)
    overdue_task_ids: List[str] = Field(default_factory=list)
    pending_task_ids: List[str] = Field(default_factory=list)

    total_tasks: int = Field(..., description="Total number of tasks")
    completed_count: int = Field(default=0)
    overdue_count: int = Field(default=0)
    pending_count: int = Field(default=0)

    episode_score_so_far: float = Field(default=0.01, ge=0.01, le=0.99)
    steps_taken: int = Field(default=0)
    max_steps: int = Field(...)
    cumulative_reward: float = Field(default=0.0)

    last_action_type: Optional[str] = Field(default=None)
    last_action_status: str = Field(default="none")
    last_action_message: str = Field(default="")

    urgent_tasks: List[str] = Field(
        default_factory=list,
        description="Task IDs with deadlines within 2 days"
    )


# ---------------------------------------------------------------------------
# Reward Model
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    """Detailed breakdown of the reward signal."""

    total: float = Field(..., ge=0.01, le=0.99, description="Total reward for this step, clamped to [0,1]")

    task_completion_bonus: float = Field(default=0.0, description="Reward for completing a task")
    early_completion_bonus: float = Field(default=0.0, description="Extra reward for completing before deadline")
    progress_reward: float = Field(default=0.0, description="Reward for making progress")
    priority_alignment_reward: float = Field(default=0.0, description="Reward for working on high-priority tasks")

    deadline_miss_penalty: float = Field(default=0.0, description="Penalty for missing a deadline (negative)")
    idle_penalty: float = Field(default=0.0, description="Penalty for skipping or idle time (negative)")
    dependency_violation_penalty: float = Field(default=0.0, description="Penalty for working on blocked tasks")
    low_priority_penalty: float = Field(default=0.0, description="Penalty for ignoring urgent tasks")

    raw_total: float = Field(default=0.0, description="Pre-clamp raw reward")


# ---------------------------------------------------------------------------
# Full Internal State Model
# ---------------------------------------------------------------------------

class EnvironmentState(BaseModel):
    """Complete, serializable internal state of the environment."""

    # Time
    current_day: int = Field(default=1)
    current_hour: float = Field(default=0.0)
    remaining_time_today: float = Field(...)
    max_hours_per_day: float = Field(...)
    total_days: int = Field(...)

    # Tasks
    tasks: Dict[str, Task] = Field(default_factory=dict)
    priority_order: List[str] = Field(default_factory=list)
    active_task_id: Optional[str] = Field(default=None)

    # Episode tracking
    steps_taken: int = Field(default=0)
    max_steps: int = Field(...)
    cumulative_reward: float = Field(default=0.0)
    episode_done: bool = Field(default=False)

    # History
    action_history: List[Dict[str, Any]] = Field(default_factory=list)
    reward_history: List[float] = Field(default_factory=list)
    day_summaries: List[Dict[str, Any]] = Field(default_factory=list)

    # Seed
    seed: int = Field(default=42)

    @property
    def completed_tasks(self) -> List[Task]:
        return [t for t in self.tasks.values() if t.is_completed]

    @property
    def overdue_tasks(self) -> List[Task]:
        return [t for t in self.tasks.values() if t.is_overdue]

    @property
    def pending_tasks(self) -> List[Task]:
        return [t for t in self.tasks.values() if not t.is_completed and not t.is_overdue]

    @property
    def urgent_tasks(self) -> List[Task]:
        return [
            t for t in self.pending_tasks
            if (t.deadline_day - self.current_day) <= 2
        ]
