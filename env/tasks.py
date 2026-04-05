"""
Task definitions for the Student Task Manager Environment.

Provides three difficulty levels (EASY, MEDIUM, HARD) with a fixed seed
to ensure fully deterministic, reproducible task sets.
"""

from __future__ import annotations

from typing import Dict, List
from env.models import Task


# ---------------------------------------------------------------------------
# EASY scenario – 4 independent tasks, comfortable deadlines
# ---------------------------------------------------------------------------

EASY_TASKS: List[Task] = [
    Task(
        task_id="E1",
        subject="Mathematics",
        title="Chapter 3 Problem Set",
        deadline_day=5,
        estimated_duration=2.0,
        difficulty=2,
        importance=3,
        dependencies=[],
        progress=0.0,
    ),
    Task(
        task_id="E2",
        subject="History",
        title="Essay: Causes of WWI",
        deadline_day=7,
        estimated_duration=3.0,
        difficulty=2,
        importance=3,
        dependencies=[],
        progress=0.0,
    ),
    Task(
        task_id="E3",
        subject="Biology",
        title="Lab Report – Cell Division",
        deadline_day=6,
        estimated_duration=1.5,
        difficulty=1,
        importance=2,
        dependencies=[],
        progress=0.0,
    ),
    Task(
        task_id="E4",
        subject="English",
        title="Reading Summary: Chapter 1-3",
        deadline_day=4,
        estimated_duration=1.0,
        difficulty=1,
        importance=2,
        dependencies=[],
        progress=0.0,
    ),
]


# ---------------------------------------------------------------------------
# MEDIUM scenario – overlapping deadlines, requires prioritization
# ---------------------------------------------------------------------------

MEDIUM_TASKS: List[Task] = [
    Task(
        task_id="M1",
        subject="Computer Science",
        title="Sorting Algorithms Implementation",
        deadline_day=4,
        estimated_duration=4.0,
        difficulty=3,
        importance=5,
        dependencies=[],
        progress=0.0,
    ),
    Task(
        task_id="M2",
        subject="Physics",
        title="Thermodynamics Problem Set",
        deadline_day=4,
        estimated_duration=3.0,
        difficulty=4,
        importance=4,
        dependencies=[],
        progress=0.0,
    ),
    Task(
        task_id="M3",
        subject="Chemistry",
        title="Organic Reactions Worksheet",
        deadline_day=5,
        estimated_duration=2.5,
        difficulty=3,
        importance=3,
        dependencies=[],
        progress=0.0,
    ),
    Task(
        task_id="M4",
        subject="Mathematics",
        title="Calculus Integration Test Prep",
        deadline_day=6,
        estimated_duration=5.0,
        difficulty=4,
        importance=5,
        dependencies=[],
        progress=0.0,
    ),
    Task(
        task_id="M5",
        subject="English",
        title="Literary Analysis Essay",
        deadline_day=5,
        estimated_duration=3.5,
        difficulty=3,
        importance=4,
        dependencies=[],
        progress=0.0,
    ),
    Task(
        task_id="M6",
        subject="History",
        title="Research Presentation Slides",
        deadline_day=7,
        estimated_duration=2.0,
        difficulty=2,
        importance=3,
        dependencies=[],
        progress=0.0,
    ),
]


# ---------------------------------------------------------------------------
# HARD scenario – dependencies + tight deadlines + limited time
# ---------------------------------------------------------------------------

HARD_TASKS: List[Task] = [
    Task(
        task_id="H1",
        subject="Computer Science",
        title="Database Schema Design",
        deadline_day=3,
        estimated_duration=3.0,
        difficulty=4,
        importance=5,
        dependencies=[],
        progress=0.0,
    ),
    Task(
        task_id="H2",
        subject="Computer Science",
        title="Backend API Implementation",
        deadline_day=5,
        estimated_duration=5.0,
        difficulty=5,
        importance=5,
        dependencies=["H1"],  # Requires schema first
        progress=0.0,
    ),
    Task(
        task_id="H3",
        subject="Computer Science",
        title="Frontend Integration",
        deadline_day=7,
        estimated_duration=4.0,
        difficulty=4,
        importance=5,
        dependencies=["H2"],  # Requires backend
        progress=0.0,
    ),
    Task(
        task_id="H4",
        subject="Mathematics",
        title="Linear Algebra Final Prep",
        deadline_day=3,
        estimated_duration=4.0,
        difficulty=5,
        importance=5,
        dependencies=[],
        progress=0.0,
    ),
    Task(
        task_id="H5",
        subject="Physics",
        title="Quantum Mechanics Problem Set",
        deadline_day=4,
        estimated_duration=3.5,
        difficulty=5,
        importance=4,
        dependencies=["H4"],  # Physics builds on math
        progress=0.0,
    ),
    Task(
        task_id="H6",
        subject="Chemistry",
        title="Lab Report: Spectroscopy",
        deadline_day=4,
        estimated_duration=2.5,
        difficulty=3,
        importance=4,
        dependencies=[],
        progress=0.0,
    ),
    Task(
        task_id="H7",
        subject="English",
        title="Capstone Research Paper – Draft",
        deadline_day=6,
        estimated_duration=6.0,
        difficulty=4,
        importance=5,
        dependencies=[],
        progress=0.0,
    ),
    Task(
        task_id="H8",
        subject="English",
        title="Capstone Research Paper – Final Edit",
        deadline_day=8,
        estimated_duration=2.0,
        difficulty=3,
        importance=5,
        dependencies=["H7"],  # Needs draft first
        progress=0.0,
    ),
]


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

SCENARIO_CONFIGS: Dict[str, Dict] = {
    "easy": {
        "tasks": EASY_TASKS,
        "total_days": 8,
        "max_hours_per_day": 8.0,
        "max_steps": 60,
        "description": "4 independent tasks with comfortable deadlines and no dependencies.",
    },
    "medium": {
        "tasks": MEDIUM_TASKS,
        "total_days": 9,
        "max_hours_per_day": 7.0,
        "max_steps": 100,
        "description": "6 tasks with overlapping deadlines requiring careful prioritization.",
    },
    "hard": {
        "tasks": HARD_TASKS,
        "total_days": 10,
        "max_hours_per_day": 6.0,
        "max_steps": 150,
        "description": (
            "8 tasks with dependency chains, tight deadlines, and reduced daily hours. "
            "Requires optimal sequencing to avoid cascading failures."
        ),
    },
}


def get_scenario(scenario_name: str) -> Dict:
    """Return a deep copy of the requested scenario config."""
    import copy

    name = scenario_name.lower()
    if name not in SCENARIO_CONFIGS:
        raise ValueError(
            f"Unknown scenario '{scenario_name}'. Choose from: {list(SCENARIO_CONFIGS.keys())}"
        )
    cfg = copy.deepcopy(SCENARIO_CONFIGS[name])
    # Re-parse tasks to ensure fresh Pydantic instances
    cfg["tasks"] = [Task(**t.model_dump()) for t in cfg["tasks"]]
    return cfg
