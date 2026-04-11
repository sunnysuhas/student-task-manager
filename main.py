"""
FastAPI REST API server for the Student Task Manager Environment.

Exposes:
  POST /reset  – reset the environment
  POST /step   – execute one action
  GET  /state  – get full internal state
  GET  /health – health check
  GET  /metadata – environment metadata (required by openenv validate)
  GET  /schema   – action/observation/state schemas (required by openenv validate)

Compatible with OpenEnv API conventions.

Run with:
  python main.py
  uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.environment import StudentTaskManagerEnv
from env.models import Action, Observation, RewardBreakdown

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Student Task Manager – OpenEnv API",
    description=(
        "A production-grade reinforcement learning environment for academic task scheduling. "
        "An agent manages assignments, deadlines, dependencies, and limited daily work hours."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton environment (thread-safe for single-process deployments)
_env: Optional[StudentTaskManagerEnv] = None


def get_env() -> StudentTaskManagerEnv:
    global _env
    if _env is None:
        scenario = os.environ.get("ENV_SCENARIO", "medium")
        seed = int(os.environ.get("ENV_SEED", "42"))
        _env = StudentTaskManagerEnv(scenario=scenario, seed=seed)
    return _env


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    scenario: str = "medium"
    seed: int = 42


class StepRequest(BaseModel):
    action: Dict[str, Any]


class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    message: str = "Environment reset successfully."


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    reward_breakdown: Dict[str, Any]
    done: bool
    info: Dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root() -> Dict[str, str]:
    """Root landing page."""
    return {
        "message": "Student Task Manager Environment is running!",
        "documentation": "Visit /docs for the interactive API explorer",
        "status": "healthy"
    }


@app.get("/health")
def health() -> Dict[str, str]:
    """Health check endpoint. Returns status=healthy (required by openenv validate)."""
    return {"status": "healthy", "service": "student-task-manager-env"}


@app.get("/metadata")
def metadata() -> Dict[str, Any]:
    """
    Return environment metadata.
    Required by openenv validate — must include 'name' and 'description' fields.
    """
    return {
        "name": "student-task-manager",
        "version": "1.0.0",
        "description": (
            "A production-grade OpenEnv environment simulating a real academic workflow "
            "where an AI agent helps a student manage assignments, prioritize tasks, "
            "and schedule work efficiently across multiple days with limited daily hours."
        ),
        "benchmark": "student-task-manager",
        "tasks": ["easy", "medium", "hard"],
        "action_types": ["select_task", "allocate_time", "mark_complete", "skip_day", "reorder_priority"],
        "reward_type": "dense",
        "reward_range": [0.01, 0.99],
        "score_range": [0.01, 0.99],
        "tags": ["scheduling", "prioritization", "academic", "rl-environment"],
    }


@app.get("/schema")
def schema() -> Dict[str, Any]:
    """
    Return action, observation, and state JSON schemas.
    Required by openenv validate — must include 'action', 'observation', 'state' keys.
    """
    return {
        "action": {
            "type": "object",
            "description": "Agent action",
            "properties": {
                "action_type": {
                    "type": "string",
                    "enum": ["select_task", "allocate_time", "mark_complete", "skip_day", "reorder_priority"],
                    "description": "Type of action to perform",
                },
                "task_id": {"type": "string", "description": "Target task ID (optional)"},
                "hours": {"type": "number", "minimum": 0.1, "maximum": 8.0, "description": "Hours to allocate"},
                "priority_order": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ordered list of task IDs",
                },
            },
            "required": ["action_type"],
        },
        "observation": {
            "type": "object",
            "description": "Environment observation returned after each step",
            "properties": {
                "current_day": {"type": "integer"},
                "total_days": {"type": "integer"},
                "remaining_time_today": {"type": "number"},
                "max_hours_per_day": {"type": "number"},
                "tasks": {"type": "array", "items": {"type": "object"}},
                "completed_task_ids": {"type": "array", "items": {"type": "string"}},
                "overdue_task_ids": {"type": "array", "items": {"type": "string"}},
                "pending_task_ids": {"type": "array", "items": {"type": "string"}},
                "urgent_tasks": {"type": "array", "items": {"type": "string"}},
                "active_task_id": {"type": ["string", "null"]},
                "episode_score_so_far": {"type": "number", "minimum": 0.01, "maximum": 0.99},
                "steps_taken": {"type": "integer"},
                "max_steps": {"type": "integer"},
                "cumulative_reward": {"type": "number"},
                "last_action_status": {"type": "string"},
                "last_action_message": {"type": "string"},
            },
        },
        "state": {
            "type": "object",
            "description": "Full internal environment state",
            "properties": {
                "current_day": {"type": "integer"},
                "tasks": {"type": "object"},
                "steps_taken": {"type": "integer"},
                "cumulative_reward": {"type": "number"},
                "max_steps": {"type": "integer"},
                "total_days": {"type": "integer"},
                "episode_done": {"type": "boolean"},
                "action_history": {"type": "array"},
                "reward_history": {"type": "array"},
            },
        },
    }


@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest = ResetRequest()) -> ResetResponse:
    """
    Reset the environment.

    - **scenario**: one of `easy`, `medium`, `hard`
    - **seed**: random seed for reproducibility
    """
    global _env
    _env = StudentTaskManagerEnv(scenario=request.scenario, seed=request.seed)
    obs = _env.reset()
    return ResetResponse(observation=obs.model_dump())


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest) -> StepResponse:
    """
    Execute one action in the environment.

    Action format:
    ```json
    {
      "action": {
        "action_type": "allocate_time",
        "task_id": "M1",
        "hours": 2.0
      }
    }
    ```

    Valid action types: `select_task`, `allocate_time`, `mark_complete`, `skip_day`, `reorder_priority`
    """
    env = get_env()
    try:
        obs, reward_breakdown, done, info = env.step(request.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # ✅ ADDED: safe clamp at the API boundary — last line of defense
    safe_reward = max(1e-6, min(1.0 - 1e-6, reward_breakdown.total))

    # ✅ ADDED: clamp final_score inside info too
    if "final_score" in info:
        info["final_score"] = max(1e-6, min(1.0 - 1e-6, float(info["final_score"])))
    if "episode_grade" in info and "score" in info["episode_grade"]:
        info["episode_grade"]["score"] = max(0.01, min(0.99, float(info["episode_grade"]["score"])))

    return StepResponse(
        observation=obs.model_dump(),
        reward=safe_reward,
        reward_breakdown=reward_breakdown.model_dump(),
        done=done,
        info=info,
    )


@app.get("/state")
def state() -> Dict[str, Any]:
    """Return full serialized internal environment state."""
    env = get_env()
    return env.state()


@app.get("/scenarios")
def list_scenarios() -> Dict[str, Any]:
    """List available scenarios and their configurations."""
    from env.tasks import SCENARIO_CONFIGS
    return {
        name: {
            "description": cfg["description"],
            "total_days": cfg["total_days"],
            "max_hours_per_day": cfg["max_hours_per_day"],
            "max_steps": cfg["max_steps"],
            "task_count": len(cfg["tasks"]),
        }
        for name, cfg in SCENARIO_CONFIGS.items()
    }


@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    """
    Enumerate all task names and grader metadata.
    Required by the openenv validate command.
    Each task corresponds to a scenario (easy / medium / hard).
    """
    from env.tasks import SCENARIO_CONFIGS
    tasks = []
    for name, cfg in SCENARIO_CONFIGS.items():
        tasks.append({
            "task_id": name,
            "name": name,
            "description": cfg["description"],
            "difficulty": name,
            "grader": "deterministic",
            "score_range": [0.01, 0.99],
            "total_days": cfg["total_days"],
            "max_hours_per_day": cfg["max_hours_per_day"],
            "max_steps": cfg["max_steps"],
            "task_count": len(cfg["tasks"]),
        })
    return {"tasks": tasks, "count": len(tasks)}


@app.get("/info")
def env_info() -> Dict[str, Any]:
    """Alias for /metadata — returns environment metadata."""
    return metadata()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Start the uvicorn server. Used by openenv serve and pyproject.toml scripts."""
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
