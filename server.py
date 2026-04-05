"""
FastAPI REST API server for the Student Task Manager Environment.

Exposes:
  POST /reset  – reset the environment
  POST /step   – execute one action
  GET  /state  – get full internal state
  GET  /health – health check

Compatible with OpenEnv API conventions.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.environment import StudentTaskManagerEnv
from env.models import Action

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

@app.get("/health")
def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "student-task-manager-env"}


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

    return StepResponse(
        observation=obs.model_dump(),
        reward=reward_breakdown.total,
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
            "score_range": [0.0, 1.0],
            "total_days": cfg["total_days"],
            "max_hours_per_day": cfg["max_hours_per_day"],
            "max_steps": cfg["max_steps"],
            "task_count": len(cfg["tasks"]),
        })
    return {"tasks": tasks, "count": len(tasks)}


@app.get("/info")
def env_info() -> Dict[str, Any]:
    """Return environment metadata (name, version, description, tasks)."""
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
        "reward_range": [-0.30, 1.0],
        "score_range": [0.0, 1.0],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("server:app", host=host, port=port, reload=False)
