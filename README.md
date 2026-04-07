---
title: Student Task Manager Environment
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - scheduling
  - academic
pinned: false
---

# Student Task Manager Environment

> **A production-grade OpenEnv reinforcement learning environment** simulating a real-world academic workflow where an AI agent helps a student manage assignments, prioritize work, and schedule tasks efficiently.

---


## 🎯 Motivation

Students face constant pressure from overlapping deadlines, varying task difficulties, and limited daily study time. Poor prioritization leads to last-minute cramming, missed deadlines, and burnout.

This environment models that challenge precisely — with a dense reward system, task dependencies, procrastination penalties, and a deterministic grader — making it ideal for training AI scheduling agents.

---

## 📁 Project Structure

```
.
├── env/
│   ├── __init__.py
│   ├── models.py          # Pydantic models: Task, Action, Observation, State
│   ├── environment.py     # Core environment (reset/step/state)
│   ├── tasks.py           # EASY / MEDIUM / HARD task scenarios
│   └── grader.py          # Deterministic episode grader
├── main.py              # FastAPI REST API server
├── inference.py           # Baseline LLM inference script
├── openenv.yaml           # OpenEnv specification
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚙️ Environment Scenarios

| Scenario | Tasks | Days | Hours/Day | Max Steps | Description |
|----------|-------|------|-----------|-----------|-------------|
| `easy`   | 4     | 8    | 8.0       | 60        | Independent tasks, comfortable deadlines |
| `medium` | 6     | 9    | 7.0       | 100       | Overlapping deadlines, requires prioritization |
| `hard`   | 8     | 10   | 6.0       | 150       | Dependencies + tight deadlines + limited hours |

---

## 🎮 Action Space

| Action             | Required Fields         | Optional Fields | Description |
|--------------------|------------------------|-----------------|-------------|
| `select_task`      | `task_id`              | –               | Set active task |
| `allocate_time`    | `task_id`, `hours`     | –               | Work on a task (advances progress) |
| `mark_complete`    | `task_id`              | –               | Mark task done (requires ≥80% progress) |
| `skip_day`         | –                      | –               | Advance to next day (idle penalty) |
| `reorder_priority` | `priority_order` (list)| –               | Reorder task queue |

### Example Actions

```json
{"action_type": "allocate_time", "task_id": "M1", "hours": 2.5}
{"action_type": "mark_complete", "task_id": "H3"}
{"action_type": "reorder_priority", "priority_order": ["M1", "M3", "M2"]}
{"action_type": "skip_day"}
```

---

## 👀 Observation Space

```json
{
  "current_day": 2,
  "total_days": 9,
  "remaining_time_today": 5.0,
  "max_hours_per_day": 7.0,
  "tasks": [
    {
      "task_id": "M1",
      "subject": "Computer Science",
      "title": "Sorting Algorithms Implementation",
      "deadline_day": 4,
      "difficulty": 3,
      "importance": 5,
      "progress": 50.0,
      "remaining_hours": 2.0,
      "is_completed": false,
      "is_overdue": false,
      "has_unmet_dependencies": false,
      "days_until_deadline": 2
    }
  ],
  "completed_task_ids": [],
  "overdue_task_ids": [],
  "pending_task_ids": ["M1", "M2", "M3"],
  "urgent_tasks": ["M1"],
  "active_task_id": "M1",
  "episode_score_so_far": 0.45,
  "steps_taken": 5,
  "max_steps": 100,
  "cumulative_reward": 2.25,
  "last_action_status": "success",
  "last_action_message": "Allocated 2.00h to 'M1'. Progress: 50.0%."
}
```

---

## 🏆 Reward Function

| Signal | Value | Trigger |
|--------|-------|---------|
| Task completion | +0.40 | When a task is fully completed |
| Early completion bonus | +0.00 to +0.15 | Days before deadline |
| Progress reward | +0.025/hour | Each hour of productive work |
| Priority alignment | +0.00 to +0.10 | Working on high-importance/urgent tasks |
| Deadline miss penalty | −0.30 | Task deadline passes without completion |
| Idle penalty | −0.05 | `skip_day` action used |
| Dependency violation | −0.10 | Working on a task with unmet prerequisites |
| Low-priority penalty | −0.02/hour | Working on low-priority while urgent tasks exist |

All per-step rewards are clamped to **[0.0, 1.0]**.

---

## 📊 Grader

The deterministic grader evaluates the full episode and returns a score in **[0.0, 1.0]**:

| Component | Weight | Description |
|-----------|--------|-------------|
| Completion rate | 40% | Fraction of tasks fully completed (partial credit for progress) |
| Deadline adherence | 30% | How promptly tasks were finished relative to deadlines |
| Scheduling efficiency | 20% | Hour utilization + priority alignment |
| Dependency adherence | 10% | Penalty for dependency violations |

---

## 🛑 Termination Conditions

- ✅ All tasks completed → `done = True`
- ⏱️ `max_steps` reached → `done = True`  
- 📅 `max_days` exceeded → `done = True`
- ⚠️ >50% of tasks overdue → `done = True`

---

## 🚀 Setup & Running Locally

### Prerequisites

- Python 3.11+

### Install

```bash
pip install -r requirements.txt
touch env/__init__.py
```

### Start the API Server

```bash
python main.py
# Or with uvicorn directly:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Run the Inference Script

```bash
# 1. Start the environment server (in a separate terminal)
python main.py

# 2. Run the baseline inference against all 3 tasks
export API_BASE_URL=https://api.openai.com/v1   # LLM endpoint (OpenAI-compatible)
export MODEL_NAME=gpt-4o-mini                   # Model identifier
export HF_TOKEN=your_api_key_here               # HF token or OpenAI key
export ENV_SERVER_URL=http://localhost:8000      # Environment server URL

python inference.py                             # Runs easy + medium + hard
python inference.py --scenario medium           # Single scenario
python inference.py --scenario hard --seed 123  # Custom seed
```

### Interactive API Docs

Visit `http://localhost:8000/docs` for the Swagger UI.

---

## 🐳 Docker

### Build

```bash
docker build -t student-task-manager .
```

### Run

```bash
docker run -p 8000:8000 \
  -e ENV_SCENARIO=medium \
  -e ENV_SEED=42 \
  student-task-manager
```

### With Custom Scenario and LLM

```bash
docker run -p 8000:8000 -e ENV_SCENARIO=hard student-task-manager

# Then run inference against it:
export ENV_SERVER_URL=http://localhost:8000
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=sk-...
python inference.py
```

---

## ☁️ HuggingFace Space Deployment

This environment is designed to run as a containerized HuggingFace Space.

### `README.md` frontmatter (add to top for HF Spaces):

```yaml
---
title: Student Task Manager Environment
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---
```

The Dockerfile exposes port `8000` and responds to `/reset` and `/step` per OpenEnv spec.

### Using with HF-hosted LLMs

```bash
# The environment server runs on HF Space (docker)
# The inference script calls the LLM separately
export ENV_SERVER_URL=https://your-space.hf.space  # Environment server on HF
export API_BASE_URL=https://router.huggingface.co/v1  # LLM API endpoint
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=hf_...

python inference.py
```

---

## 🔌 REST API Reference

### `POST /reset`

```json
// Request
{"scenario": "medium", "seed": 42}

// Response
{
  "observation": { ... },
  "message": "Environment reset successfully."
}
```

### `POST /step`

```json
// Request
{
  "action": {
    "action_type": "allocate_time",
    "task_id": "M1",
    "hours": 2.0
  }
}

// Response
{
  "observation": { ... },
  "reward": 0.1125,
  "reward_breakdown": { ... },
  "done": false,
  "info": { ... }
}
```

### `GET /state`

Returns full serialized internal state.

### `GET /health`

```json
{"status": "ok", "service": "student-task-manager-env"}
```

---

## 📈 Baseline Scores

| Scenario | Random Agent | Greedy (Deadline) | LLM Agent (GPT-4o-mini) |
|----------|-------------|-------------------|--------------------------|
| Easy     | ~0.30       | ~0.65             | ~0.81                    |
| Medium   | ~0.22       | ~0.55             | ~0.73                    |
| Hard     | ~0.15       | ~0.45             | ~0.62                    |

*Scores are approximate and depend on model quality and available time.*

---

## ⚡ Performance Constraints

- **Runtime**: < 20 minutes per full episode
- **Memory**: < 512 MB (well within 8 GB limit)
- **Deterministic**: Same seed → same tasks → same score

---

## 🔁 Reproducibility

All scenarios use a fixed default seed (`42`). Given the same seed and scenario:
- Task set is identical
- Grading is deterministic
- Episode outcomes are fully reproducible

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
