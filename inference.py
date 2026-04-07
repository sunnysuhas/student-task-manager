"""
Baseline Inference Script — Student Task Manager Environment
============================================================

MANDATORY ENVIRONMENT VARIABLES:
  API_BASE_URL   – LLM API endpoint (OpenAI-compatible)
  MODEL_NAME     – Model identifier for LLM inference
  HF_TOKEN       – HuggingFace / API key
  ENV_SERVER_URL – Environment server base URL (default: http://localhost:8000)

STDOUT FORMAT (strictly followed):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests

try:
    import openai
    from openai import OpenAI
except ImportError:
    print("openai package not installed. Run: pip install openai", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# LLM endpoint — used by the OpenAI client (OpenAI-compatible)
# Default points to HF inference router. Override with API_BASE_URL env var.
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN")

# Environment server — separate from LLM endpoint
ENV_SERVER_URL: str = os.environ.get("ENV_SERVER_URL", "http://localhost:8000")

BENCHMARK: str = "student-task-manager"
SEED: int = int(os.environ.get("ENV_SEED", "42"))
MAX_RETRIES: int = 5
STEP_DELAY: float = 12.5   # 12.5s delay fits 5 requests per minute (Gemini free tier)
SUCCESS_SCORE_THRESHOLD: float = 0.4  # score >= this → success

# Build OpenAI client for LLM calls
llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def robust_request(method: str, url: str, **kwargs) -> requests.Response:
    """Perform a request with retries and exponential backoff."""
    retries = 5
    backoff = 2.0
    for i in range(retries):
        try:
            resp = requests.request(method, url, **kwargs)
            if resp.status_code == 503:  # HF Space waking up
                raise requests.exceptions.RequestException("Space is starting up...")
            resp.raise_for_status()
            return resp
        except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as e:
            if i == retries - 1:
                raise
            sleep_time = backoff * (2 ** i)
            print(f"  [DEBUG] API call failed ({e}). Retrying in {sleep_time}s... ({i+1}/{retries})", file=sys.stderr)
            time.sleep(sleep_time)
    raise RuntimeError("Request failed after retries")


# ---------------------------------------------------------------------------
# Mandatory stdout log helpers (exact format required by evaluator)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# Environment API helpers
# ---------------------------------------------------------------------------

def env_reset(scenario: str, seed: int = SEED) -> Dict[str, Any]:
    """POST /reset on the environment server."""
    resp = robust_request(
        "POST",
        f"{ENV_SERVER_URL}/reset",
        json={"scenario": scenario, "seed": seed},
        timeout=45,
    )
    return resp.json()


def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    """POST /step on the environment server."""
    resp = robust_request(
        "POST",
        f"{ENV_SERVER_URL}/step",
        json={"action": action},
        timeout=45,
    )
    return resp.json()


# ---------------------------------------------------------------------------
# LLM Agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert AI academic scheduler. Your goal is to help a student
complete all their assignments as efficiently as possible, respecting deadlines and dependencies.

You operate in a simulated environment. Each turn you see the current state and must output
ONE action as valid JSON.

AVAILABLE ACTIONS:
1. select_task       – {"action_type": "select_task", "task_id": "X"}
2. allocate_time     – {"action_type": "allocate_time", "task_id": "X", "hours": N}
3. mark_complete     – {"action_type": "mark_complete", "task_id": "X"}
4. skip_day          – {"action_type": "skip_day"}
5. reorder_priority  – {"action_type": "reorder_priority", "priority_order": ["X", "Y", ...]}

STRATEGY GUIDELINES:
- Always check task dependencies before allocating time  
- Prioritize tasks with closest deadlines and highest importance
- Never allocate more hours than available today
- Mark tasks complete when progress >= 80%
- Only use skip_day as a last resort

FORMATTING:
- You MUST output ONLY raw JSON. No conversational text.
- Do not add explanations.
- If you cannot use a tool, use {"action_type": "skip_day"}.

Example: {"action_type": "allocate_time", "task_id": "M1", "hours": 2.0}
"""


def clean_json_response(content: str) -> str:
    """Extract raw JSON from common verbose LLM outputs."""
    content = content.replace("```json", "").replace("```", "").strip()
    # Find the first { and last } to handle trailing filler
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1:
        return content[start:end+1]
    return content


def build_user_prompt(obs: Dict[str, Any], step_num: int) -> str:
    tasks = obs.get("tasks", [])
    pending = [t for t in tasks if not t["is_completed"] and not t["is_overdue"]]
    completed = [t["task_id"] for t in tasks if t["is_completed"]]
    overdue = [t["task_id"] for t in tasks if t["is_overdue"]]
    urgent = obs.get("urgent_tasks", [])

    task_lines = []
    for t in pending:
        dep_flag = " [BLOCKED]" if t["has_unmet_dependencies"] else ""
        urgent_flag = " [URGENT]" if t["task_id"] in urgent else ""
        task_lines.append(
            f"  - {t['task_id']}: {t['title']} ({t['subject']}) | "
            f"Deadline: Day {t['deadline_day']} ({t['days_until_deadline']}d left) | "
            f"Progress: {t['progress']}% | Remaining: {t['remaining_hours']}h | "
            f"Difficulty: {t['difficulty']}/5 | Importance: {t['importance']}/5"
            f"{dep_flag}{urgent_flag}"
        )

    return (
        f"STEP {step_num}\n"
        f"Day: {obs['current_day']}/{obs['total_days']}\n"
        f"Time remaining today: {obs['remaining_time_today']}h / {obs['max_hours_per_day']}h\n"
        f"Active task: {obs.get('active_task_id') or 'None'}\n\n"
        f"PENDING TASKS:\n"
        f"{chr(10).join(task_lines) if task_lines else '  (none)'}\n\n"
        f"COMPLETED: {completed}\n"
        f"OVERDUE: {overdue}\n"
        f"URGENT (<=2 days): {urgent}\n\n"
        f"Episode score so far: {obs['episode_score_so_far']:.3f}\n"
        f"Last action: {obs.get('last_action_status', 'none')} - {obs.get('last_action_message', '')}\n\n"
        f"Output your next action as JSON:"
    )


def get_llm_action(
    conversation: List[Dict[str, str]], obs: Dict[str, Any], step_num: int
) -> tuple[Dict[str, Any], Optional[str]]:
    """Query LLM for next action with retry logic. Returns (action, error_str)."""
    conversation.append({"role": "user", "content": build_user_prompt(obs, step_num)})
    last_error: Optional[str] = None

    for attempt in range(MAX_RETRIES):
        try:
            response = llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=conversation,
                temperature=0.2,
                max_tokens=256,
            )
            content = (response.choices[0].message.content or "").strip()
            content = clean_json_response(content)
            
            action = json.loads(content)
            conversation.append({"role": "assistant", "content": json.dumps(action)})
            return action, None
        except openai.RateLimitError as e:
            last_error = f"Rate limit reached: {e}"
            # Extract retry-after from error or default to 30s
            wait = 30.0
            print(f"  [DEBUG] Rate limit (429). Waiting {wait}s before retry {attempt+1}/{MAX_RETRIES}...", file=sys.stderr)
            time.sleep(wait)
        except json.JSONDecodeError as e:
            last_error = f"JSON parse error: {e}"
            print(f"  [DEBUG] LLM returned invalid JSON (attempt {attempt+1}): {e}", file=sys.stderr)
            time.sleep(1.0)
        except Exception as e:
            last_error = str(e)
            print(f"  [DEBUG] LLM API error (attempt {attempt+1}): {e}", file=sys.stderr)
            time.sleep(2.0)

    fallback = {"action_type": "skip_day"}
    conversation.append({"role": "assistant", "content": json.dumps(fallback)})
    return fallback, last_error


# ---------------------------------------------------------------------------
# Single task (scenario) inference loop
# ---------------------------------------------------------------------------

def run_task(scenario: str, seed: int = SEED) -> float:
    """
    Run one full episode for the given scenario/task.
    Emits mandatory [START], [STEP]*, [END] lines.
    [END] is ALWAYS emitted, even on exception (per spec).
    Returns final score in [0, 1].
    """
    log_start(task=scenario, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        # Reset environment
        try:
            reset_resp = env_reset(scenario=scenario, seed=seed)
        except Exception as e:
            print(f"[DEBUG] env_reset failed: {e}", file=sys.stderr)
            # [END] emitted in finally
            return 0.0

        obs = reset_resp["observation"]
        max_steps = obs.get("max_steps", 150)

        # Initialize LLM conversation
        conversation: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        done = False
        info: Dict[str, Any] = {}

        for step_num in range(1, max_steps + 1):
            if done:
                break

            # Get LLM action
            action, llm_error = get_llm_action(conversation, obs, step_num)
            action_str = action.get("action_type", "unknown")
            if action.get("task_id"):
                action_str += f"({action['task_id']})"
            if action.get("hours") is not None:
                action_str += f",hours={action['hours']}"

            # Execute action in environment
            step_error: Optional[str] = llm_error
            try:
                step_resp = env_step(action)
                obs = step_resp["observation"]
                reward = float(step_resp["reward"])
                done = bool(step_resp["done"])
                info = step_resp.get("info", {})
                action_status = info.get("last_action_status", "")
                if action_status in ("error", "invalid", "dependency_blocked"):
                    step_error = info.get("last_action_message", action_status)
            except Exception as e:
                step_error = str(e)
                reward = 0.0
                done = True  # Can't continue without env response

            rewards.append(reward)
            steps_taken = step_num

            log_step(step=step_num, action=action_str, reward=reward, done=done, error=step_error)

            if done:
                # Extract final score from episode_grade (set by grader at termination)
                if "episode_grade" in info:
                    score = float(info["episode_grade"].get("score", 0.0))
                elif "final_score" in info:
                    score = float(info["final_score"])
                else:
                    score = float(obs.get("episode_score_so_far", 0.0))
                break

        # If loop ended without done
        if not done:
            score = float(obs.get("episode_score_so_far", 0.0))

        score = max(0.0, min(1.0, score))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Unexpected error in run_task: {exc}", file=sys.stderr)

    finally:
        # Spec: [END] is always emitted, even on exception
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    time.sleep(STEP_DELAY)
    return score


# ---------------------------------------------------------------------------
# Entry point — run all 3 tasks
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Student Task Manager – Baseline Inference")
    parser.add_argument(
        "--scenario",
        default=None,
        choices=["easy", "medium", "hard"],
        help="Run a single scenario. Omit to run all 3.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    tasks_to_run = [args.scenario] if args.scenario else ["easy", "medium", "hard"]
    all_scores: List[float] = []

    for task in tasks_to_run:
        s = run_task(scenario=task, seed=args.seed)
        all_scores.append(s)

    # Summary to stderr so it doesn't pollute evaluator stdout
    print("\n--- Baseline Summary ---", file=sys.stderr)
    for task, sc in zip(tasks_to_run, all_scores):
        print(f"  {task:8s}: {sc:.4f}", file=sys.stderr)
    print(f"  Average : {sum(all_scores)/len(all_scores):.4f}", file=sys.stderr)

    sys.exit(0)
