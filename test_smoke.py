"""
Smoke test for the Student Task Manager Environment.
Run: python test_smoke.py
"""
import sys
sys.path.insert(0, ".")

from env.environment import StudentTaskManagerEnv
from env.models import Action, ActionType
from env.grader import grade

errors = []

def check(condition, msg):
    if not condition:
        errors.append(f"FAIL: {msg}")
        print(f"  ✗ {msg}")
    else:
        print(f"  ✓ {msg}")

# ─── Easy Scenario ────────────────────────────────────────────────────────
print("=== EASY Scenario ===")
env = StudentTaskManagerEnv(scenario="easy", seed=42)
obs = env.reset()
check(obs.current_day == 1, "Reset starts on day 1")
check(obs.total_tasks == 4, "Easy has 4 tasks")
check(obs.remaining_time_today == 8.0, "Easy has 8h/day")

# Allocate time to E4 (deadline day 4, 1.0h needed)
action1 = Action(action_type=ActionType.ALLOCATE_TIME, task_id="E4", hours=1.0)
obs, reward, done, info = env.step(action1)
check(reward.total > 0, f"Positive reward for valid allocation: {reward.total:.4f}")
check(info["last_action_status"] == "success", "Allocation succeeds")

# Mark E4 complete (progress should be 100%)
e4_task = next(t for t in obs.tasks if t.task_id == "E4")
check(e4_task.progress >= 99.0, f"E4 progress=100% after full allocation: {e4_task.progress}")

# Test invalid task
action_inv = {"action_type": "allocate_time", "task_id": "ZZZZ", "hours": 1.0}
obs, reward, done, info = env.step(action_inv)
check(reward.total == 0.0, "Invalid task gives 0 reward")
check(info["last_action_status"] == "error", "Invalid task gives error status")

# skip_day
action_skip = Action(action_type=ActionType.SKIP_DAY)
obs, reward, done, info = env.step(action_skip)
check(obs.current_day == 2, f"Day advanced after skip: {obs.current_day}")
check(reward.total < 0.5, f"Skip day incurs penalty: {reward.total:.4f}")

# State keys
state = env.state()
required_keys = ["current_day", "tasks", "steps_taken", "cumulative_reward", "max_steps", "total_days"]
for k in required_keys:
    check(k in state, f"State has key '{k}'")

# ─── Hard Scenario: Dependencies ──────────────────────────────────────────
print("\n=== HARD Scenario: Dependency Enforcement ===")
env2 = StudentTaskManagerEnv(scenario="hard", seed=42)
obs2 = env2.reset()
check(obs2.total_tasks == 8, "Hard has 8 tasks")

# H2 depends on H1 — should be blocked
dep_action = Action(action_type=ActionType.ALLOCATE_TIME, task_id="H2", hours=2.0)
obs2, reward2, done2, info2 = env2.step(dep_action)
check(info2["last_action_status"] == "dependency_blocked", "H2 blocked before H1 done")
check(reward2.total < 0.5, f"Dep violation penalty applied: {reward2.total:.4f}")

# Complete H1
h1_task = next(t for t in obs2.tasks if t.task_id == "H1")
action_h1 = Action(action_type=ActionType.ALLOCATE_TIME, task_id="H1", hours=3.0)
obs2, reward2, done2, info2 = env2.step(action_h1)
check(info2["last_action_status"] == "success", "H1 allocation succeeds")

# Now H2 should be unblocked
h1_now = next(t for t in obs2.tasks if t.task_id == "H1")
check(h1_now.progress >= 99.0, f"H1 at 100% progress: {h1_now.progress}")
obs2, reward2, done2, info2 = env2.step(
    Action(action_type=ActionType.ALLOCATE_TIME, task_id="H2", hours=2.0)
)
check(info2["last_action_status"] == "success", "H2 now accessible after H1 complete")

# ─── Medium Scenario: Reorder Priority ────────────────────────────────────
print("\n=== MEDIUM Scenario: Reorder Priority ===")
env3 = StudentTaskManagerEnv(scenario="medium", seed=42)
obs3 = env3.reset()
new_order = ["M4", "M1", "M2", "M5", "M3", "M6"]
action_reorder = Action(action_type=ActionType.REORDER_PRIORITY, priority_order=new_order)
obs3, reward3, done3, info3 = env3.step(action_reorder)
check(env3._internal_state.priority_order[:3] == ["M4", "M1", "M2"], "Priority reorder applied")

# ─── Grader Test ──────────────────────────────────────────────────────────
print("\n=== GRADER Test ===")
env4 = StudentTaskManagerEnv(scenario="easy", seed=42)
obs4 = env4.reset()
# Complete all tasks
task_ids = [t.task_id for t in obs4.tasks]
for tid in task_ids:
    t_info = next(t for t in obs4.tasks if t.task_id == tid)
    a = Action(action_type=ActionType.ALLOCATE_TIME, task_id=tid, hours=t_info.remaining_hours + 0.1)
    obs4, r, done4, info4 = env4.step(a)

state4 = env4._internal_state
result = grade(state4)
check(0.0 <= result["score"] <= 1.0, f"Grade in [0,1]: {result['score']}")
check("breakdown" in result, "Grade has breakdown")
check("completion" in result["breakdown"], "Breakdown has completion metric")
check(result["breakdown"]["completion"]["score"] > 0.5, f"Good completion score: {result['breakdown']['completion']['score']}")

# ─── Episode termination ──────────────────────────────────────────────────
print("\n=== Termination Test ===")
env5 = StudentTaskManagerEnv(scenario="easy", seed=42)
env5._internal_state.steps_taken = env5._internal_state.max_steps - 1
obs5, reward5, done5, info5 = env5.step(
    Action(action_type=ActionType.ALLOCATE_TIME, task_id="E1", hours=0.5)
)
check(done5 == True, f"Episode ends at max_steps: done={done5}")

# ─── Summary ──────────────────────────────────────────────────────────────
print("\n" + "="*50)
if errors:
    print(f"FAILED: {len(errors)} test(s) failed")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)
else:
    print(f"ALL TESTS PASSED ({len([x for x in open(__file__).read().split('check(') if x.strip()])-1} checks)")
    sys.exit(0)
