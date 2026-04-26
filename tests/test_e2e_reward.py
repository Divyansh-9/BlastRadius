"""End-to-end test: simulates exactly what environment_reward_func does during GRPO training."""
import sys
import json
sys.path.insert(0, '.')

from incident_env.server.incident_environment import IncidentEnvironment
from incident_env.models import IncidentAction

COMMANDER_OPEN = "<action>"
COMMANDER_CLOSE = "</action>"

print("=== Simulating environment_reward_func for a batch of 4 completions ===")
print()

completions = [
    '<think>DB pool exhaustion</think><action>{"command": "check_logs", "target": "database"}</action>',
    '<think>Let me check status</think><action>{"command": "check_status"}</action>',
    '<think>Should diagnose</think><action>{"command": "diagnose", "parameters": {"root_cause": "database", "causal_chain": ["pool exhausted", "api timeouts"], "confidence": 0.9}}</action>',
    "garbage output with no tags",
]
roles = ["commander", "commander", "commander", "commander"]
task_ids = ["easy", "easy", "easy", "easy"]
steps = [3, 3, 3, 3]
histories: list[list[str]] = [[], [], [], []]

rewards = []
for i, (comp, role, tid, step, history) in enumerate(
    zip(completions, roles, task_ids, steps, histories)
):
    if role == "scout":
        rewards.append(0.0)
        continue

    # Fresh env per completion (the fix!)
    env = IncidentEnvironment()
    try:
        env.reset(task_id=tid)
        for _ in range(step - 1):
            env._state.time_elapsed_minutes += 5
            assert env._graph is not None
            env._graph.tick(5)
    except Exception as e:
        print(f"  Completion {i}: ENV RESET FAILED: {e}")
        rewards.append(0.0)
        continue

    try:
        action_text = comp.split(COMMANDER_OPEN)[1].split(COMMANDER_CLOSE)[0].strip()
        action_dict = json.loads(action_text)
        action = IncidentAction(
            command=action_dict.get("command", "check_status"),
            target=action_dict.get("target") or "",
            parameters=action_dict.get("parameters", {}),
        )
    except Exception:
        print(f"  Completion {i}: PARSE FAILED -> reward=-1.0")
        rewards.append(-1.0)
        continue

    try:
        result = env.step(action)
        r = result["reward"]
        info = result.get("info", {})
        if info.get("is_resolved", False):
            r += 0.5
        rewards.append(r)
        print(f"  Completion {i}: cmd={action_dict.get('command')} target={action_dict.get('target','')} -> reward={r:+.4f}")
    except Exception as e:
        print(f"  Completion {i}: STEP FAILED: {e}")
        rewards.append(0.0)

print()
print(f"Rewards for batch: {rewards}")
assert len(rewards) == 4, f"Expected 4 rewards, got {len(rewards)}"
assert all(isinstance(r, float) for r in rewards)
# Completion 3 (garbage) should have gotten -1.0
assert rewards[3] == -1.0, f"Expected garbage completion to get -1.0, got {rewards[3]}"
print()
print("=== ENVIRONMENT REWARD FUNCTION E2E TEST PASSED ===")
