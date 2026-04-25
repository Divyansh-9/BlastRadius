"""Functional tests for the GRPO reward functions without requiring GPU/Unsloth."""
import sys, types, importlib, importlib.util, builtins
sys.path.insert(0, '.')

# ── Stub out Unsloth + TRL + datasets so train_grpo.py can be imported on CPU ──
_real_import = builtins.__import__
def _mock_import(name, *args, **kwargs):
    if name in ('unsloth', 'datasets', 'transformers'):
        mod = types.ModuleType(name)
        if name == 'unsloth':
            mod.FastLanguageModel = None
            mod.PatchFastRL = lambda *a, **k: None
            mod.is_bfloat16_supported = lambda: False
        elif name == 'datasets':
            mod.load_dataset = lambda *a, **k: None
        elif name == 'transformers':
            mod.TrainingArguments = object
        return mod
    if name == 'trl':
        mod = types.ModuleType(name)
        mod.GRPOConfig = object
        mod.GRPOTrainer = object
        return mod
    return _real_import(name, *args, **kwargs)

builtins.__import__ = _mock_import
_real_exit = sys.exit
sys.exit = lambda *a: None

spec = importlib.util.spec_from_file_location('train_grpo', 'agent/train_grpo.py')
tg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tg)

builtins.__import__ = _real_import
sys.exit = _real_exit

# ── Test format_reward_func (the core formatting reward) ──
# Perfect commander output: <think> + <action>{valid JSON}
perfect_cmdr = '<think>analyzing</think><action>{"command": "check_status"}</action>'
r = tg.format_reward_func([perfect_cmdr], ['commander'])
assert r[0] > 0.5, f"Perfect commander should score > 0.5, got {r[0]}"
print(f"PASS  format_reward: perfect commander = {r[0]}")

# Commander with broken JSON inside tags: should be LOW (tags ok, json bad)
broken_json = '<think>analyzing</think><action>not json at all</action>'
r = tg.format_reward_func([broken_json], ['commander'])
assert r[0] <= 0.5, f"Broken JSON should score <= 0.5, got {r[0]}"
print(f"PASS  format_reward: broken JSON commander = {r[0]}")

# No tags at all (garbage output): should be strongly negative
garbage = 'I am just chatting, no tags anywhere'
r = tg.format_reward_func([garbage], ['commander'])
assert r[0] <= -0.5, f"Garbage output should be <= -0.5, got {r[0]}"
print(f"PASS  format_reward: garbage = {r[0]}")

# Perfect scout output
perfect_scout = '<think>triaging</think><triage>database is down</triage>'
r = tg.format_reward_func([perfect_scout], ['scout'])
assert r[0] > 0.5, f"Perfect scout should score > 0.5, got {r[0]}"
print(f"PASS  format_reward: perfect scout = {r[0]}")

# Scout with missing triage tags
bad_scout = '<think>triaging</think>just text no triage tags'
r = tg.format_reward_func([bad_scout], ['scout'])
assert r[0] < 0.5, f"Bad scout should score < 0.5, got {r[0]}"
print(f"PASS  format_reward: bad scout = {r[0]}")

# ── Test environment_reward_func exists and is callable ──
assert callable(tg.environment_reward_func), "environment_reward_func should be callable"
print("PASS  environment_reward_func is callable")

print()
print("=== ALL REWARD FUNCTION TESTS PASSED ===")
