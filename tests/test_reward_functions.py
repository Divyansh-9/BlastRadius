"""Functional tests for the 5 GRPO reward functions without requiring GPU/Unsloth."""
import sys, types, importlib, importlib.util, builtins
sys.path.insert(0, '.')

# ── Stub out Unsloth + TRL so train_grpo.py can be imported on CPU ──
_real_import = builtins.__import__
def _mock_import(name, *args, **kwargs):
    if name == 'unsloth':
        mod = types.ModuleType(name)
        mod.FastLanguageModel = None
        mod.PatchFastRL = lambda *a, **k: None
        mod.is_bfloat16_supported = lambda: False
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

# ── Test curriculum constant ──
# Dynamic builder should include at least the 3 core scenarios
assert "easy" in tg._DIFFICULTY_ORDER and tg._DIFFICULTY_ORDER["easy"] == 0
assert "medium" in tg._DIFFICULTY_ORDER and tg._DIFFICULTY_ORDER["medium"] == 1
assert "hard" in tg._DIFFICULTY_ORDER and tg._DIFFICULTY_ORDER["hard"] == 2
# Should also include extended scenarios
assert len(tg._DIFFICULTY_ORDER) >= 3  # at minimum the 3 core ones
print("PASS  _DIFFICULTY_ORDER:", tg._DIFFICULTY_ORDER)

# ── Test action_validity_reward ──
valid_comp   = '<think>t</think><action>{"command": "check_status"}</action>'
invalid_comp = '<think>t</think><action>{"command": "hack_everything"}</action>'
r_valid   = tg.action_validity_reward([valid_comp],   ['commander'])
r_invalid = tg.action_validity_reward([invalid_comp], ['commander'])
r_scout   = tg.action_validity_reward([valid_comp],   ['scout'])
assert r_valid[0]   ==  0.2,  f"Expected 0.2 got {r_valid}"
assert r_invalid[0] == -0.3,  f"Expected -0.3 got {r_invalid}"
assert r_scout[0]   ==  0.0,  f"Expected 0.0 for scout got {r_scout}"
print("PASS  action_validity_reward: valid=0.2, invalid=-0.3, scout=0.0")

# ── Test diagnosis_quality_reward ──
import json
diag_full = json.dumps({
    "command": "diagnose",
    "parameters": {
        "root_cause": "database",
        "causal_chain": ["pool exhausted", "api timeouts"],
        "confidence": 0.9
    }
})
diag_comp = f"<think>t</think><action>{diag_full}</action>"
non_diag  = '<think>t</think><action>{"command": "check_status"}</action>'
r_diag     = tg.diagnosis_quality_reward([diag_comp], ['commander'])
r_non_diag = tg.diagnosis_quality_reward([non_diag],  ['commander'])
assert r_diag[0] == 0.60, f"Expected 0.60 got {r_diag}"
assert r_non_diag[0] == 0.0
print("PASS  diagnosis_quality_reward: full=0.60, non-diagnose=0.0")

# ── Test brevity_reward ──
long_comp  = ' '.join(['word'] * 500)
med_comp   = ' '.join(['word'] * 300)
short_comp = 'short output'
r_long  = tg.brevity_reward([long_comp])
r_med   = tg.brevity_reward([med_comp])
r_short = tg.brevity_reward([short_comp])
assert r_long[0]  == -0.20, f"Expected -0.20 got {r_long}"
assert r_med[0]   == -0.05, f"Expected -0.05 got {r_med}"
assert r_short[0] ==  0.10, f"Expected 0.10 got {r_short}"
print("PASS  brevity_reward: long=-0.20, medium=-0.05, short=+0.10")

# ── Test format_reward_func (FM3: aggressive penalties) ──
# Perfect commander output: <think> + <action>{valid JSON}
perfect_cmdr = '<think>analyzing</think><action>{"command": "check_status"}</action>'
r = tg.format_reward_func([perfect_cmdr], ['commander'])
assert r[0] > 0.5, f"Perfect commander should score > 0.5, got {r[0]}"

# Commander with broken JSON inside tags: should be LOW (tags ok, json bad)
broken_json = '<think>analyzing</think><action>not json at all</action>'
r = tg.format_reward_func([broken_json], ['commander'])
assert r[0] < 0.5, f"Broken JSON should score low (< 0.5), got {r[0]}"

# No tags at all (garbage output): should be strongly negative
garbage = 'I am just chatting, no tags anywhere'
r = tg.format_reward_func([garbage], ['commander'])
assert r[0] < -0.5, f"Garbage output should be < -0.5, got {r[0]}"

# Perfect scout output
perfect_scout = '<think>triaging</think><triage>database is down</triage>'
r = tg.format_reward_func([perfect_scout], ['scout'])
assert r[0] > 0.5, f"Perfect scout should score > 0.5, got {r[0]}"

print("PASS  format_reward_func: perfect=positive, broken_json=negative, garbage=very_negative")

print()
print("=== ALL 6 REWARD FUNCTION TESTS PASSED ===")
