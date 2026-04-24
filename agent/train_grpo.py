"""
MATPO GRPO Training Script
==========================
Phase 3 of the BlastRadius Reinforcement Learning Pipeline.

This script implements Group Relative Policy Optimization (GRPO) on a 
6GB VRAM constraint using Unsloth's integrated vLLM (`fast_inference=True`).

Memory Bottleneck Details (Option A + E Hybrid Strategy):
G=4 generations per prompt consumes ~1.8GB of KV Cache. We combine this
with 4-bit quantization, LoRA r=32, and 8-bit AdamW to squeeze the entire 
training loop into ~4.5GB VRAM, leaving 1.5GB of safety headroom.
On A100 80GB, pass --num_generations 8 and --gpu_memory_utilization 0.85
for significantly higher throughput.

Reward Functions (5 independent signals for robust RLVR):
1. `format_reward_func`:        MATPO dual-role XML tag compliance.
2. `environment_reward_func`:   TF-IDF semantic grader score from live env.
3. `action_validity_reward`:    Valid command names only — blocks hallucinated cmds.
4. `diagnosis_quality_reward`:  Structured root_cause + causal_chain validation.
5. `brevity_reward`:            Anti-padding / anti-dense-text hacking signal.
"""

import os
import sys
import argparse
import json
import re
from typing import List, Dict, Any
from pathlib import Path

from datasets import load_dataset
from transformers import TrainingArguments

try:
    from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
    # Patch TRL for ultra-fast/memory-optimized GRPO
    PatchFastRL("GRPO", FastLanguageModel)
except ImportError:
    print("Please install unsloth GRPO: pip install unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git")
    sys.exit(1)

from trl import GRPOConfig, GRPOTrainer

# Add project root to path to access the environment
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from incident_env.server.incident_environment import IncidentEnvironment
from incident_env.models import IncidentAction
from agent.prompts import (
    SCOUT_TAGS,
    COMMANDER_TAGS,
    THINK_TAGS,
    COMMANDER_SYSTEM_PROMPT,
)


# ─────────────────────────────────────────────────────────────
# Reward Functions (The RL Signal)
# ─────────────────────────────────────────────────────────────

def format_reward_func(completions: List[str], role: List[str], **kwargs) -> List[float]:
    """
    Rewards the model strictly if it followed the single-model dual-role
    formatting tags. We expect <think> tags for both, then <triage> for 
    the scout and <action> for the commander.

    FAILURE MODE 3 FIX (Lazy Likelihood Displacement):
    Broken formatting gets NEGATIVE reward, not zero. The model must
    *fear* format collapse, not merely fail to be rewarded for it.
    This prevents the silent killer where reward goes up but output
    quality degrades because format compliance drifts unchecked.
    """
    rewards = []
    for comp, current_role in zip(completions, role):
        reward = 0.0
        
        # 1. Did it think? — Mandatory for both roles
        if THINK_TAGS[0] in comp and THINK_TAGS[1] in comp:
            reward += 0.15
        else:
            # AGGRESSIVE: missing think tags = negative (FM3 fix)
            reward -= 0.3
            
        # 2. Did it use the correct role tag?
        if current_role == "scout":
            if SCOUT_TAGS[0] in comp and SCOUT_TAGS[1] in comp:
                reward += 0.75
            else:
                reward -= 0.5  # Hard penalty for breaking MATPO contract
        else:  # commander
            if COMMANDER_TAGS[0] in comp and COMMANDER_TAGS[1] in comp:
                reward += 0.5
                
                # 3. For commander, is the action parseable JSON?
                try:
                    action_text = comp.split(COMMANDER_TAGS[0])[1].split(COMMANDER_TAGS[1])[0].strip()
                    json.loads(action_text)
                    reward += 0.25  # Clean JSON bonus
                except Exception:
                    # AGGRESSIVE: broken JSON inside tags = negative (FM3 fix)
                    reward -= 0.5
            else:
                # AGGRESSIVE: no action tags at all = hard penalty
                reward -= 0.5
                
        rewards.append(reward)
    return rewards


def environment_reward_func(completions: List[str], role: List[str], task_id: List[str], step: List[int], history_log: List[List[str]], **kwargs) -> List[float]:
    """
    The main RL signal. We recreate the BlastRadius environment state 
    for each prompt, apply the model's generated action, and return 
    the exact TF-IDF / Anti-Cheat score from grader.py.

    CRITICAL: A fresh IncidentEnvironment is created for EACH completion
    to prevent internal reward accumulator leaking between G=4 rollouts
    on the same prompt (P1 BUG FIX).
    """
    rewards = []
    
    for comp, current_role, tid, current_step, history in zip(completions, role, task_id, step, history_log):
        # 1. Scout is evaluated on formatting only; environmental reward comes from Cmdr
        if current_role == "scout":
            rewards.append(0.0) # Format reward handles the scout's baseline
            continue
            
        # 2. Recreate environment state (fresh instance per completion to prevent leaking)
        env = IncidentEnvironment()
        try:
            env.reset(task_id=tid)
            # Fast-forward time (we skip actual execution logic and just pump the tick)
            # A true on-policy framework would run continuous episodes, but for
            # offline GRPO we simulate the time elapsed based on the step number.
            for _ in range(current_step - 1):
                # Access private attributes directly — the .state property returns
                # asdict() (a dict), not the IncidentState dataclass. (P0 BUG FIX)
                env._state.time_elapsed_minutes += 5
                env._graph.tick(5)
        except Exception as e:
            print(f"- Env reset failed for task_id={tid}: {e}")
            rewards.append(0.0)
            continue
            
        # 3. Parse action from completion
        try:
            action_text = comp.split(COMMANDER_TAGS[0])[1].split(COMMANDER_TAGS[1])[0].strip()
            # Handle markdown if the model hallucinates it
            if "```json" in action_text:
                action_text = action_text.replace("```json", "").replace("```", "").strip()
                
            action_dict = json.loads(action_text)
            action = IncidentAction(
                command=action_dict.get("command", "check_status"),
                # Coerce None to "" — IncidentAction.target is typed as str,
                # and passing None crashes _deobfuscate() (P0 BUG FIX)
                target=action_dict.get("target") or "",
                parameters=action_dict.get("parameters", {})
            )
        except Exception:
            # Complete failure to output action = big penalty
            rewards.append(-1.0)
            continue

        # 4. Execute action against Grader
        try:
            result = env.step(action)
            # The heart of the RL phase: we extract the reward exactly 
            # as calculated by the TF-IDF Grader overhaul.
            reward_val = result["reward"]

            # Small bonus if it resolved the incident
            info = result.get("info", {})
            if info.get("is_resolved", False):
                reward_val += 0.5

            # FAILURE MODE 2 FIX (Reward Hacking via GRPO Math):
            # Absolute reward floor. If the grader score is below 0.15,
            # the rollout is garbage — don't let GRPO learn from it.
            # Without this floor, GRPO's group-mean baseline will assign
            # positive advantage to the "least bad" failure in a batch
            # of all-failing rollouts, reinforcing bad behavior.
            if reward_val > 0 and reward_val < 0.15:
                reward_val = 0.0  # Floor: partial credit below 15% is noise
                
            rewards.append(reward_val)
        except Exception as e:
            rewards.append(0.0)

    return rewards


def action_validity_reward(
    completions: List[str], role: List[str], **kwargs
) -> List[float]:
    """
    Reward 3: Command Validity Gate.

    Rewards the agent for using a command that actually exists in the
    environment's VALID_COMMANDS set. Penalises hallucinated commands
    like 'fix_everything' or 'delete_service' which the model may invent.
    This closes a reward-hacking vector where the model outputs an invalid
    command string that looks plausible but cannot be executed.
    """
    from incident_env.models import VALID_COMMANDS

    rewards = []
    for comp, current_role in zip(completions, role):
        if current_role == "scout":
            rewards.append(0.0)
            continue
        try:
            action_text = comp.split(COMMANDER_TAGS[0])[1].split(COMMANDER_TAGS[1])[0].strip()
            if "```json" in action_text:
                action_text = action_text.replace("```json", "").replace("```", "").strip()
            action_dict = json.loads(action_text)
            cmd = action_dict.get("command", "").lower().strip()
            rewards.append(0.2 if cmd in VALID_COMMANDS else -0.3)
        except Exception:
            # Could not parse → treat as invalid command
            rewards.append(-0.1)
    return rewards


def diagnosis_quality_reward(
    completions: List[str], role: List[str], **kwargs
) -> List[float]:
    """
    Reward 4: Structured Diagnosis Quality.

    Specifically targets the `diagnose` command and rewards the agent for
    providing a well-structured diagnosis with:
    - a non-trivial `root_cause` string  (+0.30)
    - a `causal_chain` list with ≥2 entries (+0.20)
    - a numeric `confidence` value          (+0.10)

    Non-diagnose actions score 0.0 (neutral) so this doesn't interfere
    with investigation phases. This is a lightweight form of process
    supervision that rewards structured reasoning during diagnosis.
    """
    rewards = []
    for comp, current_role in zip(completions, role):
        if current_role == "scout":
            rewards.append(0.0)
            continue
        try:
            action_text = comp.split(COMMANDER_TAGS[0])[1].split(COMMANDER_TAGS[1])[0].strip()
            if "```json" in action_text:
                action_text = action_text.replace("```json", "").replace("```", "").strip()
            action_dict = json.loads(action_text)
            if action_dict.get("command") != "diagnose":
                rewards.append(0.0)
                continue
            params = action_dict.get("parameters", {})
            root_cause = params.get("root_cause", "")
            causal_chain = params.get("causal_chain", [])
            reward = 0.0
            if isinstance(root_cause, str) and len(root_cause.strip()) > 3:
                reward += 0.30
            if isinstance(causal_chain, list) and len(causal_chain) >= 2:
                reward += 0.20
            if isinstance(params.get("confidence"), (int, float)):
                reward += 0.10
            rewards.append(reward)
        except Exception:
            rewards.append(0.0)
    return rewards


def brevity_reward(completions: List[str], **kwargs) -> List[float]:
    """
    Reward 5: Anti-Padding / Anti-Dense-Text Signal.

    Targets the reward hacking vector where the model outputs an extremely
    long paragraph attempting to 'guess' every possible answer (mirroring
    the grader's dense-text penalty but at the generation level).

    Token thresholds (word-split estimate):
    - >400 words: strong penalty (-0.20) — clearly stuffing the context
    - >250 words: soft penalty  (-0.05) — verbose but not egregious
    - ≤250 words: small reward  (+0.10) — concise and on-task
    """
    rewards = []
    for comp in completions:
        word_count = len(comp.split())
        if word_count > 400:
            rewards.append(-0.20)
        elif word_count > 250:
            rewards.append(-0.05)
        else:
            rewards.append(0.10)
    return rewards


# ─────────────────────────────────────────────────────────────
# Preprocessing Dataset
# ─────────────────────────────────────────────────────────────

# Curriculum ordering: easy → medium → hard
# This ensures the model sees high-reward rollouts early and doesn't stall
# on zero-reward hard scenarios before it has learned basic task structure.
# Built dynamically from the scenario registry so it works with all 10 scenarios,
# not just the 3 core ones. Unknown task_ids default to rank 1 (medium).
_DIFFICULTY_RANK = {"easy": 0, "medium": 1, "hard": 2}
try:
    from incident_env.server.scenarios import SCENARIOS
    _DIFFICULTY_ORDER = {}
    for task_id, scenario_cls in SCENARIOS.items():
        # Instantiate to read the .difficulty property
        _diff = scenario_cls().difficulty
        _DIFFICULTY_ORDER[task_id] = _DIFFICULTY_RANK.get(_diff, 1)
except Exception:
    # Fallback if scenarios can't be imported (e.g., during unit tests with stubs)
    _DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}


def build_dataset_for_grpo(file_path: str):
    """
    GRPOTrainer expects a dataset with 'prompt' formatting string.
    We inject the role and task details into the dataset so the reward
    functions can read them.

    Curriculum sort is applied after mapping: easy → medium → hard.
    This guarantees the model sees achievable reward signals first,
    implementing the hackathon guide's §6 curriculum recommendation.
    """
    dataset = load_dataset("json", data_files=file_path, split="train")

    def process_row(example):
        # GRPOTrainer automatically formats lists of dicts using the chat template.
        # We only pass the user prompt; the trainer generates the completion.
        prompt = [
            {"role": "system", "content": example["system_prompt"]},
            {"role": "user", "content": example["user_prompt"]}
        ]

        # We infer history by splitting the user prompt (hacky but works for offline rl)
        history_log = []
        if "[EPISODE HISTORY]" in example["user_prompt"]:
            hist_block = example["user_prompt"].split("[EPISODE HISTORY]")[1].split("Based on")[0].strip()
            history_log = [line for line in hist_block.split("\n") if line]

        task_id = example.get("task_id", "easy")
        return {
            "prompt": prompt,
            "role": example.get("role", "commander"),
            "task_id": task_id,
            "step": example.get("step", 1),
            "history_log": history_log,
            # Curriculum rank column — used for sorting below
            "difficulty_rank": _DIFFICULTY_ORDER.get(task_id, 1),
        }

    dataset = dataset.map(process_row)

    # Curriculum sort: easy → medium → hard
    # Ensures the model sees achievable rewards first, preventing zero-reward stall.
    dataset = dataset.sort("difficulty_rank")

    return dataset


# ─────────────────────────────────────────────────────────────
# Training Routine
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MATPO GRPO Training using Unsloth")
    # Base model should be your output from train_sft.py
    parser.add_argument("--model", default="models/sft_checkpoint", help="Path to SFT model")
    parser.add_argument("--data", default="sft_data/expert_trajectories.jsonl", help="Path to offline rollouts")
    parser.add_argument("--output", default="models/grpo_checkpoint", help="Output directory")
    # A100 scaling args — increase these when running on 80GB VRAM
    parser.add_argument("--num_generations", type=int, default=4,
                        help="Rollouts per prompt. 4 for 6GB VRAM, 8 for A100 80GB.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90,
                        help="Fraction of GPU VRAM to allocate. 0.90 for 6GB, 0.85 for A100.")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  STAGE 3: MATPO-GRPO RL TRAINING (6GB BUDGET)")
    print(f"{'='*60}\n")
    
    # 1. Load Model with Colocated vLLM integration
    # This is the VRAM magic. It shares the model weights between training & generation.
    max_seq_length = 1024
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,         # ENABLES VLLM COLOCATION
        max_lora_rank=32,            # Must match PEFT rank below
        gpu_memory_utilization=args.gpu_memory_utilization,  # CLI-controlled: 0.90 for 6GB, 0.85 for A100
    )

    # 2. Attach LoRA for GRPO updates
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # 3. Configure GRPOTrainer (Strict memory constraints)
    # FAILURE MODE 1 FIX (Entropy Collapse):
    # - temperature=0.9 keeps exploration alive
    # - kl_coef (via beta) penalizes policy divergence
    # - cliprange limits per-step policy updates (PPO-style)
    # FAILURE MODE 4 FIX (OOM Mid-Training):
    # - max_grad_norm=1.0 prevents gradient explosion
    # - conservative defaults for num_generations and batch size
    training_args = GRPOConfig(
        use_vllm=True,                                      # Leverage integrated vLLM
        vllm_device="cuda:0",
        vllm_gpu_memory_utilization=0.50,                   # Split VRAM between vLLM & Trainer

        # Generation limits — controlled by CLI args for hardware portability
        num_generations=args.num_generations,               # 4 for 6GB, 8 for A100
        max_prompt_length=512,                  # Triage reports + JSON
        max_completion_length=512,              # Hard cap on response length (FM4)
        
        # ENTROPY COLLAPSE PREVENTION (FM1) —————————————————
        temperature=0.9,                        # Keep exploration alive (never < 0.7)
        # KL divergence coefficient: prevents policy from collapsing
        # to a narrow set of outputs. DO NOT remove this to "speed up"
        # training — that causes FM1.
        beta=0.05,                              # KL penalty coefficient
        
        # Optimizer limits
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,                     # RL requires lower LR
        optim="adamw_8bit",                     # Saves ~0.3GB VRAM
        max_grad_norm=1.0,                      # Gradient clipping (FM4 OOM prevention)
        
        # Training length
        num_train_epochs=2,
        logging_steps=5,
        save_steps=100,                         # Checkpoint every 100 steps for safety
        output_dir=args.output,
        
        # Ensure BFloat16 if supported
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
    )

    # 4. Load dataset and Train
    dataset = build_dataset_for_grpo(args.data)
    
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            format_reward_func,           # 1. XML tag compliance (aggressive FM3 penalties)
            environment_reward_func,      # 2. TF-IDF semantic env score (FM2 floor)
            action_validity_reward,       # 3. Valid command gate
            diagnosis_quality_reward,     # 4. Structured diagnosis validator
            brevity_reward,               # 5. Anti-padding / anti-dense-text
        ],
        args=training_args,
        train_dataset=dataset,
    )

    # FAILURE MODE 7 FIX (Not Watching Generations):
    # Register a callback that prints a real generation every 50 steps.
    # This is the #1 hackathon mistake — watching loss curves but never
    # looking at what the model is actually saying.
    from transformers import TrainerCallback

    class GenerationMonitorCallback(TrainerCallback):
        """Print a sample generation every N steps to catch drift early."""
        def __init__(self, model, tokenizer, interval=50):
            self._model = model
            self._tokenizer = tokenizer
            self._interval = interval

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % self._interval != 0 or state.global_step == 0:
                return
            try:
                # Use a fixed test prompt so we can track quality over time
                test_messages = [
                    {"role": "system", "content": COMMANDER_SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        "Step 3/20 | Last Reward: +0.05 | ⚠️ DIAGNOSE\n"
                        "[SCOUT TRIAGE REPORT]\n"
                        "database: DEGRADED (connection pool exhausted 100/100)\n"
                        "api-gateway: DEGRADED (upstream timeouts)\n\n"
                        "Based on the Scout's triage, choose your next action."
                    )},
                ]
                FastLanguageModel.for_inference(self._model)
                inputs = self._tokenizer.apply_chat_template(
                    test_messages, return_tensors="pt",
                    tokenize=True, add_generation_prompt=True
                ).to("cuda")
                import torch
                with torch.no_grad():
                    out = self._model.generate(
                        inputs, max_new_tokens=200, do_sample=False
                    )
                decoded = self._tokenizer.decode(
                    out[0][inputs.shape[-1]:], skip_special_tokens=True
                )
                print(f"\n{'='*60}")
                print(f"🔍 GENERATION SAMPLE @ step {state.global_step}")
                print(f"{'='*60}")
                print(decoded[:500])
                print(f"{'='*60}\n")
                FastLanguageModel.for_training(self._model)
            except Exception as e:
                print(f"[GEN MONITOR] Step {state.global_step}: {e}")

    trainer.add_callback(
        GenerationMonitorCallback(model, tokenizer, interval=50)
    )

    print("\nStarting GRPO Training...")
    print(f"  Generations per prompt: {args.num_generations}")
    print(f"  GPU memory utilization: {args.gpu_memory_utilization}")
    print(f"  Generation monitoring: every 50 steps")
    print(f"  Entropy collapse prevention: temperature=0.9, beta=0.05")
    print(f"  Reward floor: scores < 0.15 floored to 0.0")
    print("VRAM usage should peak at ~4.5GB. Generating rollout batches...")
    trainer.train()

    # 5. Save Finished Model (FM6: save as LoRA adapters, not merged)
    print(f"\nTraining Complete. Saving to {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

if __name__ == "__main__":
    main()
