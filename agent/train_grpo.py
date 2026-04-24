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

Reward Functions:
1. `format_reward_func`: Checks for adherence to MATPO dual-role tags.
2. `environment_reward_func`: Restores the episode state and scores the
   generated action using the exact semantic TF-IDF grader.py logic.
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
)


# ─────────────────────────────────────────────────────────────
# Reward Functions (The RL Signal)
# ─────────────────────────────────────────────────────────────

def format_reward_func(completions: List[str], role: List[str], **kwargs) -> List[float]:
    """
    Rewards the model strictly if it followed the single-model dual-role
    formatting tags. We expect <think> tags for both, then <triage> for 
    the scout and <action> for the commander.
    """
    rewards = []
    for comp, current_role in zip(completions, role):
        reward = 0.0
        
        # 1. Did it think?
        if THINK_TAGS[0] in comp and THINK_TAGS[1] in comp:
            reward += 0.25
            
        # 2. Did it use the correct role tag?
        if current_role == "scout":
            if SCOUT_TAGS[0] in comp and SCOUT_TAGS[1] in comp:
                reward += 0.75
            else:
                reward -= 0.5 # Penalty for breaking MATPO contract
        else: # commander
            if COMMANDER_TAGS[0] in comp and COMMANDER_TAGS[1] in comp:
                reward += 0.5
                
                # 3. For commander, is the action parseable JSON?
                action_text = ""
                try:
                    action_text = comp.split(COMMANDER_TAGS[0])[1].split(COMMANDER_TAGS[1])[0].strip()
                    json.loads(action_text)
                    reward += 0.25 # Clean JSON bonus
                except Exception:
                    reward -= 0.25 # Penalty for invalid JSON
            else:
                reward -= 0.5
                
        rewards.append(reward)
    return rewards


def environment_reward_func(completions: List[str], role: List[str], task_id: List[str], step: List[int], history_log: List[List[str]], **kwargs) -> List[float]:
    """
    The main RL signal. We recreate the BlastRadius environment state 
    for each prompt, apply the model's generated action, and return 
    the exact TF-IDF / Anti-Cheat score from grader.py.
    """
    rewards = []
    
    # Instantiate a clean environment pool
    env = IncidentEnvironment()
    
    for comp, current_role, tid, current_step, history in zip(completions, role, task_id, step, history_log):
        # 1. Scout is evaluated on formatting only; environmental reward comes from Cmdr
        if current_role == "scout":
            rewards.append(0.0) # Format reward handles the scout's baseline
            continue
            
        # 2. Recreate environment state
        try:
            env.reset(task_id=tid)
            # Fast-forward time (we skip actual execution logic and just pump the tick)
            # A true on-policy framework would run continuous episodes, but for
            # offline GRPO we simulate the time elapsed based on the step number.
            for _ in range(current_step - 1):
                env.state.time_elapsed_minutes += 5
                env.graph.tick(5)
        except Exception as e:
            print(f"- Env reset failed: {e}")
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
                target=action_dict.get("target"),
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
                
            rewards.append(reward_val)
        except Exception as e:
            rewards.append(0.0)

    return rewards


# ─────────────────────────────────────────────────────────────
# Preprocessing Dataset
# ─────────────────────────────────────────────────────────────

def build_dataset_for_grpo(file_path: str):
    """
    GRPOTrainer expects a dataset with 'prompt' formatting string.
    We inject the role and task details into the dataset so the reward
    functions can read them.
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
            
        return {
            "prompt": prompt,
            "role": example.get("role", "commander"),
            "task_id": example.get("task_id", "easy"),
            "step": example.get("step", 1),
            "history_log": history_log,
        }
        
    return dataset.map(process_row)


# ─────────────────────────────────────────────────────────────
# Training Routine
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MATPO GRPO Training using Unsloth")
    # Base model should be your output from train_sft.py
    parser.add_argument("--model", default="models/sft_checkpoint", help="Path to SFT model")
    parser.add_argument("--data", default="sft_data/expert_trajectories.jsonl", help="Path to offline rollouts")
    parser.add_argument("--output", default="models/grpo_checkpoint", help="Output directory")
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
        gpu_memory_utilization=0.90, # Auto-budget the 6GB VRAM
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
    training_args = GRPOConfig(
        use_vllm=True,                          # Leverage integrated vLLM
        vllm_device="cuda:0",
        vllm_gpu_memory_utilization=0.50,       # Split VRAM between vLLM & Trainer
        
        # Generation limits
        num_generations=4,                      # G=4. More = OOM on 6GB VRAM
        max_prompt_length=512,                  # Triage reports + JSON
        max_completion_length=512,              # Chain of thought length limit
        
        # Optimizer limits
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,                     # RL requires lower LR
        optim="adamw_8bit",                     # Saves ~0.3GB VRAM
        
        # Training length
        num_train_epochs=2,
        logging_steps=5,
        output_dir=args.output,
        
        # KL Divergence constraints to prevent reward hacking
        beta=0.04,
        
        # Ensure BFloat16 if supported
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
    )

    # 4. Load dataset and Train
    dataset = build_dataset_for_grpo(args.data)
    
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward_func, environment_reward_func],
        args=training_args,
        train_dataset=dataset,
    )

    print("\nStarting GRPO Training...")
    print("VRAM usage should peak at ~4.5GB. Generating rollout batches...")
    trainer.train()

    # 5. Save Finished Model
    print(f"\nTraining Complete. Saving to {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

if __name__ == "__main__":
    main()
