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
import concurrent.futures
import signal
import time
from typing import List
from pathlib import Path

from agent.curriculum import CurriculumScheduler

try:
    import wandb # type: ignore
except ImportError:
    wandb = None

from datasets import load_dataset # type: ignore
try:
    from transformers.trainer_callback import TrainerCallback # type: ignore
except ImportError:
    TrainerCallback = object

try:
    from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported # type: ignore
    # Patch TRL for ultra-fast/memory-optimized GRPO
    PatchFastRL("GRPO", FastLanguageModel)
except ImportError:
    print("Please install unsloth GRPO: pip install unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git")
    sys.exit(1)

from trl import GRPOConfig, GRPOTrainer # type: ignore

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
                
        # 4. Brevity / Anti-Rambling Penalty
        # If the model fails to output a valid action, softly penalize it based on length
        # to prevent it from farming the <think> reward indefinitely without concluding.
        if reward < 0.5 and len(comp) > 100:
            reward -= (len(comp) * 0.0001)

        rewards.append(reward)
    return rewards


def evaluate_single_env(comp: str, current_role: str, tid: str, snapshot: dict) -> float:
    """Worker function for parallel environment evaluation."""
    if current_role == "scout":
        return 0.0

    env = IncidentEnvironment()
    try:
        if snapshot:
            env.restore_snapshot(snapshot)
        else:
            env.reset(task_id=tid)
    except Exception as e:
        print(f"- Env restore failed: {e}")
        return 0.0
        
    try:
        action_text = comp.split(COMMANDER_TAGS[0])[1].split(COMMANDER_TAGS[1])[0].strip()
        if "```json" in action_text:
            action_text = action_text.replace("```json", "").replace("```", "").strip()
            
        action_dict = json.loads(action_text)
        action = IncidentAction(
            command=action_dict.get("command", "check_status"),
            target=action_dict.get("target"),
            parameters=action_dict.get("parameters", {})
        )
    except Exception:
        return -1.0

    try:
        result = env.step(action)
        reward_val = result["reward"]
        info = result.get("info", {})
        if info.get("is_resolved", False):
            reward_val += 0.5
        return reward_val
    except Exception:
        return 0.0


_env_executor = None

def environment_reward_func(completions: List[str], role: List[str], task_id: List[str], step: List[int], history_log: List[List[str]], **kwargs) -> List[float]:
    """
    The main RL signal. Uses ProcessPoolExecutor to evaluate all generated
    completions in parallel, preventing the Python environment from bottlenecking
    the GPU.
    """
    snapshots = kwargs.get("env_snapshot", [None] * len(completions))
    
    global _env_executor
    if _env_executor is None:
        max_workers = os.cpu_count() or 4
        # FIX: ProcessPoolExecutor bypasses the GIL for CPU-heavy env steps.
        # ThreadPoolExecutor bottlenecks here because env.step() is CPU-bound.
        _env_executor = concurrent.futures.ProcessPoolExecutor(max_workers=min(8, max_workers))

    futures = [
        _env_executor.submit(evaluate_single_env, comp, current_role, tid, snapshot)
        for comp, current_role, tid, snapshot in zip(completions, role, task_id, snapshots)
    ]
    rewards = [f.result() for f in futures]

    return rewards


# ─────────────────────────────────────────────────────────────
# Preprocessing Dataset
# ─────────────────────────────────────────────────────────────

# Difficulty tier ordering for curriculum sorting
_DIFFICULTY_ORDER = {
    "easy": 0, "medium": 1, "hard": 2,
    "db_failover": 3, "cert_expiry": 3, "redis_memory_leak": 3,
    "k8s_eviction": 4, "dns_propagation": 4, "regex_catastrophe": 4, "s3_keyspace": 4,
}


def build_dataset_for_grpo(file_path: str):
    """
    GRPOTrainer expects a dataset with 'prompt' formatting string.
    We inject the role and task details into the dataset so the reward
    functions can read them.

    Rows are sorted by difficulty tier (CurriculumScheduler order) so GRPO
    sees easy → medium → hard progression rather than random insertion order.
    """
    dataset = load_dataset("json", data_files=file_path, split="train")

    def process_row(example):
        prompt = [
            {"role": "system", "content": example["system_prompt"]},
            {"role": "user", "content": example["user_prompt"]}
        ]

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
            "env_snapshot": example.get("env_snapshot"),
            # Attach sort key so we can order by curriculum tier
            "_difficulty_tier": _DIFFICULTY_ORDER.get(task_id, 99),
        }

    dataset = dataset.map(process_row)

    # FIX: Sort by difficulty tier to activate curriculum learning.
    # CurriculumScheduler defines the canonical easy→hard ordering.
    dataset = dataset.sort("_difficulty_tier")
    dataset = dataset.remove_columns(["_difficulty_tier"])
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
    parser.add_argument("--hardware-profile", choices=["6gb", "a10", "a100"], default="6gb", help="Hardware scaling profile")
    
    # MLOps arguments
    parser.add_argument("--hub-model-id", default=os.environ.get("HUB_MODEL_ID", ""), help="Hugging Face repo ID (e.g. your-org/blastradius-checkpoint)")
    parser.add_argument("--wandb-project", default="blastradius-grpo", help="WandB project name")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""), help="WandB team entity")
    parser.add_argument("--max-runtime-hours", type=float, default=2.0, help="Wall-clock limit to prevent runaway jobs")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  STAGE 3: MATPO-GRPO RL TRAINING ({args.hardware_profile.upper()} BUDGET)")
    print(f"{'='*60}\n")
    
    # Define scaling params based on hardware profile
    if args.hardware_profile == "a100":
        load_in_4bit = False
        num_generations = 16
        per_device_train_batch_size = 4
        gradient_accumulation_steps = 2
        vllm_gpu_memory_utilization = 0.70
    elif args.hardware_profile == "a10":
        load_in_4bit = True
        num_generations = 8
        per_device_train_batch_size = 2
        gradient_accumulation_steps = 2
        vllm_gpu_memory_utilization = 0.60
    else: # 6gb fallback
        load_in_4bit = True
        num_generations = 4
        per_device_train_batch_size = 1
        gradient_accumulation_steps = 4
        vllm_gpu_memory_utilization = 0.50

    # 1. Load Model with Colocated vLLM integration
    # FIX: 2048 is the minimum for incident scenarios — 1024 truncates hard
    # scenario context mid-reasoning before the model sees the full system state.
    max_seq_length = 2048

    # FIX: gpu_memory_utilization in from_pretrained sets the initial vLLM pool.
    # It MUST match the per-profile vllm_gpu_memory_utilization to avoid OOM.
    # A100: 0.70 → vLLM gets 56GB, leaves 24GB for gradients/optimizer/activations.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=True,         # ENABLES VLLM COLOCATION
        max_lora_rank=32,            # Must match PEFT rank below
        gpu_memory_utilization=vllm_gpu_memory_utilization,
    )

    # 2. Attach LoRA for GRPO updates
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Global variables for the signal handler to access
    global _model_for_emergency_save, _trainer_for_emergency_save, _args_for_emergency_save
    _model_for_emergency_save = model
    _trainer_for_emergency_save = None
    _args_for_emergency_save = args

    def preemption_handler(signum, frame):
        """Called 30 seconds before Spot Instance dies — force save NOW"""
        print("\n⚠️  SIGTERM received — emergency checkpoint save to Hub", flush=True)
        step = _trainer_for_emergency_save.state.global_step if _trainer_for_emergency_save else "unknown"
        
        # Save locally
        emergency_dir = "/tmp/emergency-checkpoint"
        _model_for_emergency_save.save_pretrained(emergency_dir)
        
        # Push to hub (blocking, because we are about to die)
        if _args_for_emergency_save.hub_model_id:
            try:
                from huggingface_hub import HfApi # type: ignore
                api = HfApi()
                api.upload_folder(
                    folder_path=emergency_dir,
                    repo_id=_args_for_emergency_save.hub_model_id,
                    commit_message=f"EMERGENCY-step-{step}",
                    blocking=True,
                )
                print(f"✅ Emergency checkpoint saved to Hub at step {step}")
            except Exception as e:
                print(f"❌ Failed to upload emergency checkpoint: {e}")
        else:
            print("⚠️ No --hub-model-id provided, emergency save only exists in /tmp")
            
        sys.exit(0)

    signal.signal(signal.SIGTERM, preemption_handler)
    signal.signal(signal.SIGINT, preemption_handler)
    signal.signal(signal.SIGALRM, preemption_handler)
    
    # Set wall-clock limit to prevent runaway jobs draining HF credits
    max_seconds = int(args.max_runtime_hours * 3600)
    print(f"⏰ Setting wall-clock alarm for {max_seconds} seconds ({args.max_runtime_hours} hours)")
    signal.alarm(max_seconds)

    # Initialize WandB
    if wandb and args.wandb_entity:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"grpo-{Path(args.model).name}-G{num_generations}-{int(time.time())}",
            config={
                "model": args.model,
                "hardware_profile": args.hardware_profile,
                "num_generations": num_generations,
                "batch_size": per_device_train_batch_size,
                "kl_coeff": 0.04,
            }
        )

    # 3. Configure GRPOTrainer (Strict memory constraints)
    training_args = GRPOConfig(
        use_vllm=True,                          # Leverage integrated vLLM
        vllm_device="cuda:0",
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,       
        
        # Generation limits
        num_generations=num_generations,
        # FIX: 1024 prompt + 512 completion. Hard scenarios hit 800-900 tokens
        # on prompt alone at 512 limit — model was seeing truncated state.
        max_prompt_length=1024,
        max_completion_length=512,

        # Optimizer limits
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # FIX: 5e-6 risks policy divergence before KL kicks in on short runs.
        # 1e-6 is the stable standard for Qwen2.5 GRPO across published work.
        learning_rate=1e-6,
        optim="adamw_8bit",                     # Saves ~0.3GB VRAM

        # Training length
        num_train_epochs=2,
        logging_steps=5,
        output_dir=args.output,

        # FIX: beta=0.1 provides stronger KL constraint to prevent early divergence.
        # At 1e-6 LR + 2 epochs + limited data, policy stability > exploration speed.
        beta=0.1,
        
        # Checkpointing & Hub (Async uploads to prevent dead GPU time)
        save_steps=200,
        save_strategy="steps",
        save_total_limit=2,
        push_to_hub=bool(args.hub_model_id),
        hub_model_id=args.hub_model_id if args.hub_model_id else None,
        hub_strategy="checkpoint",  # Pushes asynchronously automatically!
        report_to="wandb" if wandb and args.wandb_entity else "none",
        
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
    
    _trainer_for_emergency_save = trainer

    print("\nStarting GRPO Training...")
    print("VRAM usage should peak at ~4.5GB. Generating rollout batches...")
    
    # Auto-recover from Hub if fresh container (no local checkpoint)
    if args.hub_model_id and not os.path.exists(args.output):
        print("Fresh container detected -- pulling checkpoint from Hub...")
        from huggingface_hub import snapshot_download  # type: ignore
        snapshot_download(repo_id=args.hub_model_id, local_dir=args.output)

    # FIX: Check for trainer_state.json, not just dir existence.
    # A raw emergency save (weights only, no TRL metadata) will crash
    # resume_from_checkpoint=True with a checkpoint metadata error.
    trainer_state_path = Path(args.output) / "trainer_state.json"
    resume = trainer_state_path.exists()
    if os.path.exists(args.output) and not resume:
        print("⚠️  Output dir exists but no trainer_state.json found — starting fresh (not resuming).")
    trainer.train(resume_from_checkpoint=resume)

    # 5. Save Finished Model
    print(f"\nTraining Complete. Saving to {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

if __name__ == "__main__":
    main()
