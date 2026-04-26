"""
MATPO GRPO Training Script
==========================
Phase 3 of the BlastRadius Reinforcement Learning Pipeline.

Rewritten for H200 robustness, explicit hardware profiles, and native TRL/HF components.
vLLM is now fully optional (--use-vllm) and Unsloth is removed to prevent fragile dependency conflicts.

Hardware profiles supported:
  - 6gb  : 4-bit base + G=4 generations + grad-accum=4
  - a10  : 4-bit base + G=8 generations + grad-accum=2
  - a100 : bf16 base  + G=16 generations + grad-accum=2
  - h200 : bf16 base  + G=16 generations + grad-accum=4  (141GB VRAM Aware)
"""

import os
import sys
import argparse
import json
import concurrent.futures
import signal
import time
import threading
from typing import List
from pathlib import Path
import torch

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from trl import GRPOConfig, GRPOTrainer

try:
    import wandb
except ImportError:
    wandb = None

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from incident_env.server.incident_environment import IncidentEnvironment
from incident_env.models import IncidentAction
from agent.prompts import (
    SCOUT_TAGS,
    COMMANDER_TAGS,
    THINK_TAGS,
)

# ─────────────────────────────────────────────────────────────
# Runtime Validations
# ─────────────────────────────────────────────────────────────

def validate_environment(args):
    """Fail early if the environment is broken before loading heavy models."""
    if not torch.cuda.is_available():
        raise RuntimeError("FATAL: CUDA is not available. GPU is required.")
        
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"FATAL: Dataset not found at {args.data}")
        
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"FATAL: Base SFT model not found at {args.model}")
        
    try:
        os.makedirs(args.output, exist_ok=True)
        test_file = os.path.join(args.output, ".write_test")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except Exception as e:
        raise PermissionError(f"FATAL: Output directory {args.output} is not writable. {e}")
        
    if args.hub_model_id:
        if not os.environ.get("HF_TOKEN"):
            raise ValueError("FATAL: --hub-model-id provided but HF_TOKEN environment variable is missing.")

# ─────────────────────────────────────────────────────────────
# Reward Functions
# ─────────────────────────────────────────────────────────────

def format_reward_func(completions: List[str], role: List[str], **kwargs) -> List[float]:
    rewards = []
    for comp, current_role in zip(completions, role):
        reward = 0.0
        if THINK_TAGS[0] in comp and THINK_TAGS[1] in comp:
            reward += 0.25
        if current_role == "scout":
            if SCOUT_TAGS[0] in comp and SCOUT_TAGS[1] in comp:
                reward += 0.75
            else:
                reward -= 0.5 
        else: 
            if COMMANDER_TAGS[0] in comp and COMMANDER_TAGS[1] in comp:
                reward += 0.5
                action_text = ""
                try:
                    action_text = comp.split(COMMANDER_TAGS[0])[1].split(COMMANDER_TAGS[1])[0].strip()
                    json.loads(action_text)
                    reward += 0.25 
                except Exception:
                    reward -= 0.25 
            else:
                reward -= 0.5
        if reward < 0.5 and len(comp) > 100:
            reward -= (len(comp) * 0.0001)
        rewards.append(reward)
    return rewards

def evaluate_single_env(comp: str, current_role: str, tid: str, snapshot: dict) -> float:
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
    snapshots = kwargs.get("env_snapshot", [None] * len(completions))
    global _env_executor
    if _env_executor is None:
        max_workers = os.cpu_count() or 4
        _env_executor = concurrent.futures.ThreadPoolExecutor(max_workers=min(8, max_workers))

    futures = [
        _env_executor.submit(evaluate_single_env, comp, current_role, tid, snapshot)
        for comp, current_role, tid, snapshot in zip(completions, role, task_id, snapshots)
    ]
    return [f.result() for f in futures]

# ─────────────────────────────────────────────────────────────
# Dataset Preprocessing
# ─────────────────────────────────────────────────────────────

_DIFFICULTY_ORDER = {
    "easy": 0, "medium": 1, "hard": 2,
    "easy_dns_propagation": 0, "easy_redis_oom": 0,
    "medium_cert_expiry": 1, "medium_k8s_eviction": 1,
    "hard_regex_catastrophe": 2, "hard_db_failover": 2,
    "hard_s3_keyspace_overflow": 2,
}

def build_dataset_for_grpo(file_path: str):
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
            "_difficulty_tier": _DIFFICULTY_ORDER.get(task_id, 99),
        }
    dataset = dataset.map(process_row).sort("_difficulty_tier").remove_columns(["_difficulty_tier"])
    return dataset

# ─────────────────────────────────────────────────────────────
# Watchdog & Emergency Handlers
# ─────────────────────────────────────────────────────────────

_model_for_emergency_save = None
_trainer_for_emergency_save = None
_args_for_emergency_save = None

def preemption_handler(signum, frame):
    print("\n⚠️ SIGTERM received — emergency checkpoint save to Hub", flush=True)
    step = _trainer_for_emergency_save.state.global_step if _trainer_for_emergency_save else "unknown"
    emergency_dir = "/tmp/emergency-checkpoint"
    if _model_for_emergency_save:
        try:
            _model_for_emergency_save.save_pretrained(emergency_dir)
        except Exception as e:
            print(f"❌ Failed to save model locally: {e}")
            sys.exit(1)
            
    if _args_for_emergency_save and _args_for_emergency_save.hub_model_id:
        try:
            from huggingface_hub import HfApi
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
    sys.exit(0)

# ─────────────────────────────────────────────────────────────
# Training Routine
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MATPO GRPO Training (Native HF)")
    parser.add_argument("--model", default="models/sft_checkpoint", help="Path to SFT model")
    parser.add_argument("--data", default="sft_data/expert_trajectories.jsonl", help="Path to offline rollouts")
    parser.add_argument("--output", default="models/grpo_checkpoint", help="Output directory")
    parser.add_argument("--hardware-profile", choices=["6gb", "a10", "a100", "h200"], default="h200", help="Hardware scaling profile")
    parser.add_argument("--use-vllm", action="store_true", help="Opt-in to use vLLM for faster generation")
    
    # MLOps arguments
    parser.add_argument("--hub-model-id", default=os.environ.get("HUB_MODEL_ID", ""), help="Hugging Face repo ID")
    parser.add_argument("--wandb-project", default="blastradius-grpo", help="WandB project name")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""), help="WandB team entity")
    parser.add_argument("--max-runtime-hours", type=float, default=2.0, help="Wall-clock limit")
    parser.add_argument("--max-steps", type=int, default=-1, help="Hard step cap (-1 = use num_train_epochs)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  STAGE 3: MATPO-GRPO RL TRAINING ({args.hardware_profile.upper()})")
    print(f"{'='*60}\n")
    
    # 1. Validation
    validate_environment(args)

    # 2. Hardware profile configuration
    is_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if is_bf16 else torch.float16

    if args.hardware_profile == "h200":
        load_in_4bit = False
        num_generations = 8          # halved from 16 — cuts per-step time ~50%
        per_device_train_batch_size = 4
        gradient_accumulation_steps = 4
        vllm_gpu_memory_utilization = 0.50 # H200 has 141GB, conservative vllm ratio
        is_bf16 = True
    elif args.hardware_profile == "a100":
        load_in_4bit = False
        num_generations = 16
        per_device_train_batch_size = 4
        gradient_accumulation_steps = 2
        vllm_gpu_memory_utilization = 0.70
        is_bf16 = True
    elif args.hardware_profile == "a10":
        load_in_4bit = True
        num_generations = 8
        per_device_train_batch_size = 2
        gradient_accumulation_steps = 2
        vllm_gpu_memory_utilization = 0.60
        is_bf16 = False
    else: # 6gb
        load_in_4bit = True
        num_generations = 4
        per_device_train_batch_size = 1
        gradient_accumulation_steps = 4
        vllm_gpu_memory_utilization = 0.50
        is_bf16 = False

    # 3. Model Loading
    max_seq_length = 2048
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        bnb_config = None

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=compute_dtype,
        )
        print(f"Loaded base model via AutoModelForCausalLM: {args.model}")
    except Exception as e:
        raise RuntimeError(f"FATAL: Failed to load model {args.model}. Error: {e}")

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64,
        bias="none",
    )
    model = get_peft_model(model, peft_config)

    global _model_for_emergency_save, _trainer_for_emergency_save, _args_for_emergency_save
    _model_for_emergency_save = model
    _trainer_for_emergency_save = None
    _args_for_emergency_save = args

    signal.signal(signal.SIGTERM, preemption_handler)
    signal.signal(signal.SIGINT, preemption_handler)

    max_seconds = int(args.max_runtime_hours * 3600)
    def _wall_clock_watchdog():
        time.sleep(max_seconds)
        print(f"\nWall-clock limit ({args.max_runtime_hours}h) reached — stopping gracefully.")
        if _trainer_for_emergency_save is not None:
            _trainer_for_emergency_save.control.should_training_stop = True
        else:
            preemption_handler(None, None)

    threading.Thread(target=_wall_clock_watchdog, daemon=True, name="WallClockWatchdog").start()

    # Use a local alias so that the fallback assignment (`_wandb = None`) does NOT
    # create an UnboundLocalError — assigning to `wandb` directly inside a function
    # makes Python treat every reference to it as local, crashing the `if` check above.
    import wandb as _wandb_mod
    _wandb = _wandb_mod  # module-level wandb captured safely
    if _wandb and args.wandb_project:
        try:
            _wandb.init(
                project=args.wandb_project,
                # Do NOT pass entity — let W&B auto-detect from the API key.
                name=f"grpo-{args.hardware_profile}-G{num_generations}-{int(time.time())}",
                config={"hardware_profile": args.hardware_profile, "use_vllm": args.use_vllm}
            )
            print(f"W&B run: {_wandb.run.url}")
        except Exception as _wb_err:
            print(f"WARNING: W&B init failed ({_wb_err}) — continuing without tracking.")
            _wandb = None

    # 4. GRPO Configuration
    # trl==0.13.0 GRPOConfig does NOT support vllm_device / vllm_gpu_memory_utilization.
    # Those were added in trl>=0.15. Only pass use_vllm (bool).
    # max_steps=-1 means "use num_train_epochs" (TRL default behaviour).
    _max_steps = args.max_steps if args.max_steps > 0 else -1
    training_args = GRPOConfig(
        use_vllm=args.use_vllm,
        num_generations=num_generations,
        max_prompt_length=1024,
        max_completion_length=768,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=1e-6,
        optim="adamw_torch_fused",
        num_train_epochs=1,       # 1 epoch: halves wall-clock vs 2 epochs
        max_steps=_max_steps,
        logging_steps=5,
        output_dir=args.output,
        beta=0.1,
        save_steps=50,
        save_strategy="steps",
        save_total_limit=2,
        push_to_hub=bool(args.hub_model_id),
        hub_model_id=args.hub_model_id if args.hub_model_id else None,
        hub_strategy="checkpoint",
        report_to="wandb" if _wandb else "none",
        bf16=is_bf16,
        fp16=not is_bf16,
    )

    dataset = build_dataset_for_grpo(args.data)
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward_func, environment_reward_func],
        args=training_args,
        train_dataset=dataset,
    )
    
    _trainer_for_emergency_save = trainer

    # Graceful Hub recovery
    if args.hub_model_id and not os.path.exists(args.output):
        print("Fresh container detected -- pulling checkpoint from Hub...")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=args.hub_model_id, local_dir=args.output)
        except Exception as _hub_err:
            print(f"Hub download failed ({_hub_err}) — starting fresh.")

    # Graceful local resume
    trainer_state_path = Path(args.output) / "trainer_state.json"
    resume = False
    if trainer_state_path.exists():
        try:
            _state = json.load(open(trainer_state_path))
            resume = True
            print(f"Resuming from valid TRL checkpoint at step {_state.get('global_step', '?')}")
        except Exception:
            print("trainer_state.json unreadable — starting fresh.")
    else:
        # Check for checkpoint directories
        if os.path.exists(args.output) and any(d.startswith("checkpoint-") for d in os.listdir(args.output)):
            resume = True
            print("Found checkpoint directories, attempting to resume.")

    try:
        trainer.train(resume_from_checkpoint=resume)
    except torch.cuda.OutOfMemoryError:
        raise RuntimeError("FATAL: Out of Memory during GRPO. Reduce batch size or num_generations.")
    except Exception as e:
        raise RuntimeError(f"FATAL: GRPO training failed: {e}")

    # 5. Save Finished Model
    print(f"\nTraining Complete. Saving to {args.output}")
    try:
        model.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)
    except Exception as e:
        raise RuntimeError(f"FATAL: Failed to save final GRPO artifacts: {e}")

if __name__ == "__main__":
    main()
