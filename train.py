"""
BlastRadius — Full RL Training Pipeline
H200 GPU Optimized | Fault-Tolerant | WandB Tracked
Run: python train.py
"""

import os, sys, json, re, time, shutil, signal, argparse, logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("BlastRadius")

# ── Credentials (from environment — never hardcoded) ──────────────────────────
HF_TOKEN        = os.environ.get("HF_TOKEN", "")
WANDB_API_KEY   = os.environ.get("WANDB_API_KEY", "")
WANDB_ENTITY    = os.environ.get("WANDB_ENTITY", "")
HUB_MODEL_ID    = os.environ.get("HUB_MODEL_ID", "blastradius-grpo")
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", os.environ.get("TEACHER_API_KEY", ""))

BASE_MODEL      = "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"
SFT_DATA_PATH   = "sft_data/expert_trajectories.jsonl"
SFT_OUTPUT      = "models/sft_checkpoint"
GRPO_OUTPUT     = "models/grpo_checkpoint"
BENCHMARK_DIR   = "docs/runs"

# H200 has ~140GB VRAM — scale up from A100 defaults
H200_NUM_GENERATIONS   = 16   # was 4 on A100
H200_BATCH_SIZE        = 4    # was 1 on A100
H200_GRAD_ACCUM        = 2    # was 4
H200_MAX_SEQ_LEN_SFT   = 4096 # was 2048
H200_MAX_SEQ_LEN_GRPO  = 2048 # was 1024
H200_VLLM_MEM_UTIL     = 0.45 # safe split with trainer on 140GB
H200_GPU_MEM_UTIL      = 0.90
H200_SFT_STEPS         = 300
H200_LORA_RANK_SFT     = 32   # was 16
H200_LORA_RANK_GRPO    = 64   # was 32


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 0 — Safety Checks
# ─────────────────────────────────────────────────────────────────────────────

def stage0_safety_checks():
    log.info("=" * 60)
    log.info("  STAGE 0: SAFETY CHECKS")
    log.info("=" * 60)

    # GPU check
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU found. Aborting.")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info(f"GPU: {gpu_name} | VRAM: {vram_gb:.1f} GB")

    # Disk check (need at least 80GB free)
    disk = shutil.disk_usage(".")
    free_gb = disk.free / 1e9
    log.info(f"Disk free: {free_gb:.1f} GB")
    if free_gb < 80:
        raise RuntimeError(f"Insufficient disk space: {free_gb:.1f}GB free, need 80GB+")

    # WandB
    if WANDB_API_KEY:
        import wandb
        wandb.login(key=WANDB_API_KEY)
        log.info("WandB authenticated.")
    else:
        log.warning("WANDB_API_KEY not set — WandB disabled.")
        os.environ["WANDB_DISABLED"] = "true"

    # HF
    if HF_TOKEN:
        from huggingface_hub import login as hf_login
        hf_login(token=HF_TOKEN)
        log.info("HuggingFace authenticated.")
    else:
        log.warning("HF_TOKEN not set — Hub push will be skipped.")

    log.info("Safety checks passed.\n")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — SFT Data Generation (skip if exists)
# ─────────────────────────────────────────────────────────────────────────────

def stage1_generate_sft_data(episodes: int = 100):
    if Path(SFT_DATA_PATH).exists():
        lines = sum(1 for _ in open(SFT_DATA_PATH))
        log.info(f"SFT data already exists ({lines} examples). Skipping generation.")
        return

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY required for SFT data generation but not set.")

    log.info("=" * 60)
    log.info("  STAGE 1: SFT DATA GENERATION")
    log.info("=" * 60)

    os.makedirs("sft_data", exist_ok=True)
    os.environ.setdefault("TEACHER_API_KEY", OPENAI_API_KEY)

    from agent.generate_sft_data import ExpertEpisodeRunner
    runner = ExpertEpisodeRunner()
    tasks = ["easy", "medium", "hard", "easy_dns_propagation", "easy_redis_oom",
             "medium_cert_expiry", "medium_k8s_eviction", "hard_regex_catastrophe",
             "hard_db_failover", "hard_s3_keyspace_overflow"]
    total = 0

    with open(SFT_DATA_PATH, "w") as f:
        for ep in range(episodes):
            task_id = tasks[ep % len(tasks)]
            log.info(f"Episode {ep+1}/{episodes} [{task_id}]...")
            try:
                examples = runner.run_expert_episode(task_id)
                for ex in examples:
                    f.write(json.dumps(ex) + "\n")
                total += len(examples)
                log.info(f"  → {len(examples)} examples (total: {total})")
            except Exception as e:
                log.warning(f"  Episode failed: {e}")

    log.info(f"Generated {total} training examples → {SFT_DATA_PATH}\n")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — Cold-Start SFT Training
# ─────────────────────────────────────────────────────────────────────────────

def stage2_sft_training(resume: bool = True):
    log.info("=" * 60)
    log.info("  STAGE 2: COLD-START SFT TRAINING (H200)")
    log.info("=" * 60)

    # Check for existing checkpoint
    if resume and Path(SFT_OUTPUT).exists() and any(Path(SFT_OUTPUT).iterdir()):
        log.info(f"SFT checkpoint found at {SFT_OUTPUT}. Skipping SFT training.")
        return

    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    from unsloth import FastLanguageModel, is_bfloat16_supported

    if not Path(SFT_DATA_PATH).exists():
        raise RuntimeError(f"SFT data not found: {SFT_DATA_PATH}")

    data_lines = sum(1 for _ in open(SFT_DATA_PATH))
    log.info(f"SFT data: {data_lines} examples")
    log.info(f"Base model: {BASE_MODEL}")
    log.info(f"H200 config: rank={H200_LORA_RANK_SFT}, steps={H200_SFT_STEPS}, batch={H200_BATCH_SIZE}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=H200_MAX_SEQ_LEN_SFT,
        dtype=None,
        load_in_4bit=True,
        token=HF_TOKEN or None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=H200_LORA_RANK_SFT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=H200_LORA_RANK_SFT,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    dataset = load_dataset("json", data_files=SFT_DATA_PATH, split="train")

    def fmt(example):
        texts = []
        for sys_msg, usr_msg, resp in zip(
            example["system_prompt"], example["user_prompt"], example["response"]
        ):
            msgs = [
                {"role": "system", "content": sys_msg},
                {"role": "user",   "content": usr_msg},
                {"role": "assistant", "content": resp},
            ]
            texts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
        return {"text": texts}

    dataset = dataset.map(fmt, batched=True)

    training_args = SFTConfig(
        per_device_train_batch_size=H200_BATCH_SIZE,
        gradient_accumulation_steps=H200_GRAD_ACCUM,
        warmup_steps=20,
        max_steps=H200_SFT_STEPS,
        learning_rate=2e-5,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        save_steps=50,
        output_dir=SFT_OUTPUT,
        optim="adamw_8bit",
        dataset_text_field="text",
        max_seq_length=H200_MAX_SEQ_LEN_SFT,
        report_to="wandb" if WANDB_API_KEY else "none",
        run_name="blastradius-sft",
        save_total_limit=2,
    )

    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset, args=training_args)
    log.info("Starting SFT training...")
    trainer.train()

    model.save_pretrained(SFT_OUTPUT)
    tokenizer.save_pretrained(SFT_OUTPUT)
    log.info(f"SFT model saved to {SFT_OUTPUT}\n")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — Validate SFT Checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def stage3_validate_checkpoint(model_path: str, label: str = "SFT"):
    log.info("=" * 60)
    log.info(f"  STAGE 3: VALIDATING {label} CHECKPOINT")
    log.info("=" * 60)

    if not Path(model_path).exists():
        raise RuntimeError(f"{label} checkpoint not found at {model_path}")

    files = list(Path(model_path).iterdir())
    required = ["tokenizer_config.json", "special_tokens_map.json"]
    for req in required:
        if not (Path(model_path) / req).exists():
            raise RuntimeError(f"Missing file in checkpoint: {req}")

    has_weights = any(
        f.suffix in [".safetensors", ".bin", ".pt"] or "adapter" in f.name
        for f in files
    )
    if not has_weights:
        raise RuntimeError(f"No model weights found in {model_path}")

    log.info(f"{label} checkpoint valid: {len(files)} files found.")

    # Quick forward-pass test
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=512,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        inputs = tokenizer("Test prompt", return_tensors="pt").to("cuda")
        import torch
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=10)
        log.info(f"Forward pass OK: {tokenizer.decode(out[0], skip_special_tokens=True)[:80]}")
        del model
        import gc; gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        raise RuntimeError(f"Forward pass failed: {e}")

    log.info(f"{label} validation passed.\n")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — GRPO RL Training
# ─────────────────────────────────────────────────────────────────────────────

# ── Reward functions ──────────────────────────────────────────────────────────

THINK_OPEN   = "<think>"
THINK_CLOSE  = "</think>"
SCOUT_OPEN   = "<triage>"
SCOUT_CLOSE  = "</triage>"
CMD_OPEN     = "<action>"
CMD_CLOSE    = "</action>"

def format_reward_func(completions: List[str], role: List[str], **kwargs) -> List[float]:
    rewards = []
    for comp, r in zip(completions, role):
        reward = 0.0
        if THINK_OPEN in comp and THINK_CLOSE in comp:
            reward += 0.25
        if r == "scout":
            if SCOUT_OPEN in comp and SCOUT_CLOSE in comp:
                reward += 0.75
            else:
                reward -= 0.5
        else:
            if CMD_OPEN in comp and CMD_CLOSE in comp:
                reward += 0.5
                try:
                    txt = comp.split(CMD_OPEN)[1].split(CMD_CLOSE)[0].strip()
                    json.loads(txt)
                    reward += 0.25
                except Exception:
                    reward -= 0.25
            else:
                reward -= 0.5
        rewards.append(reward)
    return rewards


def environment_reward_func(completions, role, task_id, step, history_log, **kwargs):
    sys.path.insert(0, str(Path(__file__).parent))
    from incident_env.server.incident_environment import IncidentEnvironment
    from incident_env.models import IncidentAction

    env = IncidentEnvironment()
    rewards = []

    for comp, r, tid, s, hist in zip(completions, role, task_id, step, history_log):
        if r == "scout":
            rewards.append(0.0)
            continue
        try:
            env.reset(task_id=tid)
            for _ in range(max(0, s - 1)):
                env.state.time_elapsed_minutes += 5
                env.graph.tick(5)
        except Exception:
            rewards.append(0.0)
            continue
        try:
            txt = comp.split(CMD_OPEN)[1].split(CMD_CLOSE)[0].strip()
            txt = re.sub(r"```json|```", "", txt).strip()
            d   = json.loads(txt)
            action = IncidentAction(
                command=d.get("command", "check_status"),
                target=d.get("target"),
                parameters=d.get("parameters", {}),
            )
        except Exception:
            rewards.append(-1.0)
            continue
        try:
            result     = env.step(action)
            reward_val = result["reward"] if isinstance(result, dict) else result.reward
            if (result.get("info", {}) if isinstance(result, dict) else {}).get("is_resolved"):
                reward_val += 0.5
            rewards.append(reward_val)
        except Exception:
            rewards.append(0.0)

    return rewards


def build_grpo_dataset(file_path: str):
    from datasets import load_dataset
    ds = load_dataset("json", data_files=file_path, split="train")

    def process(ex):
        prompt = [
            {"role": "system", "content": ex["system_prompt"]},
            {"role": "user",   "content": ex["user_prompt"]},
        ]
        hist = []
        if "[EPISODE HISTORY]" in ex.get("user_prompt", ""):
            block = ex["user_prompt"].split("[EPISODE HISTORY]")[1].split("Based on")[0]
            hist  = [l for l in block.split("\n") if l.strip()]
        return {
            "prompt":      prompt,
            "role":        ex.get("role", "commander"),
            "task_id":     ex.get("task_id", "easy"),
            "step":        ex.get("step", 1),
            "history_log": hist,
        }

    return ds.map(process)


def stage4_grpo_training(resume: bool = True):
    log.info("=" * 60)
    log.info("  STAGE 4: GRPO RL TRAINING (H200 — 140GB)")
    log.info("=" * 60)

    # Graceful SIGTERM handler
    _stop = {"flag": False}
    def _sigterm(sig, frame):
        log.warning("SIGTERM received — will stop after current step.")
        _stop["flag"] = True
    signal.signal(signal.SIGTERM, _sigterm)

    from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
    PatchFastRL("GRPO", FastLanguageModel)
    from trl import GRPOConfig, GRPOTrainer

    # Resume from latest checkpoint
    resume_path = None
    if resume and Path(GRPO_OUTPUT).exists():
        ckpts = sorted(Path(GRPO_OUTPUT).glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[-1]))
        if ckpts:
            resume_path = str(ckpts[-1])
            log.info(f"Resuming GRPO from checkpoint: {resume_path}")

    model_source = resume_path or SFT_OUTPUT
    log.info(f"Loading model from: {model_source}")
    log.info(f"H200 config: G={H200_NUM_GENERATIONS}, batch={H200_BATCH_SIZE}, rank={H200_LORA_RANK_GRPO}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_source,
        max_seq_length=H200_MAX_SEQ_LEN_GRPO,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=H200_LORA_RANK_GRPO,
        gpu_memory_utilization=H200_GPU_MEM_UTIL,
        token=HF_TOKEN or None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=H200_LORA_RANK_GRPO,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=H200_LORA_RANK_GRPO,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    grpo_config = GRPOConfig(
        # vLLM colocation
        use_vllm=True,
        vllm_device="cuda:0",
        vllm_gpu_memory_utilization=H200_VLLM_MEM_UTIL,

        # H200 scaled generation
        num_generations=H200_NUM_GENERATIONS,
        max_prompt_length=1024,
        max_completion_length=1024,

        # Optimizer
        per_device_train_batch_size=H200_BATCH_SIZE,
        gradient_accumulation_steps=H200_GRAD_ACCUM,
        learning_rate=5e-6,
        optim="adamw_8bit",

        # Training length
        num_train_epochs=3,
        logging_steps=5,
        save_steps=25,
        save_total_limit=3,
        output_dir=GRPO_OUTPUT,

        # KL constraint — keep < 0.5
        beta=0.04,

        # Precision
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),

        # WandB
        report_to="wandb" if WANDB_API_KEY else "none",
        run_name="blastradius-grpo",
    )

    dataset = build_grpo_dataset(SFT_DATA_PATH)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward_func, environment_reward_func],
        args=grpo_config,
        train_dataset=dataset,
    )

    log.info("Starting GRPO training...")
    log.info(f"Expected VRAM: ~{H200_NUM_GENERATIONS * 2:.0f}GB for generations + trainer")
    trainer.train(resume_from_checkpoint=resume_path)

    model.save_pretrained(GRPO_OUTPUT)
    tokenizer.save_pretrained(GRPO_OUTPUT)
    log.info(f"GRPO model saved to {GRPO_OUTPUT}\n")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — Push to HuggingFace Hub
# ─────────────────────────────────────────────────────────────────────────────

def stage5_push_to_hub():
    log.info("=" * 60)
    log.info("  STAGE 5: PUSH TO HUGGINGFACE HUB")
    log.info("=" * 60)

    if not HF_TOKEN:
        log.warning("HF_TOKEN not set. Skipping Hub push.")
        return
    if not HUB_MODEL_ID:
        log.warning("HUB_MODEL_ID not set. Skipping Hub push.")
        return

    from huggingface_hub import HfApi
    api = HfApi()

    model_path = GRPO_OUTPUT if Path(GRPO_OUTPUT).exists() else SFT_OUTPUT
    log.info(f"Pushing {model_path} → {HUB_MODEL_ID}")

    api.upload_folder(
        folder_path=model_path,
        repo_id=HUB_MODEL_ID,
        repo_type="model",
        token=HF_TOKEN,
        commit_message=f"BlastRadius GRPO training — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    )

    log.info(f"Model live at: https://huggingface.co/{HUB_MODEL_ID}\n")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6 — Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def stage6_benchmark():
    log.info("=" * 60)
    log.info("  STAGE 6: BENCHMARK")
    log.info("=" * 60)

    import subprocess, sys as _sys
    Path(BENCHMARK_DIR).mkdir(parents=True, exist_ok=True)

    model_path = HUB_MODEL_ID if HF_TOKEN else GRPO_OUTPUT

    cmd = [
        _sys.executable, "-m", "agent.benchmark",
        "--model", model_path,
        "--scenarios", "all",
        "--output-dir", BENCHMARK_DIR,
        "--api-base", os.environ.get("API_BASE_URL", "http://localhost:8000/v1"),
        "--api-key",  os.environ.get("OPENAI_API_KEY", "dummy"),
        "--env-url",  os.environ.get("ENV_BASE_URL",  "http://127.0.0.1:7860"),
    ]

    log.info(f"Running benchmark: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        log.warning(f"Benchmark exited with code {result.returncode}")
    else:
        log.info("Benchmark complete.\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BlastRadius Full Training Pipeline — H200")
    parser.add_argument("--skip-data-gen",    action="store_true", help="Skip SFT data generation")
    parser.add_argument("--skip-sft",         action="store_true", help="Skip SFT training")
    parser.add_argument("--skip-grpo",        action="store_true", help="Skip GRPO training")
    parser.add_argument("--skip-push",        action="store_true", help="Skip HF Hub push")
    parser.add_argument("--skip-benchmark",   action="store_true", help="Skip benchmark")
    parser.add_argument("--sft-episodes",     type=int, default=100, help="Episodes for data gen")
    parser.add_argument("--no-resume",        action="store_true", help="Do not resume from checkpoint")
    args = parser.parse_args()

    resume = not args.no_resume

    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║  💥 BLASTRADIUS — FULL TRAINING PIPELINE (H200)         ║")
    log.info("╚══════════════════════════════════════════════════════════╝")
    log.info(f"Model: {BASE_MODEL}")
    log.info(f"Hub target: {HUB_MODEL_ID}")
    log.info(f"Resume: {resume}")
    log.info(f"Time: {datetime.now().isoformat()}\n")

    try:
        # 0. Safety checks
        stage0_safety_checks()

        # 1. Data generation
        if not args.skip_data_gen:
            stage1_generate_sft_data(episodes=args.sft_episodes)
        else:
            log.info("Skipping Stage 1 (data generation) — --skip-data-gen set.")

        # 2. SFT training
        if not args.skip_sft:
            stage2_sft_training(resume=resume)
        else:
            log.info("Skipping Stage 2 (SFT) — --skip-sft set.")

        # 3. Validate SFT
        stage3_validate_checkpoint(SFT_OUTPUT, label="SFT")

        # 4. GRPO
        if not args.skip_grpo:
            stage4_grpo_training(resume=resume)
        else:
            log.info("Skipping Stage 4 (GRPO) — --skip-grpo set.")

        # 5. Validate GRPO
        grpo_path = GRPO_OUTPUT if Path(GRPO_OUTPUT).exists() else SFT_OUTPUT
        stage3_validate_checkpoint(grpo_path, label="GRPO")

        # 6. Push to Hub
        if not args.skip_push:
            stage5_push_to_hub()
        else:
            log.info("Skipping Stage 5 (Hub push) — --skip-push set.")

        # 7. Benchmark
        if not args.skip_benchmark:
            stage6_benchmark()
        else:
            log.info("Skipping Stage 6 (benchmark) — --skip-benchmark set.")

        log.info("╔══════════════════════════════════════════════════════════╗")
        log.info("║  ✅ PIPELINE COMPLETE                                    ║")
        log.info("╚══════════════════════════════════════════════════════════╝")

    except Exception as e:
        log.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
