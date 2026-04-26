"""
launch_hf_job.py
─────────────────────────────────────────────────────────────
Spawns a Hugging Face Job that runs the full SFT + GRPO pipeline
for BlastRadius end-to-end.

Hardware / image strategy (HF Jobs; verified Apr 2026):
- **H200** + host driver **580 / CUDA 13.0**: ONLY use
  `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`. The NGC image
  `nvcr.io/nvidia/pytorch:24.10-py3` (CUDA 12.6) returns
  `torch.cuda.is_available() == False` with Error 802 even when
  nvidia-smi shows GPUs — the 12.4 forward-compat layer is required.
- **a100-large** is still the most battle-tested profile; default image
  `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel` (override with `HF_JOB_IMAGE`).
- NEVER use NGC/nvcr.io images on H200 HF Job nodes.

vLLM is installed *after* SFT (see `pyproject.toml` `train_sft` / `train_grpo`)
so pip cannot replace torch + bitsandbytes before the cold-start SFT run.

Run locally:
    python scripts/launch_hf_job.py
    $env:HF_JOB_FLAVOR='a100-large'; python scripts/launch_hf_job.py
    $env:HF_JOB_IMAGE='pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel'; python scripts/launch_hf_job.py
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

env_path = REPO_ROOT / ".env"
if not env_path.exists():
    env_path = REPO_ROOT.parent / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

required = ["HF_TOKEN", "WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT", "HUB_MODEL_ID"]
missing = [k for k in required if not os.environ.get(k)]
if missing:
    print(f"FAIL missing env vars in .env: {missing}")
    sys.exit(1)

HF_TOKEN = os.environ["HF_TOKEN"]
WANDB_API_KEY = os.environ["WANDB_API_KEY"]
WANDB_ENTITY = os.environ["WANDB_ENTITY"]
WANDB_PROJECT = os.environ["WANDB_PROJECT"]
HUB_MODEL_ID = os.environ["HUB_MODEL_ID"]

FLAVOR = os.environ.get("HF_JOB_FLAVOR", "h200")
TIMEOUT = os.environ.get("HF_JOB_TIMEOUT", "5h")

def _default_image(flavor: str) -> str:
    if flavor.startswith("h200"):
        return "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
    return "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel"


DOCKER_IMAGE = os.environ.get("HF_JOB_IMAGE", _default_image(FLAVOR))

# BUG #7 FIX: Ensure Hub model repo exists BEFORE the job tries to push.
# hub_strategy="checkpoint" silently fails with 404 if repo doesn't exist yet.
try:
    from huggingface_hub import HfApi  # type: ignore
    _api = HfApi(token=HF_TOKEN)
    _api.create_repo(repo_id=HUB_MODEL_ID, exist_ok=True, private=False, repo_type="model")
    print(f"Hub model repo ready: https://huggingface.co/{HUB_MODEL_ID}")
except Exception as _e:
    print(f"WARNING: Could not pre-create Hub repo ({_e}) — Hub pushes may fail during training.")

JOB_SCRIPT = f"""
set -euo pipefail
export PYTHONUNBUFFERED=1
export CUDA_MODULE_LOADING=EAGER
export PIP_BREAK_SYSTEM_PACKAGES=1
export PIP_ROOT_USER_ACTION=ignore

echo "========================================================"
echo "  BLASTRADIUS H200 TRAINING — v8 (stage-safe)"
echo "========================================================"

echo "==> nvidia-smi"
nvidia-smi

echo "==> CUDA warmup (Error 802 race fix — up to 8 retries)"
ldconfig 2>/dev/null || true
sleep 3
_ok=0
for _attempt in $(seq 1 8); do
  if python3 -c "
import os, sys
os.environ['CUDA_MODULE_LOADING'] = 'EAGER'
import torch
if torch.cuda.is_available():
    print('CUDA ready:', torch.cuda.get_device_name(0))
    sys.exit(0)
sys.exit(1)
"; then
    _ok=1
    break
  fi
  echo "  [warmup] CUDA not ready (attempt $_attempt/8), sleep 5s..."
  ldconfig 2>/dev/null || true
  sleep 5
done
if [ "$_ok" -ne 1 ]; then
  echo "FATAL: CUDA not available after 8 attempts"
  exit 1
fi

echo "==> Installing git + build-essential"
apt-get update -qq && apt-get install -y -qq git build-essential

echo "==> Cloning BlastRadius repo"
[ -d /workspace/.git ] && rm -rf /workspace
git clone --depth 1 --branch main https://github.com/Divyansh-9/BlastRadius.git /workspace
cd /workspace

echo "==> Installing deps (keeping docker torch 2.6.0)"
python3 -m pip install --quiet --upgrade pip

# Pin torch so NOTHING replaces it
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" | tr -d "[:space:]")
echo "torch==${{TORCH_VER}}" > /tmp/pin.txt
export PIP_CONSTRAINT=/tmp/pin.txt

# Exact version trio from v7 that passed Stage 1 on H200
pip install --quiet "transformers==4.51.3"
pip install --quiet "trl==0.13.0"
pip install --quiet "peft==0.13.2"
pip install --quiet "bitsandbytes>=0.43.0"
pip install --quiet "datasets>=2.18.0"
pip install --quiet "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --quiet wandb huggingface_hub python-dotenv plotly networkx

# torchao conflicts with torch 2.6 (unsloth-zoo requires it, but bnb handles 4-bit)
pip uninstall -y torchao 2>/dev/null || true

echo "==> CUDA re-warmup after pip"
ldconfig 2>/dev/null || true
sleep 3
for _attempt in $(seq 1 8); do
  if python3 -c "import torch; assert torch.cuda.is_available(); print('CUDA OK')"; then break; fi
  echo "  [post-pip warmup] attempt $_attempt/8..."
  ldconfig 2>/dev/null || true
  sleep 5
done

echo "==> Verifying imports"
python3 << 'VERIFY'
import torch
print(f"torch: {{torch.__version__}} | CUDA: {{torch.cuda.is_available()}}")
assert torch.cuda.is_available()
print(f"GPU: {{torch.cuda.get_device_name(0)}}")
from unsloth import FastLanguageModel, is_bfloat16_supported
print("unsloth: OK")
from trl import SFTTrainer, SFTConfig
print("trl: OK")
import wandb
print("wandb: OK")
print("=== ALL IMPORTS OK ===")
VERIFY

echo "==> Stage 1: SFT Training"
python3 -u -m agent.train_sft \\
    --model unsloth/Qwen2.5-14B-Instruct-bnb-4bit \\
    --data sft_data/expert_trajectories.jsonl \\
    --output models/sft_checkpoint

echo "==> Validate SFT checkpoint"
python3 -m agent.validate_save --model models/sft_checkpoint

echo "==> *** Pushing SFT checkpoint to Hub (safety save before GRPO) ***"
python3 << 'PUSH_SFT'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_TOKEN"))
hub_id = "{HUB_MODEL_ID}"
# Create repo if missing
api.create_repo(repo_id=hub_id, exist_ok=True, repo_type="model", private=False)
# Upload SFT checkpoint folder
api.upload_folder(
    folder_path="models/sft_checkpoint",
    repo_id=hub_id,
    repo_type="model",
    commit_message="Stage 1 SFT checkpoint — auto-saved before GRPO",
    path_in_repo="sft_checkpoint",
    ignore_patterns=["*.tmp"],
)
print(f"SFT checkpoint pushed to https://huggingface.co/{{hub_id}}/tree/main/sft_checkpoint")
PUSH_SFT

echo "==> Installing vLLM (after SFT, torch still pinned)"
pip install --quiet "vllm>=0.5.0"
pip uninstall -y torchao 2>/dev/null || true
ldconfig 2>/dev/null || true
sleep 3

echo "==> Stage 2: GRPO Training"
python3 -u -m agent.train_grpo \\
    --model models/sft_checkpoint \\
    --data sft_data/expert_trajectories.jsonl \\
    --output models/grpo_checkpoint \\
    --hardware-profile h200 \\
    --wandb-project {WANDB_PROJECT} \\
    --hub-model-id {HUB_MODEL_ID} \\
    --max-runtime-hours 4.0

echo "==> Validate GRPO checkpoint"
python3 -m agent.validate_save --model models/grpo_checkpoint \\
    || python3 -m agent.validate_save --model models/sft_checkpoint

echo "==> ALL DONE — model at https://huggingface.co/{HUB_MODEL_ID}"
""".strip()

cmd = [
    "hf",
    "jobs",
    "run",
    "--flavor",
    FLAVOR,
    "--timeout",
    TIMEOUT,
    "--detach",
    "--secrets",
    f"HF_TOKEN={HF_TOKEN}",
    "--secrets",
    f"WANDB_API_KEY={WANDB_API_KEY}",
    "-e",
    "HF_DEBUG=1",
    "-e",
    "PYTHONUNBUFFERED=1",
    # NOTE: WANDB_ENTITY intentionally NOT passed — we let W&B auto-detect from
    # WANDB_API_KEY. Passing HF account ID as entity causes "entity not found" CommError.
    "-e",
    f"WANDB_PROJECT={WANDB_PROJECT}",
    "-e",
    f"HUB_MODEL_ID={HUB_MODEL_ID}",
    "-e",
    f"HF_JOB_IMAGE={DOCKER_IMAGE}",
    DOCKER_IMAGE,
    "bash",
    "-c",
    JOB_SCRIPT,
]

print("=" * 60)
print(f"Launching HF Job: {FLAVOR}, {TIMEOUT} timeout")
print(f"  Image:  {DOCKER_IMAGE}")
print(f"  WANDB:  https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}")
print(f"  MODEL:  https://huggingface.co/{HUB_MODEL_ID}")
print("=" * 60)

result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:")
    print(result.stderr)
    sys.exit(result.returncode)
