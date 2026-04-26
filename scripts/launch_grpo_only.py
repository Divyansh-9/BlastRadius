"""
launch_grpo_only.py
────────────────────────────────────────────────────────
Launches a new HF Job that runs ONLY Stage 3 (GRPO).
Stages 1 (SFT) and 2 (Hub push) are already done.
The SFT checkpoint is pulled from HuggingFace Hub before
GRPO training starts.

Usage:
    python scripts/launch_grpo_only.py
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

required = ["HF_TOKEN", "WANDB_API_KEY", "WANDB_PROJECT", "HUB_MODEL_ID"]
missing = [k for k in required if not os.environ.get(k)]
if missing:
    print(f"FAIL missing env vars in .env: {missing}")
    sys.exit(1)

HF_TOKEN      = os.environ["HF_TOKEN"]
WANDB_API_KEY = os.environ["WANDB_API_KEY"]
WANDB_PROJECT = os.environ["WANDB_PROJECT"]
WANDB_ENTITY  = os.environ.get("WANDB_ENTITY", "")
HUB_MODEL_ID  = os.environ["HUB_MODEL_ID"]

FLAVOR       = os.environ.get("HF_JOB_FLAVOR", "h200")
TIMEOUT      = os.environ.get("HF_JOB_TIMEOUT", "2h")
DOCKER_IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"

# ── Ensure Hub repo exists so GRPO can push ──────────────────────────────────
try:
    from huggingface_hub import HfApi
    _api = HfApi(token=HF_TOKEN)
    _api.create_repo(repo_id=HUB_MODEL_ID, exist_ok=True, private=False, repo_type="model")
    print(f"Hub model repo ready: https://huggingface.co/{HUB_MODEL_ID}")
except Exception as _e:
    print(f"WARNING: Could not pre-create Hub repo ({_e})")

JOB_SCRIPT = f"""
set -euo pipefail
export PYTHONUNBUFFERED=1
export CUDA_MODULE_LOADING=EAGER
export PIP_BREAK_SYSTEM_PACKAGES=1
export PIP_ROOT_USER_ACTION=ignore

echo "========================================================"
echo "  BLASTRADIUS H200 — GRPO ONLY (Stage 3 resume)"
echo "  SFT checkpoint: {HUB_MODEL_ID}/sft_checkpoint"
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

echo "==> Cloning BlastRadius repo (main)"
[ -d /workspace/.git ] && rm -rf /workspace
git clone --depth 1 --branch main https://github.com/Divyansh-9/BlastRadius.git /workspace
cd /workspace

echo "==> Installing deps (keeping docker torch 2.6.0)"
python3 -m pip install --quiet --upgrade pip

TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" | tr -d "[:space:]")
echo "torch==${{TORCH_VER}}" > /tmp/pin.txt
export PIP_CONSTRAINT=/tmp/pin.txt

pip install --quiet "transformers==4.51.3"
pip install --quiet "trl==0.13.0"
pip install --quiet "peft==0.13.2"
pip install --quiet "bitsandbytes>=0.43.0"
pip install --quiet "datasets>=2.18.0"
pip install --quiet "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --quiet wandb huggingface_hub python-dotenv plotly networkx
pip install --quiet "vllm>=0.5.0"
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
from trl import GRPOTrainer, GRPOConfig
print("trl/GRPO: OK")
import wandb
print("wandb: OK")
print("=== ALL IMPORTS OK ===")
VERIFY

echo "==> Downloading SFT checkpoint from Hub"
python3 << 'PULL_SFT'
import os
from huggingface_hub import snapshot_download
hub_id = "{HUB_MODEL_ID}"
local_dir = "models/sft_checkpoint"
print(f"Downloading {{hub_id}}/sft_checkpoint → {{local_dir}} ...")
snapshot_download(
    repo_id=hub_id,
    repo_type="model",
    local_dir=local_dir,
    allow_patterns=["sft_checkpoint/**"],
    token=os.environ.get("HF_TOKEN"),
)
# Flatten: move sft_checkpoint/* one level up if needed
import shutil, pathlib
nested = pathlib.Path(local_dir) / "sft_checkpoint"
if nested.exists():
    for f in nested.iterdir():
        shutil.move(str(f), local_dir)
    nested.rmdir()
print("SFT checkpoint ready at:", local_dir)
import os
for f in os.listdir(local_dir):
    print(" ", f)
PULL_SFT

echo "==> Validating downloaded SFT checkpoint"
python3 -m agent.validate_save --model models/sft_checkpoint

echo "==> Stage 3: GRPO RL Training (hackathon-fast: 300 steps, 8 generations)"
python3 -u -m agent.train_grpo \\
    --model models/sft_checkpoint \\
    --data sft_data/expert_trajectories.jsonl \\
    --output models/grpo_checkpoint \\
    --hardware-profile h200 \\
    --wandb-project {WANDB_PROJECT} \\
    --hub-model-id {HUB_MODEL_ID} \\
    --max-steps 300 \\
    --max-runtime-hours 1.5

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
    "-e",
    f"WANDB_PROJECT={WANDB_PROJECT}",
    "-e",
    f"HUB_MODEL_ID={HUB_MODEL_ID}",
    DOCKER_IMAGE,
    "bash",
    "-c",
    JOB_SCRIPT,
]

print("=" * 60)
print(f"Launching GRPO-ONLY HF Job: {FLAVOR}, {TIMEOUT} timeout")
print(f"  Image:    {DOCKER_IMAGE}")
print(f"  SFT src:  https://huggingface.co/{HUB_MODEL_ID}/tree/main/sft_checkpoint")
print(f"  WANDB:    https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}")
print(f"  Output:   https://huggingface.co/{HUB_MODEL_ID}")
print("=" * 60)

result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:")
    print(result.stderr)
    sys.exit(result.returncode)
