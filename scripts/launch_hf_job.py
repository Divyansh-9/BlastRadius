"""
launch_hf_job.py
─────────────────────────────────────────────────────────────
Spawns a Hugging Face Job that runs the full SFT + GRPO pipeline
for BlastRadius end-to-end.

Hardware / image strategy (read this if H200 + PyTorch was failing):
- HF "h200" nodes expose the GPU to `nvidia-smi`, but the stock
  `pytorch/pytorch` wheels can still return `torch.cuda.is_available() == False`
  with "Error 802: system not yet initialized" because the *user-mode*
  CUDA stack in the container does not line up with the host driver the
  way Jobs wires devices. The **NVIDIA NGC** `nvcr.io/nvidia/pytorch` images
  ship a tested driver/runtime pairing and are the supported fix on H200.
- `a100-large` is more predictable with the official `pytorch/pytorch` CUDA
  12.1 devel image, but is often queue-blocked during hackathon crunch.

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

# NGC: full CUDA user-mode stack (fixes torch init on many HF h200 workers).
# Hopper/H200 on HF Jobs sometimes hits CUDA 802 ("system not yet initialized") if
# PyTorch loads before the device stack is ready — 25.x NGC images track newer
# host drivers; see JOB_SCRIPT for retry/warmup as well.
# Official pytorch images work better on a100 where the driver/runtime gap is smaller.
def _default_image(flavor: str) -> str:
    if flavor.startswith("h200"):
        # 25.01 verified on NGC; pairs better with current H200 host drivers than 24.10 when 802 appears.
        # Fallback: HF_JOB_IMAGE=nvcr.io/nvidia/pytorch:24.10-py3
        return "nvcr.io/nvidia/pytorch:25.01-py3"
    return "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel"


DOCKER_IMAGE = os.environ.get("HF_JOB_IMAGE", _default_image(FLAVOR))

JOB_SCRIPT = f"""
set -euo pipefail

# ── Container GPU runtime (K8s / HF Jobs) ─────────────────
export NVIDIA_VISIBLE_DEVICES="${{NVIDIA_VISIBLE_DEVICES:-all}}"
export NVIDIA_DRIVER_CAPABILITIES="${{NVIDIA_DRIVER_CAPABILITIES:-compute,utility}}"
# LAZY has been observed to worsen CUDA 802 init ordering on HF h200; EAGER is safer here.
export CUDA_MODULE_LOADING="${{CUDA_MODULE_LOADING:-EAGER}}"

# Prefer bundled CUDA libs from common locations (pytorch + NGC layouts).
for _lib in /usr/local/cuda/lib64 /usr/local/cuda-12.4/lib64 /usr/local/cuda-12.6/lib64 \\
            /usr/local/cuda-13.0/lib64 /usr/lib/x86_64-linux-gnu; do
  if [ -d "$_lib" ]; then
    export LD_LIBRARY_PATH="$_lib:${{LD_LIBRARY_PATH:-}}"
  fi
done

echo "==> System + GPU probe"
uname -a || true
ldconfig -p 2>/dev/null | head -n 5 || true
ls -la /dev/nvidia* 2>/dev/null || true

echo "==> nvidia-smi (host-visible GPUs)"
nvidia-smi
# Driver stack can lag nvidia-smi on fresh HF pods; brief wait + ldconfig reduces CUDA 802 races.
ldconfig 2>/dev/null || true
sleep 3

echo "==> Pre-pip: baseline torch (must be True on a healthy image)"
export PIP_BREAK_SYSTEM_PACKAGES=1
export PIP_ROOT_USER_ACTION=ignore

_ok=0
for _attempt in $(seq 1 45); do
  if python3 -c "
import os, sys
print('LD_LIBRARY_PATH=', (os.environ.get('LD_LIBRARY_PATH') or '')[:500])
import torch
print('torch', torch.__version__, 'built_cuda', torch.version.cuda, 'is_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device0', torch.cuda.get_device_name(0))
    sys.exit(0)
sys.exit(1)
"; then
    _ok=1
    break
  fi
  echo "  [warmup] torch CUDA not ready (attempt $_attempt/45), sleeping 2s — HF H200 802 race?"
  sleep 2
done
if [ "$_ok" -ne 1 ]; then
  echo "FATAL: torch sees no CUDA before any project pip install after warm-up retries. Try HF_JOB_IMAGE=nvcr.io/nvidia/pytorch:25.04-py3 or flavor a100-large."
  exit 1
fi

echo "==> Apt: git + build essentials (only if missing)"
if ! command -v git &>/dev/null; then
  apt-get update -qq
  apt-get install -y -qq git
fi
apt-get update -qq
apt-get install -y -qq build-essential || true

echo "==> Cloning BlastRadius main"
if [ -d /workspace/.git ]; then rm -rf /workspace; fi
git clone --depth 1 --branch main https://github.com/Divyansh-9/BlastRadius.git /workspace
cd /workspace

echo "==> pip: Stage 1 only (train_sft — no vLLM yet)"
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet -e ".[train_sft]" --no-build-isolation

echo "==> post-pip: torch check"
python3 <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "is_available", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA broke after train_sft pip"
print("ok", torch.cuda.get_device_name(0))
PY

echo "==> Stage 1: SFT training"
python3 -m agent.train_sft \\
    --model unsloth/Qwen2.5-14B-Instruct-bnb-4bit \\
    --data sft_data/expert_trajectories.jsonl \\
    --output models/sft_checkpoint

echo "==> Validate SFT checkpoint"
python3 -m agent.validate_save --model models/sft_checkpoint

echo "==> pip: vLLM for GRPO (after SFT; pin torch in constraint file)"
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" | tr -d '[:space:]')
echo "torch==$TORCH_VER" > /tmp/torch_pinned
export PIP_CONSTRAINT=/tmp/torch_pinned
python3 -m pip install --quiet "vllm>=0.5.0" --no-build-isolation

echo "==> post-vLLM: torch check"
python3 -c "import torch; assert torch.cuda.is_available()"

echo "==> Stage 2: GRPO training (auto-pushes to {HUB_MODEL_ID})"
python3 -m agent.train_grpo \\
    --model models/sft_checkpoint \\
    --data sft_data/expert_trajectories.jsonl \\
    --output models/grpo_checkpoint \\
    --hardware-profile a100 \\
    --wandb-project {WANDB_PROJECT} \\
    --wandb-entity {WANDB_ENTITY} \\
    --hub-model-id {HUB_MODEL_ID} \\
    --max-runtime-hours 4.0

echo "==> Validate final checkpoint (GRPO -> SFT fallback)"
python3 -m agent.validate_save --model models/grpo_checkpoint \\
    || python3 -m agent.validate_save --model models/sft_checkpoint

echo "==> Done. Model on https://huggingface.co/{HUB_MODEL_ID}"
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
    f"WANDB_ENTITY={WANDB_ENTITY}",
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
