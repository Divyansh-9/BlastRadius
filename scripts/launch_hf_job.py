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
# huggingface_hub reads HF_DEBUG at *first import* — set before any Python runs.
export HF_DEBUG=1
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

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

echo "===> Running image: ${{HF_JOB_IMAGE:-unknown}}"
echo "===> Checking CUDA device files:"
ls /dev/nvidia* 2>/dev/null || echo "  (no /dev/nvidia* found)"

_ok=0
for _attempt in $(seq 1 8); do
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
  echo "  [warmup] torch CUDA not ready (attempt $_attempt/8), sleep 5s..."
  ldconfig 2>/dev/null || true
  sleep 5
done
if [ "$_ok" -ne 1 ]; then
  echo "FATAL: torch sees no CUDA. Image was: ${{HF_JOB_IMAGE:-unknown}}"
  echo "  Fix: set HF_JOB_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
  echo "  NGC/nvcr.io images (e.g. nvcr.io/nvidia/pytorch:24.10-py3) are INCOMPATIBLE with H200 HF nodes."
  echo "  Alternatively use: HF_JOB_FLAVOR=a100-large"
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
python3 -m pip install --quiet --upgrade pip wheel

# ── NUCLEAR FIX for unsloth PEP-621 license error ─────────────────────────────
# ROOT CAUSE: unsloth/pyproject.toml has `license = "Apache-2.0"` (bare string).
# setuptools >= 69 (bundled in conda inside this PyTorch image) rejects this.
# Downgrading setuptools via pip does NOT work — conda's setuptools takes precedence.
# SOLUTION: Clone unsloth separately, surgically patch its pyproject.toml, then
# install from the local patched source. No conda conflict possible.
echo "  ==> Cloning unsloth to patch license field"
git clone --depth 1 https://github.com/unslothai/unsloth.git /tmp/unsloth_src
# Replace the bare string license with PEP-621-compliant table form
sed -i 's/^license = "Apache-2.0"/license = {{text = "Apache-2.0"}}/' /tmp/unsloth_src/pyproject.toml
# Verify the patch was applied
echo "  ==> Patched unsloth/pyproject.toml license line:"
grep "^license" /tmp/unsloth_src/pyproject.toml
# Install the patched unsloth with --no-deps to prevent pip from upgrading torch.
# ROOT CAUSE of previous failure: pip install /tmp/unsloth_src pulled torch 2.11.0,
# which broke torchvision==0.21.0 (still pinned to 2.6.0) → unsloth import crash.
python3 -m pip install --quiet --no-deps /tmp/unsloth_src
# Install unsloth's non-torch runtime deps that --no-deps skipped
python3 -m pip install --quiet \
    "accelerate>=0.26.0" \
    "transformers>=4.38.0" \
    "tokenizers>=0.15.0" \
    "sentencepiece>=0.1.99" \
    "packaging>=23.0" \
    "triton" || true
# xformers: install but don't abort if H200 has no wheel yet (triton fallback is fine)
python3 -m pip install --quiet "xformers" 2>&1 | tail -3 || echo "WARNING: xformers not available — using triton fallback"
# Verify unsloth is importable and torch was NOT upgraded (single-line to avoid f-string conflict)
python3 -c "import torch, torchvision; print('torch='+torch.__version__+' torchvision='+torchvision.__version__); from unsloth import FastLanguageModel; print('unsloth OK')" || {{ echo "FATAL: unsloth import failed — aborting"; exit 1; }}

echo "  ==> Installing BlastRadius [train_sft] deps (excluding unsloth — already installed)"
# Install everything in [train_sft] except the unsloth URL (already installed above)
python3 -m pip install --quiet \
    "trl>=0.12.0" \
    "peft>=0.10.0" \
    "bitsandbytes>=0.43.0" \
    "wandb>=0.16.0" \
    "huggingface_hub>=0.23.0" \
    "datasets>=2.18.0" \
    "plotly>=5.0.0" \
    "networkx>=3.0" \
    "python-dotenv>=1.0.0"
# Install the environment package itself (our pyproject.toml is PEP-621 clean)
python3 -m pip install --quiet -e "." --no-deps

echo "==> post-pip: torch check"
python3 <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "is_available", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA broke after train_sft pip"
print("ok", torch.cuda.get_device_name(0))
PY

echo "==> Stage 1: SFT training"
python3 -u -m agent.train_sft \\
    --model unsloth/Qwen2.5-14B-Instruct-bnb-4bit \\
    --data sft_data/expert_trajectories.jsonl \\
    --output models/sft_checkpoint

echo "==> Validate SFT checkpoint"
python3 -m agent.validate_save --model models/sft_checkpoint

echo "==> pip: vLLM for GRPO (after SFT; pin torch in constraint file)"
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" | tr -d '[:space:]')
echo "torch==$TORCH_VER" > /tmp/torch_pinned
export PIP_CONSTRAINT=/tmp/torch_pinned
python3 -m pip install --quiet "vllm>=0.5.0"

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
    "HF_DEBUG=1",
    "-e",
    "PYTHONUNBUFFERED=1",
    "-e",
    f"WANDB_ENTITY={WANDB_ENTITY}",
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
