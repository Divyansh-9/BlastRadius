"""
launch_hf_job.py
─────────────────────────────────────────────────────────────
Spawns a Hugging Face Job on H200 (or A100 via env override)
that runs the full SFT + GRPO pipeline for BlastRadius end-to-end.

Why H200 default:
    During the OpenEnv hackathon (~800 teams), A100-large is
    queue-blocked while H200 has open capacity AND is half the
    cost ($5/hr vs $10/hr) AND has 1.7x the VRAM (141GB vs 80GB).
    The only catch: H200 nodes ship CUDA 13.0 driver, so we use
    a pytorch:cuda12.8 base image (forward-compat works).

Pipeline inside the job (~4-5 hours):
    git clone main -> pip install -> train_sft -> validate
    -> train_grpo (auto-pushes to HUB_MODEL_ID every 200 steps
    via hub_strategy=checkpoint) -> validate

Run locally:
    python scripts/launch_hf_job.py                # default H200
    $env:HF_JOB_FLAVOR='a100-large'; python scripts/launch_hf_job.py
Then monitor via WandB and:
    hf jobs logs   <job_id>
    hf jobs ps
    hf jobs cancel <job_id>
"""

import os
import shlex
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

HF_TOKEN       = os.environ["HF_TOKEN"]
WANDB_API_KEY  = os.environ["WANDB_API_KEY"]
WANDB_ENTITY   = os.environ["WANDB_ENTITY"]
WANDB_PROJECT  = os.environ["WANDB_PROJECT"]
HUB_MODEL_ID   = os.environ["HUB_MODEL_ID"]

JOB_SCRIPT = f"""
set -euo pipefail

echo "==> Apt: git + build essentials"
apt-get update -qq
apt-get install -y -qq git build-essential

echo "==> nvidia-smi (driver + GPU before any pip)"
nvidia-smi

echo "==> Pre-install torch sanity (image baseline)"
python -c "import torch; print('IMAGE torch', torch.__version__, 'cuda', torch.version.cuda, 'is_available', torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA broken in base image'"

echo "==> Cloning BlastRadius main"
git clone --branch main https://github.com/Divyansh-9/BlastRadius.git /workspace
cd /workspace

echo "==> Pip env hardening (PEP 668 + freeze torch to image build)"
export PIP_BREAK_SYSTEM_PACKAGES=1
export PIP_ROOT_USER_ACTION=ignore
TORCH_VER=$(python -c "import torch; print(torch.__version__)")
TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda)")
echo "Locking torch==$TORCH_VER (cuda $TORCH_CUDA) so deps cannot downgrade it"
echo "torch==$TORCH_VER" > /tmp/torch.constraint
export PIP_CONSTRAINT=/tmp/torch.constraint

echo "==> Installing project + train extras (torch is locked)"
pip install --quiet --upgrade pip
pip install --quiet -e '.[train]' --no-build-isolation
pip install --quiet 'trl>=0.12.0' wandb python-dotenv --no-build-isolation

echo "==> Post-install torch sanity (must still see GPU)"
python -c "import torch; print('POST torch', torch.__version__, 'cuda', torch.version.cuda, 'is_available', torch.cuda.is_available(), 'device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); assert torch.cuda.is_available(), 'CUDA broken AFTER pip install — a dependency replaced torch'"

echo "==> Stage 1: SFT training"
python -m agent.train_sft \\
    --model unsloth/Qwen2.5-14B-Instruct-bnb-4bit \\
    --data sft_data/expert_trajectories.jsonl \\
    --output models/sft_checkpoint

echo "==> Validate SFT checkpoint"
python -m agent.validate_save --model models/sft_checkpoint

echo "==> Stage 2: GRPO training (auto-pushes to {HUB_MODEL_ID})"
python -m agent.train_grpo \\
    --model models/sft_checkpoint \\
    --data sft_data/expert_trajectories.jsonl \\
    --output models/grpo_checkpoint \\
    --hardware-profile a100 \\
    --wandb-project {WANDB_PROJECT} \\
    --wandb-entity {WANDB_ENTITY} \\
    --hub-model-id {HUB_MODEL_ID} \\
    --max-runtime-hours 4.0

echo "==> Validate final checkpoint (GRPO -> SFT fallback)"
python -m agent.validate_save --model models/grpo_checkpoint \\
    || python -m agent.validate_save --model models/sft_checkpoint

echo "==> Done. Model on https://huggingface.co/{HUB_MODEL_ID}"
""".strip()

FLAVOR  = os.environ.get("HF_JOB_FLAVOR", "h200")
TIMEOUT = os.environ.get("HF_JOB_TIMEOUT", "5h")

cmd = [
    "hf", "jobs", "run",
    "--flavor", FLAVOR,
    "--timeout", TIMEOUT,
    "--detach",
    "--secrets", f"HF_TOKEN={HF_TOKEN}",
    "--secrets", f"WANDB_API_KEY={WANDB_API_KEY}",
    "-e", f"WANDB_ENTITY={WANDB_ENTITY}",
    "-e", f"WANDB_PROJECT={WANDB_PROJECT}",
    "-e", f"HUB_MODEL_ID={HUB_MODEL_ID}",
    os.environ.get("HF_JOB_IMAGE", "pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel"),
    "bash", "-c", JOB_SCRIPT,
]

print("=" * 60)
print(f"Launching HF Job: {FLAVOR}, {TIMEOUT} timeout")
print(f"  WANDB:  https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}")
print(f"  MODEL:  https://huggingface.co/{HUB_MODEL_ID}")
print("=" * 60)

result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:")
    print(result.stderr)
    sys.exit(result.returncode)
