"""
launch_hf_job.py
Spawns a Hugging Face Job that runs the full SFT + GRPO pipeline.

FIX (Apr 25 2026):
- Write PIP_CONSTRAINT /tmp/torch_pinned BEFORE any pip install.
  Root cause of exit-code-1: accelerate/transformers upgraded torch
  2.6->2.11, breaking torchvision 0.21+cu124 -> nms does not exist.
- Add nest-asyncio + pydantic to unsloth runtime deps install.
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
    print(f"FAIL missing env vars: {missing}")
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

JOB_SCRIPT = f"""
set -euo pipefail
export HF_DEBUG=1
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export NVIDIA_VISIBLE_DEVICES="${{NVIDIA_VISIBLE_DEVICES:-all}}"
export NVIDIA_DRIVER_CAPABILITIES="${{NVIDIA_DRIVER_CAPABILITIES:-compute,utility}}"
export CUDA_MODULE_LOADING="${{CUDA_MODULE_LOADING:-EAGER}}"
for _lib in /usr/local/cuda/lib64 /usr/local/cuda-12.4/lib64 /usr/local/cuda-12.6/lib64 \\
            /usr/local/cuda-13.0/lib64 /usr/lib/x86_64-linux-gnu; do
  if [ -d "$_lib" ]; then export LD_LIBRARY_PATH="$_lib:${{LD_LIBRARY_PATH:-}}"; fi
done
echo "==> System + GPU probe"
uname -a || true
nvidia-smi
ldconfig 2>/dev/null || true
sleep 3
export PIP_BREAK_SYSTEM_PACKAGES=1
export PIP_ROOT_USER_ACTION=ignore
_ok=0
for _attempt in $(seq 1 8); do
  if python3 -c "
import os,sys,torch
print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available())
if torch.cuda.is_available(): print('device0',torch.cuda.get_device_name(0)); sys.exit(0)
sys.exit(1)"; then
    _ok=1; break
  fi
  echo "  [warmup] CUDA not ready attempt $_attempt/8"
  ldconfig 2>/dev/null || true; sleep 5
done
[ "$_ok" -eq 1 ] || {{ echo "FATAL: torch no CUDA"; exit 1; }}
echo "==> Apt git+build-essential"
command -v git &>/dev/null || (apt-get update -qq && apt-get install -y -qq git)
apt-get update -qq && apt-get install -y -qq build-essential || true
echo "==> Cloning BlastRadius"
[ -d /workspace/.git ] && rm -rf /workspace
git clone --depth 1 --branch main https://github.com/Divyansh-9/BlastRadius.git /workspace
cd /workspace
echo "==> STEP 0: Pin torch NOW before any pip install"
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" | tr -d '[:space:]')
echo "torch==$TORCH_VER" > /tmp/torch_pinned
python3 -c "import torchvision; v=torchvision.__version__; print('torchvision=='+v)" 2>/dev/null | tr -d '[:space:]' >> /tmp/torch_pinned || true
export PIP_CONSTRAINT=/tmp/torch_pinned
echo "  Pinned versions: $(cat /tmp/torch_pinned)"
python3 -m pip install --quiet --upgrade pip wheel
echo "==> STEP 1: Clone+patch unsloth (bypass conda setuptools)"
git clone --depth 1 https://github.com/unslothai/unsloth.git /tmp/unsloth_src
sed -i 's/^license = "Apache-2.0"/license = {{text = "Apache-2.0"}}/' /tmp/unsloth_src/pyproject.toml
grep '^license' /tmp/unsloth_src/pyproject.toml
python3 -m pip install --quiet --no-deps /tmp/unsloth_src
echo "==> STEP 2: Unsloth runtime deps (no torch listed = cannot upgrade)"
python3 -m pip install --quiet \
  "accelerate>=0.26.0" "transformers>=4.38.0" "tokenizers>=0.15.0" \
  "sentencepiece>=0.1.99" "packaging>=23.0" "triton" \
  "nest-asyncio" "pydantic>=2.0.0"
python3 -m pip install --quiet "xformers" 2>&1 | tail -3 || echo "WARNING: xformers not available"
echo "==> STEP 3: Verify torch NOT upgraded + unsloth OK"
python3 << 'VERIFY'
import sys, torch
v = torch.__version__
print(f'torch: {{v}}')
if '2.6' not in v:
    print(f'FATAL: torch was upgraded to {{v}} - pin failed!')
    sys.exit(1)
try:
    import torchvision
    print(f'torchvision: {{torchvision.__version__}}')
except Exception as e:
    print(f'torchvision error: {{e}}')
    sys.exit(1)
from unsloth import FastLanguageModel
print('unsloth OK')
VERIFY
echo "==> STEP 4: BlastRadius train_sft deps"
python3 -m pip install --quiet \
  "trl>=0.12.0" "peft>=0.10.0" "bitsandbytes>=0.43.0" "wandb>=0.16.0" \
  "huggingface_hub>=0.23.0" "datasets>=2.18.0" "plotly>=5.0.0" \
  "networkx>=3.0" "python-dotenv>=1.0.0"
python3 -m pip install --quiet -e "." --no-deps
echo "==> post-pip torch check"
python3 << 'PY'
import torch
print("torch",torch.__version__,"cuda",torch.version.cuda,"avail",torch.cuda.is_available())
assert torch.cuda.is_available()
print("ok",torch.cuda.get_device_name(0))
PY
echo "==> Stage 1: SFT"
python3 -u -m agent.train_sft \\
  --model unsloth/Qwen2.5-14B-Instruct-bnb-4bit \\
  --data sft_data/expert_trajectories.jsonl \\
  --output models/sft_checkpoint
echo "==> Validate SFT"
python3 -m agent.validate_save --model models/sft_checkpoint
echo "==> pip vLLM (PIP_CONSTRAINT active, torch stays pinned)"
python3 -m pip install --quiet "vllm>=0.5.0"
python3 -c "import torch; assert torch.cuda.is_available()"
echo "==> Stage 2: GRPO"
python3 -m agent.train_grpo \\
  --model models/sft_checkpoint \\
  --data sft_data/expert_trajectories.jsonl \\
  --output models/grpo_checkpoint \\
  --hardware-profile a100 \\
  --wandb-project {WANDB_PROJECT} \\
  --wandb-entity {WANDB_ENTITY} \\
  --hub-model-id {HUB_MODEL_ID} \\
  --max-runtime-hours 4.0
echo "==> Validate final"
python3 -m agent.validate_save --model models/grpo_checkpoint \\
  || python3 -m agent.validate_save --model models/sft_checkpoint
echo "==> Done: https://huggingface.co/{HUB_MODEL_ID}"
""".strip()

cmd = [
    "hf", "jobs", "run",
    "--flavor", FLAVOR,
    "--timeout", TIMEOUT,
    "--detach",
    "--secrets", f"HF_TOKEN={HF_TOKEN}",
    "--secrets", f"WANDB_API_KEY={WANDB_API_KEY}",
    "-e", "HF_DEBUG=1",
    "-e", "PYTHONUNBUFFERED=1",
    "-e", f"WANDB_ENTITY={WANDB_ENTITY}",
    "-e", f"WANDB_PROJECT={WANDB_PROJECT}",
    "-e", f"HUB_MODEL_ID={HUB_MODEL_ID}",
    "-e", f"HF_JOB_IMAGE={DOCKER_IMAGE}",
    DOCKER_IMAGE,
    "bash", "-c", JOB_SCRIPT,
]

print("=" * 60)
print(f"Launching HF Job: {FLAVOR}, {TIMEOUT} timeout")
print(f"  Image: {DOCKER_IMAGE}")
print(f"  WANDB: https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}")
print(f"  MODEL: https://huggingface.co/{HUB_MODEL_ID}")
print("=" * 60)

result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr)
    sys.exit(result.returncode)
