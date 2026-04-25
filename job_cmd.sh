set -euo pipefail
export PYTHONUNBUFFERED=1
export CUDA_MODULE_LOADING=EAGER
export PIP_BREAK_SYSTEM_PACKAGES=1
export PIP_ROOT_USER_ACTION=ignore

echo "========================================================"
echo "  BLASTRADIUS H200 TRAINING — v7"
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

echo "==> Installing git"
apt-get update -qq && apt-get install -y -qq git build-essential

echo "==> Cloning repo"
git clone --depth 1 --branch main https://github.com/Divyansh-9/BlastRadius.git /workspace
cd /workspace

echo "==> Patching indentation bugs"
python3 << 'PATCH'
with open('agent/train_sft.py','r') as f: c=f.read()
c=c.replace('\noutput_dir=args.output,', '\n        output_dir=args.output,')
with open('agent/train_sft.py','w') as f: f.write(c)
print('Patched train_sft.py')

with open('agent/train_grpo.py','r') as f: lines=f.readlines()
fixed=[]
for i,line in enumerate(lines):
    if line.strip().startswith('print') and 'Failed to upload emergency' in line and not line.startswith(' '):
        fixed.append('                '+line)
    elif line.startswith('if os.path.exists(args.output)') and i > 400:
        fixed.append('    '+line)
    else:
        fixed.append(line)
with open('agent/train_grpo.py','w') as f: f.writelines(fixed)
print('Patched train_grpo.py')
PATCH

echo "==> Installing deps (keeping docker torch 2.6.0)"
python3 -m pip install --quiet --upgrade pip

# Pin torch so nothing replaces it
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" | tr -d "[:space:]")
echo "torch==${TORCH_VER}" > /tmp/pin.txt
export PIP_CONSTRAINT=/tmp/pin.txt

pip install --quiet "transformers==4.51.3"
pip install --quiet "trl==0.13.0"
pip install --quiet "peft==0.13.2"
pip install --quiet "bitsandbytes>=0.43.0"
pip install --quiet "datasets>=2.18.0"
pip install --quiet "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --quiet wandb huggingface_hub python-dotenv plotly networkx

echo "==> Removing torchao (incompatible with torch 2.6, bnb handles 4-bit)"
pip uninstall -y torchao 2>/dev/null || true

echo "==> Patching transformers for peft tensor_parallel import"
python3 << 'TP_FIX'
# peft tries to import EmbeddingParallel etc. from transformers
# but they don't exist in transformers 4.51.3 (added in 4.52+).
# Add dummy classes — never used on single GPU (no tensor parallelism).
import transformers.integrations.tensor_parallel as tp
import inspect
src = inspect.getfile(tp)
with open(src, 'r') as f:
    code = f.read()
if 'EmbeddingParallel' not in code:
    with open(src, 'a') as f:
        f.write("""
# BlastRadius fix: dummy TP classes for peft compatibility
class EmbeddingParallel:
    pass
class ColumnLinearParallel:
    pass
class RowLinearParallel:
    pass
""")
    print('Patched: added EmbeddingParallel to transformers')
else:
    print('transformers already has EmbeddingParallel')
TP_FIX

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
print(f"torch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
assert torch.cuda.is_available()
print(f"GPU: {torch.cuda.get_device_name(0)}")

from unsloth import FastLanguageModel, is_bfloat16_supported
print("unsloth: OK")

from trl import SFTTrainer, SFTConfig
print("trl: OK")

print("=== ALL IMPORTS OK ===")
VERIFY

echo "==> Stage 1: SFT Training"
python3 -u -m agent.train_sft \
    --model unsloth/Qwen2.5-14B-Instruct-bnb-4bit \
    --data sft_data/expert_trajectories.jsonl \
    --output models/sft_checkpoint

echo "==> Validate SFT"
python3 -m agent.validate_save --model models/sft_checkpoint

echo "==> Installing vLLM"
pip install --quiet "vllm>=0.5.0"
pip uninstall -y torchao 2>/dev/null || true
ldconfig 2>/dev/null || true
sleep 3

echo "==> Stage 2: GRPO Training"
python3 -u -m agent.train_grpo \
    --model models/sft_checkpoint \
    --data sft_data/expert_trajectories.jsonl \
    --output models/grpo_checkpoint \
    --hardware-profile a100 \
    --max-runtime-hours 4.0

echo "==> Validate GRPO"
python3 -m agent.validate_save --model models/grpo_checkpoint \
    || python3 -m agent.validate_save --model models/sft_checkpoint

echo "==> ALL DONE"
