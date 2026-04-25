"""
launch_benchmark.py
────────────────────────────────────────────────────────
Launches an HF Job that:
  1. Downloads GRPO LoRA checkpoint from Hub
  2. Starts a lightweight Unsloth OpenAI-compatible server
  3. Starts the BlastRadius incident env server
  4. Runs the full benchmark (easy / medium / hard)
  5. Uploads the HTML report back to the Hub

NOTE: The GRPO checkpoint is a LoRA adapter — we use Unsloth
      (not vLLM) to load base + LoRA together and expose an
      OpenAI-compatible /v1/chat/completions endpoint.

Usage:
    python scripts/launch_benchmark.py
    python scripts/launch_benchmark.py --flavor h200
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Load .env ───────────────────────────────────────────────────────────────
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

required = ["HF_TOKEN", "HUB_MODEL_ID"]
missing = [k for k in required if not os.environ.get(k)]
if missing:
    print(f"FAIL: missing env vars: {missing}")
    sys.exit(1)

HF_TOKEN      = os.environ["HF_TOKEN"]
HUB_MODEL_ID  = os.environ["HUB_MODEL_ID"]

parser = argparse.ArgumentParser()
parser.add_argument("--flavor", default="h200", help="HF Job GPU flavor (default: h200)")
parser.add_argument("--scenarios", default="easy medium hard", help="Space-separated scenario IDs")
args, _ = parser.parse_known_args()

FLAVOR       = args.flavor
SCENARIOS    = args.scenarios
TIMEOUT      = "1h"
DOCKER_IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"

# ── The inline server script (written to disk inside the job) ────────────────
INFERENCE_SERVER_PY = r'''
"""
Minimal OpenAI-compatible inference server using Unsloth.
Supports: POST /v1/chat/completions
"""
import os, json, time, threading
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI()
model = None
tokenizer = None
model_lock = threading.Lock()

BASE_MODEL   = os.environ.get("BASE_MODEL", "unsloth/Qwen2.5-14B-Instruct-bnb-4bit")
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "/workspace/models/grpo_adapter")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))


def load_model():
    global model, tokenizer
    print(f"Loading base model: {BASE_MODEL}")
    print(f"Loading LoRA adapter: {ADAPTER_PATH}")
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_PATH,
        max_seq_length=4096,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)
    print("Model loaded and ready for inference.")


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "grpo-checkpoint"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = MAX_NEW_TOKENS
    temperature: Optional[float] = 0.7
    stop: Optional[List[str]] = None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": "grpo-checkpoint", "object": "model", "created": int(time.time())}]
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        tokenize=True,
        add_generation_prompt=True,
    ).to("cuda")
    with model_lock:
        with torch.no_grad():
            out = model.generate(
                inputs,
                max_new_tokens=req.max_tokens or MAX_NEW_TOKENS,
                do_sample=(req.temperature or 0) > 0,
                temperature=req.temperature or 1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
    new_tokens = out[0][inputs.shape[-1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": inputs.shape[-1], "completion_tokens": len(new_tokens), "total_tokens": inputs.shape[-1] + len(new_tokens)}
    }


if __name__ == "__main__":
    load_model()
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

JOB_SCRIPT = f"""
set -euo pipefail
export PYTHONUNBUFFERED=1
export CUDA_MODULE_LOADING=EAGER
export PIP_BREAK_SYSTEM_PACKAGES=1
export PIP_ROOT_USER_ACTION=ignore

echo "========================================================"
echo "  BLASTRADIUS — GRPO BENCHMARK JOB"
echo "  Model: {HUB_MODEL_ID}"
echo "  Scenarios: {SCENARIOS}"
echo "========================================================"

nvidia-smi

echo "==> CUDA warmup"
ldconfig 2>/dev/null || true
sleep 3
for _attempt in $(seq 1 8); do
  if python3 -c "import torch; assert torch.cuda.is_available(); print('CUDA OK')"; then break; fi
  echo "  [warmup] attempt $_attempt/8, sleep 5s..."
  ldconfig 2>/dev/null || true
  sleep 5
done

echo "==> Installing system deps"
apt-get update -qq && apt-get install -y -qq git build-essential curl

echo "==> Cloning BlastRadius repo (main)"
[ -d /workspace/.git ] && rm -rf /workspace
git clone --depth 1 --branch main https://github.com/Divyansh-9/BlastRadius.git /workspace
cd /workspace

echo "==> Installing Python deps"
python3 -m pip install --quiet --upgrade pip

TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" | tr -d "[:space:]")
echo "torch==${{TORCH_VER}}" > /tmp/pin.txt
export PIP_CONSTRAINT=/tmp/pin.txt

pip install --quiet "transformers==4.51.3" "trl==0.13.0" "peft==0.13.2"
pip install --quiet "bitsandbytes>=0.43.0" "datasets>=2.18.0"
pip install --quiet "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --quiet huggingface_hub python-dotenv
pip install --quiet "uvicorn[standard]" fastapi pydantic plotly networkx
pip uninstall -y torchao 2>/dev/null || true

echo "==> CUDA re-warmup after pip"
ldconfig 2>/dev/null || true && sleep 3
python3 -c "import torch; assert torch.cuda.is_available(); print('Post-pip CUDA OK')"

echo "==> Downloading GRPO LoRA adapter from Hub"
python3 << 'DOWNLOAD'
import os
from huggingface_hub import snapshot_download
hub_id = "{HUB_MODEL_ID}"
out = "/workspace/models/grpo_adapter"
snapshot_download(
    repo_id=hub_id,
    local_dir=out,
    allow_patterns=["grpo_checkpoint/**"],
    token=os.environ.get("HF_TOKEN"),
)
import shutil, os
# Flatten: move grpo_checkpoint/* -> /workspace/models/grpo_adapter/
src = os.path.join(out, "grpo_checkpoint")
if os.path.isdir(src):
    for f in os.listdir(src):
        shutil.move(os.path.join(src, f), os.path.join(out, f))
    os.rmdir(src)
print("Adapter ready at:", out)
import os; [print(" ", f) for f in os.listdir(out)]
DOWNLOAD

echo "==> Writing inference server script"
cat > /workspace/inference_server.py << 'SERVEREOF'
{INFERENCE_SERVER_PY}
SERVEREOF

echo "==> Starting BlastRadius env server on port 7860 (background)"
BASE_MODEL="unsloth/Qwen2.5-14B-Instruct-bnb-4bit" \\
ADAPTER_PATH="/workspace/models/grpo_adapter" \\
python3 -m uvicorn incident_env.server.app:app --host 0.0.0.0 --port 7860 &
ENV_PID=$!
sleep 8
curl -sf http://localhost:7860/health | python3 -c "import sys,json; d=json.load(sys.stdin); print('Env server OK:', d)" || echo "WARNING: env health check soft-failed"

echo "==> Starting Unsloth inference server on port 8000 (background)"
BASE_MODEL="unsloth/Qwen2.5-14B-Instruct-bnb-4bit" \\
ADAPTER_PATH="/workspace/models/grpo_adapter" \\
python3 /workspace/inference_server.py &
INFER_PID=$!

echo "==> Waiting for inference server (up to 3 min)..."
for i in $(seq 1 36); do
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "Inference server ready!"
    break
  fi
  echo "  [infer warmup] attempt $i/36, sleeping 5s..."
  sleep 5
done

echo "==> Running benchmark — scenarios: {SCENARIOS}"
mkdir -p docs/runs
python3 -m agent.benchmark \\
    --model grpo-checkpoint \\
    --scenarios {SCENARIOS} \\
    --output-dir docs/runs \\
    --api-base http://localhost:8000/v1 \\
    --api-key dummy \\
    --env-url http://127.0.0.1:7860

echo "==> Uploading HTML report to HuggingFace Hub"
HUB_MODEL_ID_VAL="{HUB_MODEL_ID}"
python3 - "$HUB_MODEL_ID_VAL" << 'UPLOAD'
import sys, os, glob
from huggingface_hub import HfApi
hub_id = sys.argv[1]
api = HfApi(token=os.environ.get("HF_TOKEN"))
reports = sorted(glob.glob("docs/runs/benchmark_*.html"))
if reports:
    latest = reports[-1]
    report_name = latest.split("/")[-1]
    url = api.upload_file(
        path_or_fileobj=latest,
        path_in_repo=f"benchmark_results/{{report_name}}",
        repo_id=hub_id,
        repo_type="model",
        commit_message="Auto: GRPO benchmark report (post-training)",
    )
    print(f"Report uploaded: {{url}}")
else:
    print("WARNING: No HTML report found.")
UPLOAD

kill $INFER_PID $ENV_PID 2>/dev/null || true
echo "==> ALL DONE"
""".strip()

cmd = [
    "hf", "jobs", "run",
    "--flavor", FLAVOR,
    "--timeout", TIMEOUT,
    "--detach",
    "--secrets", f"HF_TOKEN={HF_TOKEN}",
    "-e", "PYTHONUNBUFFERED=1",
    "-e", f"HUB_MODEL_ID={HUB_MODEL_ID}",
    DOCKER_IMAGE,
    "bash", "-c", JOB_SCRIPT,
]

print("=" * 60)
print(f"  Launching BENCHMARK Job on {FLAVOR}")
print(f"  Timeout:   {TIMEOUT}")
print(f"  Scenarios: {SCENARIOS}")
print(f"  Model:     {HUB_MODEL_ID}")
print(f"  Image:     {DOCKER_IMAGE}")
print("=" * 60)

result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr)
    sys.exit(result.returncode)
