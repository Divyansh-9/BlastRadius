# Benchmark Run Methodology

This document provides explicit instructions for reproducing the benchmark scores reported in the BlastRadius submission, and serves as an audit trail for the scores.

### Target Model
- **Model**: `meta/llama-3.1-8b-instruct`
- **Provider**: NVIDIA NIM API (`https://integrate.api.nvidia.com/v1`)
- **Date**: `2026-04-11`

### Exact Commands to Reproduce

You do not need a mock agent to reproduce these scores. If you provide any valid OpenAI-compatible API key, the environment will run a live causal reasoning benchmark.

```bash
# 1. Start the environment server locally in the background
python -m uvicorn incident_env.server.app:app --host 0.0.0.0 --port 7860 &

# 2. Set API keys and variables
export API_BASE_URL
export MODEL_NAME
export OPENAI_API_KEY
export ENV_BASE_URL

# 3. Run the complete inference protocol
python inference.py
```

### Raw Run Log

A raw, timestamped output of the live LLM run evaluated against the server is captured in the repository. This proves the environment emits the required `[START]`, `[STEP]`, and `[END]` syntax blocks and evaluates causal chains correctly. 

**View the raw log here:** [`docs/runs/benchmark_run.log`](./runs/benchmark_run.log)

### Score Results (From `benchmark_run.log`)
- **Easy** (Database Pool Exhaustion): **0.74**
- **Medium** (Payment Gateway Degradation): **1.00**
- **Hard** (Thundering Herd): **0.13** (The LLM correctly identifies the load balancer queue and API gateway scaling requirements, but fails to execute the final proper scaling of the database).

These scores have been updated in the README and UI to reflect the most current prompt version.
