---
title: BlastRadius
emoji: 💥
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
---

# BlastRadius: The 3 AM Incident Simulator

> **An RL environment and training pipeline for teaching AI agents to respond to production infrastructure incidents.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

> 📖 **Read the deep-dive blog post in [blog.md](blog.md)**. It covers the full problem, environment design, reward math, training pipeline, and benchmark results in one place.

## 🎯 The Story: The 3 AM Incident

**03:00 AM.** Your phone is buzzing relentlessly. The pager app is screaming. The main website is down, payments are failing, and users are complaining on social media. 

You open your laptop. A dashboard of microservices blinks back at you in angry reds and yellows. Logs are scrolling with cryptic exceptions. Metrics—CPU, memory, latency—are spiking in unpredictable ways. The clock is ticking.

You trace the errors. It looks like the payment service is failing. You restart it. *Big mistake.* The payment service was just a victim. The actual root cause was a broken JWT signing deployment on the auth service 12 minutes ago. By restarting the payment service, you've triggered a thundering herd, and now the database connection pool is exhausted. You've just turned a P2 into a P1.

**This environment drops an AI agent into that exact scenario.**

The agent must investigate logs, check metrics, trace dependencies, diagnose root causes, and apply fixes. Every single action costs simulated time. As the minutes tick by, failures spread across the system like a contagion via a simulated logical clock. The environment creates genuine urgency and forces a real explore-vs-exploit tradeoff.

### What Makes This Different

| Feature | Typical Environments | BlastRadius |
|---|---|---|
| State | Static puzzle | **Dynamic**: Failures cascade over time |
| Diagnosis | Fix something and you are done | Agent must **explain the causal chain** |
| Actions | Free | **Cost simulated time**: Exploration tradeoff |
| Reward | Binary (0 or 1) | **Continuous** with 8 specific reward signals |
| Red herrings | None | **Misleading signals** that test real reasoning |

## 📋 Environment Description

Real SRE and DevOps incident response requires:
- **Causal reasoning**: Finding *why* something broke, not just *what* broke.
- **Prioritization under pressure**: Failures spread while you investigate.
- **Ordered remediation**: Fixing things in the wrong order makes the situation worse.

### Action Space (8 Commands)

| Command | Time Cost | Description |
|---|---|---|
| `check_status` | 0 min | View health of all services |
| `check_logs` | 2 min | View recent logs for a service |
| `check_metrics` | 1 min | View CPU, memory, latency, and errors |
| `check_dependencies` | 1 min | View service dependency graph |
| `diagnose` | 0 min | Submit root cause and causal chain hypothesis |
| `restart_service` | 3 min | Restart a service (risky) |
| `rollback_deploy` | 5 min | Roll back the last deployment |
| `scale_service` | 2 min | Scale service resources |

### Reward Function

The environment uses a continuous reward signal instead of a binary pass or fail:

| Signal | Reward | Trigger |
|---|---|---|
| Useful investigation | +0.05 | Checking a relevant service |
| Root cause correct | +0.15 | Submitting the correct diagnosis |
| Causal chain accurate | +0.10 | Matching the ground truth chain |
| Correct fix | +0.20 | Applying a fix that resolves a service |
| Speed bonus | +0.10 | Solving the incident in optimal steps |
| Irrelevant investigation | -0.02 | Checking the wrong service |
| Wrong fix | -0.05 | Restarting or rolling back the wrong target |
| Collateral damage | -0.15 | Wrong fix order causes a cascade |

The final score is normalized to **[0.0, 1.0]**.

## 🎮 Tasks (10 Scenarios Shipped)

We ship with 10 historically accurate, real-world postmortem scenarios.

### Easy: Database Connection Pool Exhaustion
The database has exhausted its connection pool. The API gateway is returning 503 errors. 
*Tests: Basic investigation and single-service fix.*

### Medium: Bad Deployment Cascade
The payment service is DOWN, but it is merely a victim, not the cause. The auth service deployed broken JWT signing 12 minutes ago. Payment logs *say* "auth token validation failed", which is a red herring.
*Tests: Root cause analysis vs. symptom chasing. Causal chain reasoning.*

### Hard: Thundering Herd After CDN Cache Invalidation
The CDN cache was invalidated. This is routine and NOT the cause. All traffic hits the backend, overwhelming the API gateway, which cascades into a database connection storm. CDN metrics look scary but it is functioning correctly. Fix ORDER matters immensely. 
*Tests: Misleading signals, multi-service causal reasoning, ordered remediation.*

### Real-World Scenarios
- **Stale DNS TTL Propagation (Easy)** `easy_dns_propagation`: Route failures post-migration (inspired by Cloudflare DNS drops).
- **Redis OOM Catastrophe (Easy)** `easy_redis_oom`: Unbounded session allocations trigger kernel OOM kills.
- **Internal mTLS Certificate Expiry (Medium)** `medium_cert_expiry`: Silent internal mesh connection failures causing upstream 502s (inspired by MS Teams and Ericsson).
- **Kubernetes Pod Eviction Storm (Medium)** `medium_k8s_eviction`: Noisy neighbor exhausts node memory, triggering eviction cascades.
- **WAF Regex Catastrophe (Hard)** `hard_regex_catastrophe`: ReDoS WAF backtracking pegs CPU to 100 percent, masking the root cause (inspired by Cloudflare 2019).
- **Database Split-Brain Failover (Hard)** `hard_db_failover`: Dual-master writes after temporary network partition (inspired by GitHub 2018).
- **Object Storage Keyspace Overflow (Hard)** `hard_s3_keyspace_overflow`: Batch workloads exhausting internal metadata index capacity (inspired by AWS S3 2017).

## 🤖 MATPO Architecture

The agent stack abandons traditional "Two-Model" architectures (which cause OOM errors and credit assignment failure) in favor of **MATPO (Multi-Agent Tool-Integrated Policy Optimization)**. 

Instead of having a separate Scout model and Commander model, MATPO uses a single model with a unified schema. This allows us to train one cohesive policy using GRPO, keeping VRAM usage drastically lower while retaining the explicit reasoning capabilities of multi-agent patterns. For a deep dive into the MATPO schema and credit assignment mechanics, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## 🧠 MLOps: Spot-Aware GRPO Training on A100

We provide a production-ready RL training pipeline designed for a low compute budget. It targets 14B reasoning models (default: `unsloth/Qwen2.5-14B-Instruct-bnb-4bit`) and utilizes **Spot Instances**, **WandB live tracking**, and **Async Checkpointing**.

To survive Spot instance preemptions with zero wasted GPU time, the `train_grpo.py` loop hooks into `SIGTERM` and forces an emergency push to the Hugging Face Hub 30 seconds before the instance is killed.

### Credentials

Copy `.env.example` to `.env` (one level up from the repo, or inside it — both locations are picked up by the notebook) and fill in:

| Variable | Used by | Notes |
|---|---|---|
| `HF_TOKEN` | GRPO checkpoint pushes, hub auto-recovery | Needs **write** scope on `HUB_MODEL_ID` |
| `HUB_MODEL_ID` | `train_grpo.py --hub-model-id` | e.g. `your-org/BlastRadius-GRPO-Checkpoints` |
| `WANDB_API_KEY` | WandB run init | Get from https://wandb.ai/authorize |
| `WANDB_ENTITY` | WandB org/user namespace | e.g. `your-wandb-team` |
| `WANDB_PROJECT` | WandB project name | Defaults to `blastradius-grpo` |

For **HF Jobs** (remote A100), set the same variables as **Job secrets** in the HF UI — the notebook reads them from `os.environ` either way.

### Generating SFT Data
```bash
python -m agent.generate_sft_data \
    --teacher-model gpt-4o-mini \
    --episodes 100 \
    --output sft_data/expert_trajectories.jsonl
```

### Starting the Training Run
```bash
# Option A: rely on .env (loaded automatically by the notebook)
python -m agent.train_grpo \
    --model models/sft_checkpoint \
    --data sft_data/expert_trajectories.jsonl \
    --output models/grpo_checkpoint \
    --hardware-profile a100 \
    --wandb-entity "$WANDB_ENTITY" \
    --hub-model-id "$HUB_MODEL_ID"

# Option B: pass everything inline
WANDB_API_KEY=your_key python -m agent.train_grpo \
    --model models/sft_checkpoint \
    --data sft_data/expert_trajectories.jsonl \
    --output models/grpo_checkpoint \
    --hardware-profile a100 \
    --wandb-entity your_wandb_org \
    --hub-model-id your_hf_org/BlastRadius-GRPO
```

### Verifying the Model
```bash
python -m agent.validate_save --model models/grpo_checkpoint
```

## 🚀 Setup & Usage

### Quick Start (Local)

```bash
# Install dependencies
pip install -r requirements.txt

# Start the environment server
uvicorn incident_env.server.app:app --host 0.0.0.0 --port 7860

# Run the baseline agent (in another terminal)
API_BASE_URL=https://integrate.api.nvidia.com/v1 \
MODEL_NAME=meta/llama-3.1-8b-instruct \
HF_TOKEN=your_key \
python inference.py
```

### API Usage

```bash
# Reset environment
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy"}'

# Take an action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"command": "check_status"}'
```

## 🏗️ Architecture & Codebase Health

The entire repository adheres to strict `ruff` and `mypy` typing standards, ensuring absolute stability during multi-day A100 training runs.

```text
incident_env/
├── models.py                    # Typed Action/Observation/State models
├── client.py                    # HTTP client for remote usage
├── server/
│   ├── app.py                   # FastAPI server (OpenEnv HTTP API)
│   ├── incident_environment.py  # Core Environment (reset/step/state)
│   ├── scenarios/               # 10 pre-built failure scenarios
│   └── engine/                  # Simulation core
│       ├── infrastructure.py    # Service graph + temporal state machine
│       ├── log_generator.py     # Realistic log generation
│       ├── metrics_generator.py # Dashboard-style metrics
│       └── grader.py            # Causal chain evaluation + scoring
agent/
├── generate_sft_data.py         # Generates expert trajectories via GPT-4o
├── train_grpo.py                # Spot-aware TRL GRPO training loop
├── validate_save.py             # Validation script for checkpoint integrity
└── orchestrator.py              # Logic mapping and LLM handling
tests/
└── test_debug_audit.py          # Comprehensive integration testing
```

## License

MIT
