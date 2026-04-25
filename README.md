---
title: BlastRadius
emoji: 💥
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
---

# IT Incident Response Environment (OpenEnv)

> **An RL environment for training AI agents to respond to production infrastructure incidents.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> 📖 **Read the deep-dive blog post**: [blog.md](blog.md) — covers the full problem, environment design, reward math, training pipeline, and benchmark results in one place.

## 🎯 What Is This?

It's 3 AM. Your phone blows up. The website is down. Users are complaining.

You open your laptop and see a dashboard of services — some red, some yellow. Logs are scrolling with errors. Metrics are spiking in weird ways.

**This environment drops an AI agent into that exact scenario.**

The agent can investigate logs, check metrics, trace dependencies, diagnose root causes, and apply fixes. Every action costs simulated time, and **failures spread via a simulated logical clock** as the incident progresses — creating genuine urgency and a real explore-vs-exploit tradeoff.

### What Makes This Different

| Feature | Most Env's | This Env |
|---|---|---|
| State | Static puzzle | **Dynamic** — failures cascade over time |
| Diagnosis | Fix something → done | Agent must **explain the causal chain** |
| Actions | Free | **Cost simulated time** — exploration tradeoff |
| Reward | Binary (0/1) | **Continuous** with 8 reward signals |
| Red herrings | None | **Misleading signals** that test real reasoning |

## 📋 Environment Description

### Motivation

Real SRE/DevOps incident response requires:
- **Causal reasoning** — finding *why* something broke, not just *what* broke
- **Prioritization under pressure** — failures spread while you investigate
- **Ordered remediation** — fixing things in the wrong order makes it worse

No existing OpenEnv environment captures these dynamics. This fills that gap.

### Action Space (8 Commands)

| Command | Time Cost | Description |
|---|---|---|
| `check_status` | 0 min | View health of all services |
| `check_logs` | 2 min | View recent logs for a service |
| `check_metrics` | 1 min | View CPU/memory/latency/errors |
| `check_dependencies` | 1 min | View service dependency graph |
| `diagnose` | 0 min | Submit root cause + causal chain hypothesis |
| `restart_service` | 3 min | Restart a service (risky) |
| `rollback_deploy` | 5 min | Roll back last deployment |
| `scale_service` | 2 min | Scale service resources |

### Observation Space

Each observation includes:
- **`output`**: Human-readable command output (logs, metrics, status)
- **`services_status`**: `{service_name: "healthy"|"degraded"|"down"}`
- **`active_alerts`**: List of firing alerts
- **`time_elapsed_minutes`**: Simulated time since incident start
- **`incident_severity`**: `P1` / `P2` / `P3`
- **`services_at_risk`**: Services trending toward failure
- **`hint`**: Grading feedback from last action

### Reward Function

Continuous reward signal (not binary):

| Signal | Reward | Trigger |
|---|---|---|
| Useful investigation | +0.05 | Checking relevant service |
| Root cause correct | +0.15 | Correct diagnosis |
| Causal chain accurate | +0.10 | Matching ground truth chain |
| Correct fix | +0.20 | Fix that resolves a service |
| Speed bonus | +0.10 | Solving in optimal steps |
| Irrelevant investigation | -0.02 | Checking wrong service |
| Wrong fix | -0.05 | Restart/rollback wrong target |
| Collateral damage | -0.15 | Wrong fix order causes cascade |

Final score normalized to **[0.0, 1.0]**.

## 🎮 Tasks (10 Scenarios — All Shipped)

### Easy: Database Connection Pool Exhaustion
**Expected score: 0.8-1.0**

The database has exhausted its connection pool. API gateway is returning 503s. Fix is straightforward if you investigate the right service.

*Tests: Basic investigation and single-service fix.*

### Medium: Bad Deployment Cascade
**Expected score: 0.5-0.7**

Payment service is DOWN — but it's a victim, not the cause. Auth service deployed broken JWT signing 12 minutes ago. Payment logs *say* "auth token validation failed" — a red herring that tempts you to restart payment.

*Tests: Root cause analysis vs. symptom chasing. Causal chain reasoning.*

### Hard: Thundering Herd After CDN Cache Invalidation
**Expected score: 0.4-0.6**

CDN cache was invalidated (routine, NOT the cause). All traffic hits the backend, overwhelming the API gateway, which cascades into a database connection storm. CDN metrics look scary but it's functioning correctly. Fix ORDER matters — wrong order causes thundering herd.

*Tests: Misleading signals, multi-service causal reasoning, ordered remediation.*

### Real-World Postmortem Scenarios (All Implemented):
- **Stale DNS TTL Propagation (Easy)** `easy_dns_propagation`: Route failures post-migration (inspired by Cloudflare DNS drops).
- **Redis OOM Catastrophe (Easy)** `easy_redis_oom`: Unbounded session allocations trigger kernel OOM kills.
- **Internal mTLS Certificate Expiry (Medium)** `medium_cert_expiry`: Silent internal mesh connection failures causing upstream 502s (inspired by MS Teams/Ericsson).
- **Kubernetes Pod Eviction Storm (Medium)** `medium_k8s_eviction`: Noisy neighbor exhausts node memory, triggering eviction cascades.
- **WAF Regex Catastrophe (Hard)** `hard_regex_catastrophe`: ReDoS WAF backtracking pegs CPU to 100% masking root cause (inspired by Cloudflare 2019).
- **Database Split-Brain Failover (Hard)** `hard_db_failover`: Dual-master writes after temporary network partition (inspired by GitHub 2018).
- **Object Storage Keyspace Overflow (Hard)** `hard_s3_keyspace_overflow`: Batch workloads exhausting internal metadata index capacity (inspired by AWS S3 2017).

## 🤖 Multi-Model AI Benchmark
We benchmarked 3 leading models against the incidents. BlastRadius grades reasoning effectively because simply restarting all services blindly drastically penalizes scores.

| Task | Llama 3.1 (8B) | Gemini 1.5 Flash | Llama 3.3 (70B) |
|---|---|---|---|
| **Easy** | 0.74 🟢 | 0.88 🟢 | 0.90 🟢 |
| **Medium** | 1.00 🟢 | *(hit rate limits)* | 0.75 🟢 |
| **Hard** | 0.13 🔴 | 0.85 🟢 | 0.88 🟢 |

> ⓘ **Note**: The environment evaluates causal reasoning strictly using TF-IDF cosine similarity. For example, Llama 3.1 scored a perfect `1.0` on Medium by cleanly rolling back an upstream deployment, but struggled on Hard (`0.13`) because it correctly diagnosed and scaled the frontend load balancer but subsequently failed to properly scale the backend database.
>
> *Scores reflect honest normalization. The maximum possible reward in the environment acts as the denominator, so agents must earn every single decimal point.*
> **You can verify this exact run yourself.** See the raw timestamped LLM log in [docs/BENCHMARK.md](docs/BENCHMARK.md).

## 🧠 MLOps: Spot-Aware GRPO Training on A100

To surpass the benchmarks and hit 97%+ accuracy, we provide a production-ready RL training pipeline designed for $30/teammate compute budgets. 

It targets 32B reasoning models (e.g., `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` or `Qwen/Qwen2.5-Coder-32B-Instruct`) and utilizes **Spot Instances**, **WandB live tracking**, and **Async Checkpointing**. 

To survive Spot instance preemptions with zero wasted GPU time, the `train_grpo.py` loop hooks into `SIGTERM` and forces an emergency push to the Hugging Face Hub 30 seconds before the instance is killed.

```bash
# Example A100 Spot Training Job
WANDB_API_KEY=your_key python -m agent.train_grpo \
    --model models/sft_checkpoint \
    --data sft_data/expert_trajectories.jsonl \
    --output models/grpo_checkpoint \
    --hardware-profile a100 \
    --wandb-entity your_wandb_org \
    --hub-model-id your_hf_org/BlastRadius-GRPO
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

### Docker

```bash
# Build
docker build -t incident-response-env .

# Run
docker run -p 7860:7860 incident-response-env

# Test health
curl http://localhost:7860/health

# Access Interactive UI
http://localhost:7860/ui
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

# Check state
curl http://localhost:7860/state
```

### Python Client

```python
from incident_env.client import IncidentEnv

with IncidentEnv("http://localhost:7860") as env:
    result = env.reset(task_id="medium")
    print(result.observation["output"])

    result = env.step(command="check_logs", target="auth-service")
    print(result.observation["output"])
    print(f"Reward: {result.reward}")
```

## 📊 Evaluation Methodology

Causal chains are evaluated using TF-IDF cosine similarity. This means agents receive partial credit for paraphrased but semantically correct diagnostics, rather than brittle substring matching. Additionally, score normalization operates with accurate scenario ceilings (e.g., maximum reward 1.22 on Hard scenarios), generating mathematically honest final metrics clamped between `[0.0, 1.0]`.

## 🏗️ Architecture

```
incident_env/
├── models.py                    # Typed Action/Observation/State models
├── client.py                    # HTTP client for remote usage
├── server/
│   ├── app.py                   # FastAPI server (OpenEnv HTTP API)
│   ├── incident_environment.py  # Core Environment (reset/step/state)
│   ├── scenarios/               # 10 pre-built failure scenarios
│   │   ├── easy.py              # DB pool exhaustion
│   │   ├── medium.py            # Bad deployment cascade
│   │   ├── hard.py              # Thundering herd (CDN + fix-order)
│   │   ├── dns_propagation.py   # Stale DNS TTL
│   │   ├── redis_memory_leak.py # Redis OOM
│   │   ├── cert_expiry.py       # mTLS cert expiry
│   │   ├── k8s_eviction.py      # K8s pod eviction storm
│   │   ├── regex_catastrophe.py # WAF ReDoS
│   │   ├── db_failover.py       # Split-brain failover
│   │   └── s3_keyspace.py       # Object storage overflow
│   └── engine/                  # Simulation core
│       ├── infrastructure.py    # Service graph + temporal state machine
│       ├── log_generator.py     # Realistic log generation
│       ├── metrics_generator.py # Dashboard-style metrics
│       └── grader.py            # Causal chain evaluation + scoring
openenv.yaml                     # OpenEnv manifest (all 10 tasks)
Dockerfile                       # Container for HF Spaces
docker-compose.yml               # Full stack (server + agent) local run
Dockerfile.agent                 # Agent-only container
inference.py                     # Baseline LLM agent
requirements.txt
tests/
└── test_environment.py          # 45 tests covering all components
```

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `API_BASE_URL` | Yes | LLM API endpoint |
| `MODEL_NAME` | Yes | Model identifier |
| `HF_TOKEN` | Yes | API key |
| `ENV_BASE_URL` | No | Environment URL (default: localhost:7860) |

## License

MIT
