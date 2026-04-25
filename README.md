---
title: BlastRadius
emoji: 💥
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
---

# BlastRadius

An RL environment for training AI agents to respond to production infrastructure incidents.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Quick Start

### Local Setup
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
# Build the container
docker build -t incident-response-env .

# Run the container
docker run -p 7860:7860 incident-response-env
```
Access the Interactive UI at `http://localhost:7860/ui`

## Features

- **Dynamic State:** Failures cascade over time based on a simulated logical clock.
- **Causal Diagnosis:** Agent must explain the causal chain, evaluated via TF-IDF cosine similarity.
- **Costly Actions:** Every action costs simulated time, creating a real explore-vs-exploit tradeoff.
- **Continuous Reward:** Rich 8-signal reward space rather than a binary win/loss.
- **10 Real-World Scenarios:** Includes Redis OOM, K8s Eviction Storms, DB Failovers, and WAF ReDoS.
- **Spot-Aware RL Pipeline:** Production-ready GRPO MLOps pipeline targeting 32B models on consumer budgets.

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | LLM API endpoint for baseline agent | (Required) |
| `MODEL_NAME` | Model identifier | (Required) |
| `HF_TOKEN` | Hugging Face / API token | (Required) |
| `ENV_BASE_URL` | Environment URL | `http://localhost:7860` |

## Documentation

- [Deep Dive Blog](./blog.md)
- [Architecture & Design](./docs/ARCHITECTURE.md)
- [Benchmark Results](./docs/BENCHMARK.md)

## License

MIT
