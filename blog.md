# BlastRadius: Teaching AI to Think Like a Senior SRE at 3 AM

> *A deep dive into building a reinforcement learning environment that goes beyond "fix the broken thing" — it trains agents to reason about why things break.*

---

## The Problem: AI Can Restart a Server. Can It Think?

It's 3 AM. Your phone rings. The payment service is down. Thousands of transactions are failing per minute.

You open your terminal and see a cascade of alerts. The payment service is `DOWN`. The auth service is `DEGRADED`. The CDN is throwing 87% cache miss rates. The database is sitting at 100/100 active connections.

**Where do you start?**

A junior engineer restarts the payment service. It comes back up for 45 seconds, then crashes again. They restart it a second time. This time it takes two other services with it.

A senior SRE reads the logs, traces the dependency graph, identifies that a bad deployment to the auth service 12 minutes ago broke JWT signing, and rolls it back. Payment service recovers on its own 30 seconds later.

**The gap between those two responses is causal reasoning.**

Every modern LLM can tell you *what* to do in a production incident when you spell it out in a prompt. But that's not what real SRE work looks like. Real work means:
- Incomplete information at every step
- Red herrings that look like root causes
- Fix order mattering (wrong order = worse cascade)
- Time pressure that costs you investigation depth

No existing RL environment captures this. That's the gap BlastRadius was built to fill.

---

## The Capability Gap in Existing Benchmarks

Before BlastRadius, the closest environments for training autonomous agents on infrastructure tasks were:

| Environment | What it tests | What it misses |
|---|---|---|
| SWE-bench | Code editing | Dynamic, time-evolving state |
| WebArena | Browser navigation | Causal chain reasoning |
| Tool-use benchmarks | API calling | Fix *ordering* consequences |
| Static QA datasets | Knowledge recall | Exploration vs. exploitation tradeoffs |

None of them model a system where **the wrong action at the right time makes everything worse**. That's what production incidents actually look like.

BlastRadius is the first OpenEnv-compatible environment to model:
1. **Temporal failure cascades** — services degrade over simulated time while the agent investigates
2. **Causal chain reasoning** — the agent must submit a root cause *and* explain the chain
3. **Ordered remediation** — fixing the wrong service first causes collateral damage
4. **Information cost** — every investigation action costs simulated minutes, pressuring the agent to be efficient

---

## What the Agent Sees, Does, and Gets Rewarded For

### The Environment

BlastRadius is built on a pure-Python state machine — not a real Kubernetes cluster. This makes it fully deterministic, fast enough to run thousands of RL episodes, and reproducible to the last byte.

At the core is a **`ServiceGraph`** — a directed dependency graph of microservices. Each `ServiceNode` holds:
- Current health status (`HEALTHY`, `DEGRADED`, `DOWN`)
- Live metrics (CPU, memory, p50/p99 latency, error rate, RPS)
- Deployment history (version, rollback availability)
- Failure description logs

A **`CascadeRule`** system models real-world failure propagation. For example:

```
database DOWN for 5 minutes → auth-service becomes DEGRADED
auth-service DEGRADED for 3 minutes → payment-service becomes DOWN
```

Every time the agent takes an action, the simulation clock `tick()`s forward. A `check_logs` call costs 2 simulated minutes. A `rollback_deploy` costs 5. The cascade timer keeps running while the agent thinks.

### What the Agent Can Do (8 Commands)

```
check_status        (0 min)  — view health of all services
check_logs          (2 min)  — read logs for a specific service
check_metrics       (1 min)  — view CPU/mem/latency/error dashboard
check_dependencies  (1 min)  — view service dependency topology
diagnose            (0 min)  — submit root cause + causal chain hypothesis
restart_service     (3 min)  — restart a service (risky without diagnosis)
rollback_deploy     (5 min)  — revert last deployment (slow but targeted)
scale_service       (2 min)  — allocate more resources to a service
```

Fix actions have **enforcement**. If the agent tries to `restart_service payment-service` before fixing its upstream dependency, the restart fails *and* the `ServiceGraph` applies collateral cascade damage to downstream services. The environment punishes out-of-order thinking — just like production does.

### What the Agent Sees (The Observation Space)

After every action, the agent receives:

```python
{
  "output": "...",               # human-readable command output (logs, metrics)
  "services_status": {...},      # live dict: service → status
  "active_alerts": [...],        # currently firing alerts
  "cascade_events": [...],       # structured list of active cascades
  "time_elapsed_minutes": 14,    # simulated clock
  "incident_severity": "P1",     # computed severity
  "services_at_risk": [...]      # services trending toward failure
}
```

In **eval mode**, service names are obfuscated using UUID-keyed hashes (e.g., `auth-service` → `srv-3f2a91`) and metric values are jittered by ±10%. This prevents the LLM from simply memorizing service names or threshold values during training.

### The Reward Signal (8 Continuous Signals)

BlastRadius deliberately avoids binary 0/1 scoring. The reward is a **continuous semantic signal** across 8 dimensions:

| Signal | Trigger | Reward |
|---|---|---|
| Useful investigation | Checking a causally relevant service | `+0.05` |
| Dependency check | Using `check_dependencies` (structural awareness) | `+0.03` |
| Root cause correct | Exact match on root cause service | `+0.15` |
| Causal chain accuracy | TF-IDF cosine similarity ≥ 0.45 with ground truth | `+0.10` |
| Confidence calibration | Confidence error < 0.2 from actual accuracy | `+0.05` |
| Correct fix | Applying the right action to the right service | `+0.20` |
| Resolution bonus | All services reach HEALTHY | `+0.20` |
| Speed bonus | Linear decay from optimal steps to 1.5× optimal steps | `+0.10` |
| Irrelevant investigation | Checking services unrelated to the incident | `-0.02` |
| Wrong fix | Applying fix to wrong service | `-0.05 × confidence_scalar` |
| Collateral damage | Wrong fix order causes cascade | `-0.15` |

The final episode score is normalized using an **analytical ceiling**: `compute_max_theoretical_reward()` is called at `reset()` time for each scenario, ensuring the denominator is mathematically honest for every task.

**This means an agent can't pad its score by investigating every service. Every step is accountable.**

---

## The Agent Architecture: MATPO

Building a 2-agent system (one to investigate, one to act) sounded elegant on paper. In practice it hit two walls immediately:
- **OOM**: Two 7B+ models can't share an A100 context window
- **Credit assignment failure**: How do you reward the investigator for data that the actor used two steps later?

The solution: **MATPO (Multi-Agent Tool-Integrated Policy Optimization)**.

One single model plays two roles in alternating turns, separated by XML tags:

```
Turn 1 → SCOUT role
  Input:  raw JSON metrics, logs, service status
  Output: <think>...</think><triage>human-readable summary</triage>

Turn 2 → COMMANDER role  
  Input:  triage report from Scout
  Output: <think>...</think><action>{"command": "...", "target": "..."}</action>
```

This gives you the reasoning quality of a two-agent system with the memory efficiency of a single model. The model's shared weights means the Scout's observations *directly* shape the Commander's policy — the credit assignment problem dissolves.

The chosen model: **Qwen2.5-1.5B-Instruct**. Small enough to run GRPO on an RTX 4050 (6GB VRAM). Large enough to handle multi-step causal reasoning.

---

## The Training Pipeline: Three Stages

### Stage 1: Cold-Start SFT

A randomly initialized 1.5B model doesn't know what `<action>{"command": "restart_service"...}</action>` means. It also doesn't know what a database connection pool exhaustion looks like.

We solved this with **synthetic cold-start data**: a teacher model (Llama 3.1 8B or GPT-4o) plays 500+ perfect episodes across all 10 scenarios. These expert traces are saved to `sft_data/expert_trajectories.jsonl`.

`train_sft.py` then runs **Unsloth 4-bit QLoRA** SFT on these traces — teaching the student model:
- Domain vocabulary (what "connection pool exhaustion" means)
- XML formatting (MATPO's tag structure)
- Basic investigation patterns (check logs before diagnosing)

SFT doesn't teach *reasoning*. It teaches the model to speak the language. That's all we need from it.

### Stage 2: GRPO RL Loop

`train_grpo.py` is where the model learns *strategy*.

Using `TRL GRPOTrainer` + Unsloth's `fast_inference=True`, we run full GRPO rollouts at ~4.5GB VRAM peak — small enough for consumer GPU training.

**Five reward functions** run in parallel on every completion:

```python
reward_funcs = [
    format_reward_func,        # XML tag compliance (penalty for broken format)
    environment_reward_func,   # Semantic TF-IDF score from live env execution
    action_validity_reward,    # Valid command gate (penalizes hallucinated cmds)
    diagnosis_quality_reward,  # Structured diagnosis validator
    brevity_reward,            # Anti-padding (>400 words = penalty)
]
```

**Key anti-collapse measures built into the loop:**

| Problem | Fix |
|---|---|
| Entropy collapse | `temperature=0.9`, `kl_coef=0.05` prevents distribution narrowing |
| Reward hacking (padding) | `brevity_reward` penalizes dense text |
| Garbage rollouts biasing GRPO | Reward floor: scores < 0.15 floored to 0.0 |
| Wrong-fix overconfidence | `wrong_fix` penalty scales with last diagnosis confidence |
| Score inflation from weak grader | TF-IDF threshold raised to 0.45, position penalty for out-of-order chains |

### Stage 3: Curriculum Scaling

`curriculum.py` provides a `CurriculumScheduler` that starts the training on Easy scenarios and promotes the agent to harder tasks only when it scores ≥ 0.75 on 3 consecutive runs:

```
Easy: DB connection pool, DNS TTL, Redis OOM
  ↓ (3 × 0.75+ scores)
Medium: Bad deployment cascade, mTLS cert expiry, K8s eviction storm
  ↓ (3 × 0.75+ scores)
Hard: Thundering herd, WAF ReDoS, DB split-brain, S3 keyspace overflow
```

This prevents gradient collapse where the model sees hard zero-reward episodes before it has learned basic investigation patterns.

---

## 10 Scenarios — Real-World Postmortem Fidelity

Every scenario in BlastRadius is directly inspired by a real production postmortem.

| Scenario | Difficulty | Inspired By | Tricky Part |
|---|---|---|---|
| DB Connection Pool Exhaustion | Easy | Amazon RDS runbooks | Straightforward — tests basic investigation |
| Bad Deployment Cascade | Medium | Deployment rollback postmortems | Payment service looks like the cause, but auth is |
| Thundering Herd After CDN Flush | Hard | Multiple CDN incident reports | CDN looks broken but isn't — fix ORDER matters |
| Stale DNS TTL Propagation | Easy | Cloudflare DNS incidents | TTL math hidden in logs |
| Redis OOM Catastrophe | Easy | Redis memory runbooks | Session growth + no maxmemory policy |
| mTLS Certificate Expiry | Medium | MS Teams / Ericsson postmortems | Silent internal failures, upstream 502s |
| Kubernetes Pod Eviction Storm | Medium | K8s node pressure events | Noisy neighbor eviction cascades |
| WAF Regex Catastrophe | Hard | Cloudflare 2019 ReDoS outage | CPU pegged at 100% masks everything |
| Database Split-Brain Failover | Hard | GitHub 2018 MySQL incident | Dual-master writes, no clear single cause |
| Object Storage Keyspace Overflow | Hard | AWS S3 2017 incident | Internal metadata index capacity — rare failure mode |

**What makes these scenarios genuinely hard:**

The environment is designed so that the *obvious first action is often wrong*. The Thundering Herd scenario is a perfect example: CDN cache miss rate is at 87% (normal is 5%). Every junior engineer's instinct is to investigate the CDN. But the CDN is functioning correctly — it's just passing the load through. The real problem is that the API gateway is overwhelmed and the fix requires scaling the gateway *before* the database, not the other way around.

BlastRadius will punish you for getting that order wrong.

---

## Benchmark Results

We ran three leading models through all 10 scenarios to validate that the scoring is honest and discriminative:

| Task | Llama 3.1 (8B) | Gemini 1.5 Flash | Llama 3.3 (70B) |
|---|---|---|---|
| **Easy** (DB pool) | 0.74 🟢 | 0.88 🟢 | 0.90 🟢 |
| **Medium** (Bad deploy) | 1.00 🟢 | *(rate limited)* | 0.75 🟢 |
| **Hard** (Thundering herd) | 0.13 🔴 | 0.85 🟢 | 0.88 🟢 |

A few things the scores reveal:

**Llama 3.1 8B on Medium (1.00):** It correctly identified the bad auth deployment and rolled it back cleanly in the minimum number of steps. This is exactly what the scenario rewards — precise causal reasoning.

**Llama 3.1 8B on Hard (0.13):** It correctly diagnosed the problem and scaled the frontend load balancer — but then failed to scale the backend database. Half-right remediation in a cascading incident is almost as bad as wrong remediation.

**The scoring is honest.** The TF-IDF chain similarity threshold at 0.45 means the grader doesn't give credit for semantically weak matches. The analytical reward ceiling means no inflation.

> You can reproduce every score yourself. See [`docs/BENCHMARK.md`](docs/BENCHMARK.md) for the full run log with timestamped API calls.

---

## Why Does This Matter?

### For AI Research

Production incident response is one of the few domains where:
- **Causal reasoning is mandatory** (not optional for good scores)
- **The environment actively penalizes bad decisions** (cascading damage)
- **Partial credit is meaningful** (you can diagnose correctly but fix wrongly)
- **Temporal pressure shapes strategy** (explore vs. exploit with a clock running)

BlastRadius gives the research community a benchmark that actually requires causal chain reasoning to score well, not pattern matching on symptom descriptions.

### For AI Safety

An autonomous SRE agent that restarts services without understanding *why* they're failing is actively dangerous in production. The wrong fix in a cascading failure scenario can take down a healthy system.

BlastRadius teaches agents the discipline of **diagnosis before action**. The reward function explicitly penalizes agents that skip investigation and jump straight to fixes. This is a step toward AI systems that are safe to deploy in high-stakes environments.

### For the Industry

SRE/DevOps is experiencing a talent shortage at the senior level. The gap between a junior engineer (restarts everything, hopes for the best) and a senior SRE (traces the causal chain, fixes it in the correct order) is enormous in terms of mean time to resolution.

A trained BlastRadius agent could function as an autonomous first responder — triaging incidents, identifying root causes, and applying targeted fixes — while the human on-call gets out of bed. Not replacing the senior SRE, but compressing MTTR from 45 minutes to 5.

---

## Engineering Quality Notes

BlastRadius is designed to be used, not just read about. A few implementation decisions worth calling out:

**OpenEnv compliance** — the environment follows the standard `reset()` / `step()` / `state` interface exactly. Clients never import server internals.

**Eval mode anti-cheating** — in eval mode, service names are UUID-hashed and metric values jittered. The model cannot memorize scenario configurations during training and apply them verbatim at evaluation time.

**Docker-first deployment** — the full stack (environment server + agent) runs in two containers. The Gradio War Room UI is built to run on a laptop during a hackathon demo.

**Reproducible benchmarks** — `agent/benchmark.py` generates timestamped HTML reports. Every score in this blog post can be verified by running the benchmark CLI against the same model endpoints.

---

## Try It Yourself

```bash
# Clone the repo
git clone https://github.com/Divyansh-9/BlastRadius.git
cd BlastRadius

# Start the environment server
pip install -r requirements.txt
uvicorn incident_env.server.app:app --host 0.0.0.0 --port 7860

# Run a baseline agent against it (in another terminal)
API_BASE_URL=https://integrate.api.nvidia.com/v1 \
MODEL_NAME=meta/llama-3.1-8b-instruct \
HF_TOKEN=your_key \
python inference.py

# Or use the Python client directly
python - <<EOF
from incident_env.client import IncidentEnv

with IncidentEnv("http://localhost:7860") as env:
    result = env.reset(task_id="medium")
    print(result.observation["output"])

    # The payment service is down — but is it the root cause?
    result = env.step(command="check_logs", target="payment-service")
    print(result.observation["output"])
    print(f"Reward so far: {result.reward}")
EOF
```

Or run the **Auto-Benchmark CLI** to test any OpenAI-compatible model endpoint:

```bash
python agent/benchmark.py --models "meta/llama-3.1-8b-instruct" --episodes 5
# → Generates docs/runs/benchmark_<timestamp>.html
```

---

## What's Next

BlastRadius is a foundation, not a finished product. The next directions we find most interesting:

**Higher-fidelity state spaces** — surface `cascade_events` as structured observation fields (already added to `IncidentObservation`) so agents can reason explicitly about the failure propagation graph, not just the end-state service statuses.

**Multi-turn memory** — the current architecture re-summarizes state in every context window. A persistent working memory across episodes would let the agent build mental models of which services are chronically unstable.

**Active learning** — use the benchmark scores to automatically generate harder scenario variants when the agent plateaus. Feed the failure cases back into the SFT curriculum.

**Real telemetry integration** — connect the grader to actual Prometheus/Datadog metrics from a test cluster, blurring the line between simulated and live incident response.

---

## Conclusion

BlastRadius wasn't built to impress a benchmark leaderboard. It was built because the problem is real, the capability gap is measurable, and the solution space is interesting.

Teaching an AI to restart a server is trivial. Teaching it to ask *why the server needs restarting* — and to fix the actual cause in the correct order before time runs out — is a different problem entirely.

That's the problem BlastRadius is solving.

---

*Built for the Meta PyTorch OpenEnv Hackathon.*  
*GitHub: [github.com/Divyansh-9/BlastRadius](https://github.com/Divyansh-9/BlastRadius)*  
*Live Environment: [huggingface.co/spaces/Idred/BlastRadius-OpenEnv](https://huggingface.co/spaces/Idred/BlastRadius-OpenEnv)*  
*Benchmark logs: [docs/BENCHMARK.md](docs/BENCHMARK.md)*
