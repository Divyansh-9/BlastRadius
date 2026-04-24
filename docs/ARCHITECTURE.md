# BlastRadius Deep Architecture Documentation

Welcome to the internal technical documentation for **BlastRadius**, a production-grade Reinforcement Learning environment and MATPO-driven autonomous agent simulator for SRE/DevOps incident response.

This document breaks down the repository into its core components, explaining the "why" and "how" behind the mathematical grading, infrastructure simulation, and the 6GB VRAM-optimized reinforcement learning pipeline.

---

## 1. Environment Engine (`incident_env/server/engine/`)

The core of BlastRadius is not a real Kubernetes cluster, but a pure-Python state machine. This allows for deterministic reinforcement learning without the overhead of spinning up real containers.

### `infrastructure.py` (The State Machine)
- **`ServiceNode`**: Represents a microservice (e.g., `auth-service`). It tracks its current `ServiceStatus` (HEALTHY, DEGRADED, DOWN), resource metrics, and deployment history.
- **`CascadeRule`**: The logic that models failures spreading over time. Example: If `database` is down for 5 simulated minutes, `auth-service` transitions to DEGRADED.
- **`ServiceGraph`**: The temporal evolution engine. Its core method `tick(minutes)` advances the simulation clock, evaluates cascade rules, and propagates collateral damage if fixes are applied out of order.

### `grader.py` (The RL Reward Signal)
The original engine used brittle substring matching. We rebuilt this into a **TF-IDF Semantic Engine**.
- **`_grade_diagnosis()`**: When the agent submits a root cause hypothesis, the text is vectorized using `TfidfVectorizer`. We compute the cosine similarity against the ground-truth hypothesis. 
- **Anti-Cheat Mechanisms**: If the agent submits extremely long paragraphs to "guess" every possible answer, the grader applies a dense-text penalty.
- **Speed Bonus**: A non-linear decay curve `max(0, 1.0 - (steps / 25)^2)` rewards the agent for fixing the issue in fewer steps, accelerating GRPO convergence.

### `log_generator.py` & `metrics_generator.py`
These provide deterministic "observations" for the LLM. If a service is marked DEGRADED, the `metrics_generator` artificially spikes the p99 latency and error rates in the JSON output, which the Agent's Scout module must read and interpret.

---

## 2. Environment Controller (`incident_environment.py`)

This is the bridge between the infrastructure state machine and the Agent. It implements the standard RL `step()` function.
- **Action Execution**: Routes the agent's 8 commands (e.g., `check_status`, `scale_service`) to the `ServiceGraph`.
- **Time Cost**: Every action advances the `tick()` clock. A `diagnose` action takes 0 minutes, but a `rollback_deploy` takes 5 minutes, giving failure cascades time to trigger.
- **Normalization**: The `max_total_reward` from the scenario configuration normalizes the final episode score perfectly between `0.0` and `1.0`.

---

## 3. The MATPO RL Architecture (`agent/`)

The agent stack abandons traditional "Two-Model" architectures (which cause OOM errors and credit assignment failure) in favor of **MATPO (Multi-Agent Tool-Integrated Policy Optimization)**. 

One single model (`Qwen2.5-1.5B`) acts as both the data analyzer (Scout) and the decision-maker (Commander).

### `prompts.py`
Defines strict XML-style schemas. 
- **Scout** receives raw JSON metrics and outputs a human-readable `<triage>` report.
- **Commander** reads the triage report, thinks via `<think>` tags, and executes a JSON action via `<action>` tags.

### `orchestrator.py`
The production runner. It calls the OpenAI-compatible API endpoints iteratively.
- **`run_episode()`**: Generates `Rollout` objects containing the full state history for training.
- **`run_episode_stream()`**: Yields token-by-token generation and state updates specifically designed for the Gradio War Room UI.

### `generate_sft_data.py` (Stage 1: Cold-Start)
To prevent "Entropy Collapse" where a randomly initialized RL agent just guesses invalid JSON, we use a Teacher Model (e.g., `Llama 3.1 8B` or `GPT-4o`) to play 500+ perfect episodes. It saves these traces to `expert_trajectories.jsonl`.

### `train_sft.py` (Stage 2: QLoRA)
Takes the expert trajectories and applies Supervised Fine-Tuning using **Unsloth 4-bit QLoRA**. This teaches the 1.5B model the domain vocabulary and XML formatting.

### `train_grpo.py` (Stage 3: RL Loop)
The crown jewel. It utilizes `TRL GRPOTrainer` combined with Unsloth's `fast_inference=True` to share weights between generation and training.
- **Memory Optimization**: By utilizing `adamw_8bit`, `r=32` LoRA, and strictly limiting `num_generations=4`, the entire GRPO loop is restricted to **~4.5GB VRAM**, allowing it to train natively on consumer GPUs (like an RTX 4050).
- **Reward Functions**: Employs `format_reward_func` (verifying XML tag obedience) and `environment_reward_func` (spawning a cloned `IncidentEnvironment` to calculate the semantic TF-IDF score).

---

## 4. Presentation Layer (`war_room_ui.py`)

A Gradio-based live dashboard engineered for hackathon presentations.
- **Plotly Network Graph**: Dynamically plots the `services_status` dict as an interactive topology map, mapping statuses to visual colors (Green/Yellow/Red).
- **Streaming Generators**: Binds directly to the `run_episode_stream` of the `MATPOOrchestrator`, writing the Agent's Chain-of-Thought live to dual hacker-themed terminal windows.
