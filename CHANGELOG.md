# Changelog

All notable changes to this project will be documented in this file.

## [2026-04-25] - Blog Documentation

### Added
- `blog.md`: Comprehensive story-driven technical blog post covering the problem statement, environment design, reward function, MATPO agent architecture, training pipeline, benchmark results, and future directions.

---

## [2026-04-24] - Hackathon Freeze

### Added
- `agent/benchmark.py`: Auto-Benchmark CLI to mass-evaluate LLMs across all 10 scenarios and output an HTML report.
- `agent/curriculum.py`: `CurriculumScheduler` added to handle progressive difficulty scaling across scenarios.
- `_NOISE_LOG_POOL`: Realistic noise added to `generate_logs()` when `eval_mode=True` to prevent LLM log memorization.
- `compute_max_theoretical_reward`: Analytical baseline computation to perfectly normalize scores per scenario difficulty.
- `cascade_events` field added to `IncidentObservation` for cleaner LLM state parsing.

### Updated
- **Grader Metrics**: `chain_similarity_threshold` bumped from 0.20 to 0.45 for stricter causal reasoning scoring.
- **Grader Logic**: Added position-penalty (0.7x) for out-of-order causal chain steps.
- **Grader Fix Penalties**: `wrong_fix` penalty now scales dynamically based on the confidence of the most recent diagnosis (overconfidence penalty).
- **Grader Resolution**: Allowed one `diagnose` revision at a 50% reward penalty instead of blocking updates completely.
- **Grader Discovery**: `check_dependencies` now grants a positive reward signal (+0.03).
- **Environment**: Synchronized `max_steps=20` across the entire codebase (Grader, SFT, Prompts, UI).

### Fixed
- **Infrastructure**: Added `_auto_recover_dependents()` to `restart_service()` and `rollback_deploy()` so downstream cascade victims automatically recover when root causes are solved.
- **Reward Math**: GRPO Reward floor ensures any total episode reward `< 0.15` is floored to `0.0` to prevent PPO advantage collapse on universally bad rollouts.
- **Docker**: Updated `Dockerfile` and `Dockerfile.agent` to correctly include the `server/`, `agent/`, and `incident_env/` directories.
- **Dependencies**: Synced Gradio to `>=5.0.0` and included `plotly` in `pyproject.toml`.
