"""
BlastRadius MATPO Agent
========================
Single-model dual-role architecture for SRE incident response.

Pipeline:
1. generate_sft_data.py  → Expert CoT trajectories (cold-start data)
2. train_sft.py          → QLoRA SFT on expert data (teaches format)
3. train_grpo.py         → MATPO-GRPO RL training (teaches reasoning)
4. orchestrator.py       → Inference runner for evaluation
"""
