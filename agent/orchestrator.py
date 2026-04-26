"""
MATPO Orchestrator — Single Model, Dual Role
=============================================
This replaces the old dual-model (Scout 1B + Commander 3B) design.

HOW IT WORKS:
─────────────
One model (Qwen2.5-14B-Instruct, 4-bit) plays both roles using different
system prompts. For each environment step:

  Step 1: Model receives SCOUT_SYSTEM_PROMPT + raw observation
          → outputs a <triage> report
  Step 2: Model receives COMMANDER_SYSTEM_PROMPT + triage report + history
          → outputs an <action> JSON

WHY THIS IS BETTER THAN TWO MODELS:
────────────────────────────────────
1. Credit assignment: GRPO trains ONE set of weights for both roles.
   When triage improves, decisions improve automatically.
2. VRAM: ~14GB inference (14B 4-bit) vs ~28GB for two models.
3. Latency: Both prompts can share KV cache context.
4. Self-improving: Both roles get better via RL, not just the Commander.

USAGE:
──────
  # For inference/evaluation (uses API endpoint or local model)
  python -m agent.orchestrator --task easy --endpoint http://localhost:8000/v1

  # For rollout collection (saves trajectories to disk for GRPO)
  python -m agent.orchestrator --task easy --save-rollouts rollouts/
"""

import json
import re
import os
import sys
import time
import argparse
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import requests # type: ignore
from openai import OpenAI

# Add project root to path so we can import incident_env
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.prompts import (
    SCOUT_SYSTEM_PROMPT,
    COMMANDER_SYSTEM_PROMPT,
    SCOUT_TAGS,
)


# ─────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────

@dataclass
class RolloutStep:
    """One step in a trajectory. Saved for SFT/GRPO training."""
    step_number: int
    role: str                          # "scout" or "commander"
    system_prompt: str
    user_prompt: str                   # Fix #6: Store REAL prompts, not placeholders
    model_response: str
    parsed_action: Optional[Dict]      # The JSON action (commander only)
    reward: float                      # Reward from grader
    cumulative_reward: float
    observation: Dict[str, Any]        # Compact observation snapshot
    triage_report: str                 # Scout's output (for commander context)


@dataclass
class Rollout:
    """A complete episode trajectory."""
    task_id: str
    steps: List[RolloutStep] = field(default_factory=list)
    final_score: float = 0.0
    total_steps: int = 0
    resolved: bool = False
    truncated: bool = False            # Fix #8: distinguish timeout from resolution


# ─────────────────────────────────────────────────────────────
# Parsing Utilities
# ─────────────────────────────────────────────────────────────

def extract_between_tags(text: str, open_tag: str, close_tag: str) -> str:
    """Extract content between XML-style tags. Returns empty string if not found."""
    pattern = re.escape(open_tag) + r"(.*?)" + re.escape(close_tag)
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def parse_action_json(text: str) -> Dict[str, Any]:
    """
    Extract and parse the JSON action from the Commander's response.
    Handles multiple formats:
    - Raw JSON
    - JSON inside <action> tags
    - JSON inside markdown code blocks

    Fix #5: Returns _parse_failure sentinel instead of silently defaulting
    to check_status, so the grader can apply a negative signal.
    """
    # Try <action> tags first
    action_text = extract_between_tags(text, "<action>", "</action>")
    if action_text:
        text = action_text

    # Try markdown code blocks
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            code = parts[1]
            if code.startswith("json"):
                code = code[4:]
            text = code.strip()

    # Clean and parse
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Last resort: find first { ... } block
        brace_match = re.search(r'\{[^{}]*\}', text)
        if brace_match:
            try:
                return json.loads(brace_match.group())
            except json.JSONDecodeError:
                pass
        # Fix #5: Return sentinel instead of silently succeeding
        return {"command": "_parse_failure", "target": None}


# ─────────────────────────────────────────────────────────────
# Triage Quality Scorer (Fix #1: Decouple Scout reward)
# ─────────────────────────────────────────────────────────────

def score_triage(triage: str, observation: Dict[str, Any]) -> float:
    """
    Independent reward for the Scout's triage quality.
    
    Fix #1: The Scout must NOT receive the Commander's env reward.
    Instead, we score the triage by checking whether it correctly
    identifies unhealthy services by name.
    """
    services = observation.get("services_status", {})
    triage_lower = triage.lower()
    
    # Count unhealthy services mentioned in the triage
    unhealthy = [name for name, status in services.items()
                 if str(status).upper() in ("DEGRADED", "DOWN")]
    
    if not unhealthy:
        # All healthy — scout should say so; give small baseline
        return 0.05
    
    hits = sum(1 for svc in unhealthy if svc.lower() in triage_lower)
    coverage = hits / len(unhealthy)
    
    # Base reward: 0.0-0.15 based on coverage of unhealthy services
    reward = 0.15 * coverage
    
    # Bonus for mentioning severity
    severity = observation.get("incident_severity", "")
    if severity and severity.lower() in triage_lower:
        reward += 0.05
    
    return round(reward, 4)


# ─────────────────────────────────────────────────────────────
# Phase Heuristic (Fix #4: State-aware, not step-count-based)
# ─────────────────────────────────────────────────────────────

def get_phase(observation: Dict[str, Any], step_num: int) -> str:
    """
    Fix #4: Determine episode phase from env state, not just step count.
    
    Hard scenarios can require 10+ investigation steps. Telling the model
    to DIAGNOSE at step 7 when it's only checked 2 services causes
    premature action and grader penalties.
    """
    services = observation.get("services_status", {})
    unhealthy_count = sum(
        1 for v in services.values()
        if str(v).upper() in ("DEGRADED", "DOWN")
    )
    
    if unhealthy_count == 0:
        return "🔴 FIX — All services show healthy. Submit final fix or verify resolution."
    
    if step_num <= 3 or unhealthy_count > 3:
        return "🔍 INVESTIGATE — Understand the blast radius first. Check status, logs, metrics."
    
    if step_num <= 6:
        return "🔍 DEEP INVESTIGATE — Narrow down the root cause. Check dependencies and logs of suspect services."
    
    return "⚠️ DIAGNOSE + FIX — Identify root cause and apply targeted remediation."


# ─────────────────────────────────────────────────────────────
# MATPO Orchestrator
# ─────────────────────────────────────────────────────────────

class MATPOOrchestrator:
    """
    Runs a BlastRadius episode using a single LLM in two roles.

    The model is called via an OpenAI-compatible API endpoint.
    This works with:
    - Local vLLM/Ollama servers
    - NVIDIA NIM endpoints
    - HuggingFace Inference Endpoints
    - Any OpenAI-compatible API
    """

    def __init__(
        self,
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "not-needed",
        # Default to the 14B 4-bit model the training pipeline actually
        # produces. The old 32B default OOMs A100 80GB at full precision and
        # silently misled anyone running the orchestrator/benchmark from CLI.
        model_name: str = "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
        env_base_url: str = "http://localhost:7860",
        temperature: float = 0.3,
        max_tokens: int = 512,
    ):
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model_name = model_name
        self.env_base_url = env_base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ── Environment Interface ────────────────────────────────

    def _env_reset(self, task_id: str) -> Dict[str, Any]:
        resp = requests.post(
            f"{self.env_base_url}/reset",
            json={"task_id": task_id}
        )
        resp.raise_for_status()
        return resp.json()

    def _env_step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        resp = requests.post(
            f"{self.env_base_url}/step",
            json=action,
        )
        resp.raise_for_status()
        return resp.json()

    # ── LLM Calls ────────────────────────────────────────────

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Single LLM call with retry logic for rate limits."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as e:
                err = str(e)
                if "429" in err and attempt < max_retries - 1:
                    wait = min(5 * (2 ** attempt), 30)
                    print(f"  [RATE LIMIT] Retrying in {wait}s...", flush=True)
                    time.sleep(wait)
                    continue
                print(f"  [LLM ERROR] {e}", flush=True)
                return ""
        return ""

    def _call_llm_stream(self, system_prompt: str, user_prompt: str):
        """Streaming LLM call that yields text chunks."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True
                )
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                return
            except Exception as e:
                err = str(e)
                if "429" in err and attempt < max_retries - 1:
                    wait = min(5 * (2 ** attempt), 30)
                    time.sleep(wait)
                    continue
                yield f"\n[LLM ERROR] {str(e)}\n"
                return
        yield "\n[RATE LIMIT ERROR]\n"

    # ── Shared Prompt Builders (Fix #7: Single source of truth) ──

    def _build_scout_user_prompt(self, observation: Dict[str, Any], history: List[str]) -> str:
        """Build the Scout's user prompt. Used by both run_episode and run_episode_stream."""
        return f"""ENVIRONMENT OBSERVATION:
Services: {json.dumps(observation.get('services_status', {}), indent=1)}
Alerts: {json.dumps(observation.get('active_alerts', []))}
Time Elapsed: {observation.get('time_elapsed_minutes', 0)} min
Severity: {observation.get('incident_severity', 'unknown')}
Output: {str(observation.get('output', ''))[:1200]}

Recent History: {'; '.join(history[-5:]) if history else 'Episode start'}"""

    def _build_commander_user_prompt(
        self, triage: str, step_num: int, last_reward: float, 
        history: List[str], observation: Dict[str, Any]
    ) -> str:
        """Build the Commander's user prompt. Used by both run_episode and run_episode_stream."""
        phase = get_phase(observation, step_num)  # Fix #4: state-aware phase
        return f"""Step {step_num}/25 | Last Reward: {last_reward:+.4f} | {phase}

[SCOUT TRIAGE REPORT]
{triage}

[EPISODE HISTORY]
{chr(10).join(history[-5:]) if history else 'No actions taken yet.'}

Based on the Scout's triage and episode phase, choose your next action.
Respond with <think>your reasoning</think> then <action>JSON</action>."""

    # ── Role Execution ───────────────────────────────────────

    def run_scout(self, observation: Dict[str, Any], history: List[str]) -> Tuple[str, str]:
        """
        ROLE A: Scout — reads raw JSON, outputs triage report.
        Returns: (full_response, triage_report)
        """
        user_prompt = self._build_scout_user_prompt(observation, history)
        full_response = self._call_llm(SCOUT_SYSTEM_PROMPT, user_prompt)

        # Extract the triage report from between tags
        triage = extract_between_tags(full_response, *SCOUT_TAGS)
        if not triage:
            # Fallback: use the full response as triage
            triage = full_response[:500]

        return full_response, triage

    def run_commander(
        self,
        triage_report: str,
        step_num: int,
        last_reward: float,
        history: List[str],
        observation: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        """
        ROLE B: Commander — reads triage report + history, emits JSON action.
        Returns: (full_response, parsed_action_dict)
        """
        user_prompt = self._build_commander_user_prompt(
            triage_report, step_num, last_reward, history, observation
        )
        full_response = self._call_llm(COMMANDER_SYSTEM_PROMPT, user_prompt)
        action = parse_action_json(full_response)

        return full_response, action

    # ── Episode Runner ───────────────────────────────────────

    def run_episode(
        self,
        task_id: str,
        max_steps: int = 25,
        verbose: bool = True,
    ) -> Rollout:
        """
        Run a complete episode against the BlastRadius environment.

        For each step:
        1. Scout analyzes the raw observation → triage report
        2. Commander reads triage → emits action JSON
        3. Action is sent to environment → reward received
        4. Everything is logged into the Rollout for training

        Returns a Rollout object containing the full trajectory.
        """
        rollout = Rollout(task_id=task_id)
        history: List[str] = []
        cumulative_reward = 0.0

        # Reset environment
        if verbose:
            print(f"\n{'='*60}")
            print(f"  EPISODE: {task_id}")
            print(f"{'='*60}")

        reset_result = self._env_reset(task_id)
        observation = reset_result.get("observation", {})

        for step_num in range(1, max_steps + 1):
            if verbose:
                print(f"\n── Step {step_num}/{max_steps} ──")

            # ── ROLE A: Scout Triage ──
            scout_user_prompt = self._build_scout_user_prompt(observation, history)
            scout_response, triage = self.run_scout(observation, history)
            if verbose:
                print(f"  [SCOUT] {triage[:120]}...")

            # Fix #1: Score the Scout's triage independently
            scout_reward = score_triage(triage, observation)

            # ── ROLE B: Commander Decision ──
            last_reward = rollout.steps[-1].reward if rollout.steps else 0.0
            cmdr_user_prompt = self._build_commander_user_prompt(
                triage, step_num, last_reward, history, observation
            )
            cmdr_response, action = self.run_commander(
                triage, step_num, last_reward, history, observation
            )
            if verbose:
                print(f"  [CMDR]  {json.dumps(action)}")

            # ── Execute Action (guard against _parse_failure → 422) ──
            if action.get("command") == "_parse_failure":
                print(f"  [WARN] Parse failure — model produced malformed output. Skipping env step.", flush=True)
                reward = -0.05   # penalty for bad format
                done = False
                env_result = {"reward": reward, "done": done, "observation": observation, "info": {}}
            else:
                env_result = self._env_step(action)
                reward = env_result.get("reward", 0.0)
                done = env_result.get("done", False)
                observation = env_result.get("observation", {})
            cumulative_reward += reward

            if verbose:
                print(f"  [ENV]   reward={reward:+.4f}  cumulative={cumulative_reward:+.4f}  done={done}")

            # ── Record Steps ──
            # Fix #1: Scout gets its own independent triage-quality reward
            # Fix #6: Store REAL prompts, not "[raw observation]" placeholders
            scout_step = RolloutStep(
                step_number=step_num,
                role="scout",
                system_prompt=SCOUT_SYSTEM_PROMPT,
                user_prompt=scout_user_prompt,
                model_response=scout_response,
                parsed_action=None,
                reward=scout_reward,
                cumulative_reward=cumulative_reward,
                observation={"services_status": observation.get("services_status", {}),
                             "active_alerts": observation.get("active_alerts", [])},
                triage_report=triage,
            )
            cmdr_step = RolloutStep(
                step_number=step_num,
                role="commander",
                system_prompt=COMMANDER_SYSTEM_PROMPT,
                user_prompt=cmdr_user_prompt,
                model_response=cmdr_response,
                parsed_action=action,
                reward=reward,
                cumulative_reward=cumulative_reward,
                observation={"services_status": observation.get("services_status", {}),
                             "active_alerts": observation.get("active_alerts", [])},
                triage_report=triage,
            )
            rollout.steps.extend([scout_step, cmdr_step])

            # ── Update History ──
            cmd = action.get("command", "unknown")
            tgt = action.get("target", "")
            history.append(f"Step {step_num}: {cmd}({tgt}) → reward={reward:+.4f}")

            if done:
                if verbose:
                    print(f"\n  ✅ Episode finished at step {step_num}")
                break

        # ── Finalize ──
        rollout.final_score = cumulative_reward
        rollout.total_steps = len(history)
        info = env_result.get("info", {})
        rollout.resolved = info.get("is_resolved", False)
        rollout.truncated = info.get("truncated", False)  # Fix #8

        if verbose:
            print(f"\n{'─'*60}")
            print(f"  RESULT: score={rollout.final_score:.4f}  steps={rollout.total_steps}  resolved={rollout.resolved}  truncated={rollout.truncated}")
            print(f"{'─'*60}\n")

        return rollout

    def run_episode_stream(self, task_id: str, max_steps: int = 25):
        """
        Generator for Gradio War Room UI. 
        Fix #7: Uses shared prompt builders to avoid train/inference mismatch.
        Yields: (observation, scout_text_accum, cmdr_text_accum, last_reward, is_done)
        """
        history: List[str] = []
        cumulative_reward = 0.0

        reset_result = self._env_reset(task_id)
        observation = reset_result.get("observation", {})
        
        scout_log = ""
        cmdr_log = ""
        
        yield observation, scout_log, cmdr_log, 0.0, False

        for step_num in range(1, max_steps + 1):
            scout_log += f"\n\n{'='*20}\n🤖 STEP {step_num} | SCOUT\n{'='*20}\n"
            yield observation, scout_log, cmdr_log, cumulative_reward, False

            # Fix #7: Use shared prompt builder
            user_prompt = self._build_scout_user_prompt(observation, history)
            scout_full = ""
            for chunk in self._call_llm_stream(SCOUT_SYSTEM_PROMPT, user_prompt):
                scout_full += chunk
                scout_log += chunk
                yield observation, scout_log, cmdr_log, cumulative_reward, False
            
            triage = extract_between_tags(scout_full, *SCOUT_TAGS)
            if not triage: 
                triage = scout_full[:500]

            cmdr_log += f"\n\n{'='*20}\n🧠 STEP {step_num} | COMMANDER\n{'='*20}\n"
            yield observation, scout_log, cmdr_log, cumulative_reward, False

            # Fix #7: Use shared prompt builder for commander too
            last_reward = cumulative_reward
            user_prompt = self._build_commander_user_prompt(
                triage, step_num, last_reward, history, observation
            )
            cmdr_full = ""
            for chunk in self._call_llm_stream(COMMANDER_SYSTEM_PROMPT, user_prompt):
                cmdr_full += chunk
                cmdr_log += chunk
                yield observation, scout_log, cmdr_log, cumulative_reward, False

            action = parse_action_json(cmdr_full)
            env_result = self._env_step(action)
            reward = env_result.get("reward", 0.0)
            done = env_result.get("done", False)
            observation = env_result.get("observation", {})
            cumulative_reward += reward

            cmd = action.get("command", "unknown")
            tgt = action.get("target", "")
            history.append(f"Step {step_num}: {cmd}({tgt}) → reward={reward:+.4f}")
            
            cmdr_log += f"\n\n[ENVIRONMENT] Executed {cmd} on {tgt} -> Reward: {reward:+.4f}"
            yield observation, scout_log, cmdr_log, cumulative_reward, done

            if done:
                break

    def save_rollout(self, rollout: Rollout, output_dir: str) -> str:
        """Save a rollout to disk as JSONL for training."""
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{rollout.task_id}_{int(time.time())}.jsonl"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w") as f:
            for step in rollout.steps:
                f.write(json.dumps(asdict(step)) + "\n")

        return filepath


# ─────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MATPO Orchestrator for BlastRadius")
    parser.add_argument("--task", default="easy", help="Scenario task_id (easy, medium, hard, etc.)")
    parser.add_argument("--endpoint", default=os.environ.get("API_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"))
    parser.add_argument("--env-url", default=os.environ.get("ENV_BASE_URL", "http://localhost:7860"))
    parser.add_argument("--api-key", default=os.environ.get("HF_TOKEN", "not-needed"))
    parser.add_argument("--save-rollouts", default=None, help="Directory to save rollout trajectories")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--quiet", action="store_true", help="Suppress step-by-step output")
    args = parser.parse_args()

    orchestrator = MATPOOrchestrator(
        api_base=args.endpoint,
        api_key=args.api_key,
        model_name=args.model,
        env_base_url=args.env_url,
    )

    scores = []
    for ep in range(args.episodes):
        print(f"\n{'#'*60}")
        print(f"  Episode {ep + 1}/{args.episodes}")
        print(f"{'#'*60}")

        rollout = orchestrator.run_episode(args.task, verbose=not args.quiet)
        scores.append(rollout.final_score)

        if args.save_rollouts:
            path = orchestrator.save_rollout(rollout, args.save_rollouts)
            print(f"  📁 Saved rollout to {path}")

    # Summary
    avg = sum(scores) / len(scores) if scores else 0
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {len(scores)} episodes | avg_score={avg:.4f}")
    print(f"  Scores: {[f'{s:.4f}' for s in scores]}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
