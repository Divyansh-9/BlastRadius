"""
Baseline Inference Script for IT Incident Response Environment.

Uses the OpenAI API client (compatible with NVIDIA NIMs) to run an
LLM agent against the environment. Produces structured stdout logs
following the [START], [STEP], [END] format required by the hackathon.

Environment variables required:
    API_BASE_URL  — The API endpoint for the LLM
    MODEL_NAME    — The model identifier (e.g., meta/llama-3.1-8b-instruct)
    HF_TOKEN      — Your HuggingFace / API key (used as OPENAI_API_KEY)

Usage:
    API_BASE_URL=https://integrate.api.nvidia.com/v1 \
    MODEL_NAME=meta/llama-3.1-8b-instruct \
    HF_TOKEN=your_key \
    python inference.py
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta/llama-3.1-8b-instruct")
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

# Agent parameters
TEMPERATURE = 0.3
MAX_TOKENS = 1024
MAX_STEPS = 25  # Must match environment's max_steps=25
SUCCESS_SCORE_THRESHOLD = 0.5

# Tasks to evaluate
TASKS = ["easy", "medium", "hard"]

# ---------------------------------------------------------------------------
# System prompt — SRE agent persona
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert SRE responding to a production incident. You must ACT FAST.

CRITICAL RULES:
1. You have MAXIMUM 25 steps total. Do NOT waste them all investigating.
2. Failures SPREAD while you investigate. Every check_logs costs 2 minutes.
3. Follow this STRICT phase plan:
   - Steps 1-2: check_status + check_dependencies (get the big picture)
   - Steps 3-5: check_logs on the 2-3 most broken services
   - Step 6: DIAGNOSE with your root cause theory
   - Steps 7+: APPLY FIXES (restart_service, rollback_deploy, or scale_service)
4. After step 5, you MUST start fixing things. No more investigating.
5. Look for: recent deployments (rollback them), resource exhaustion (scale them), crashed services (restart them)

⚠️ FIX ORDER IS CRITICAL — wrong order causes cascading damage and PENALTIES:
- ALWAYS fix the service that OTHER services depend on FIRST (the upstream service)
- The service that is DOWN and has the most downstream dependents is the true root cause
- Fix the entry point of the cascade chain (closest to traffic source) before fixing downstream
- NEVER scale or restart a downstream service while its upstream dependency is still broken
- Example: if api-gateway is DOWN and database has connection storms, fix api-gateway FIRST
  because the database storm is CAUSED BY api-gateway retries, not the other way around

Available commands (respond with EXACTLY one JSON object):
- {"command": "check_status"}
- {"command": "check_logs", "target": "<service>"}
- {"command": "check_metrics", "target": "<service>"}
- {"command": "check_dependencies"}
- {"command": "diagnose", "parameters": {"root_cause": "<service>", "causal_chain": ["step1", "step2"], "confidence": 0.8}}
- {"command": "restart_service", "target": "<service>"}
- {"command": "rollback_deploy", "target": "<service>"}
- {"command": "scale_service", "target": "<service>", "parameters": {"instances": 4}}

Key signals to look for:
- If logs mention "deployment" or version numbers → rollback_deploy that service
- If a service is DOWN and has no recent deploy → restart_service
- If service A depends on service B, and B is broken → fix B first, THEN fix A
- High connection counts on a service that OTHER services depend on = likely root cause → fix it FIRST
- A service showing "connection storm" or "pool exhausted" may be a VICTIM, not the cause

Respond with ONLY a valid JSON object. No markdown. No explanation."""

# ---------------------------------------------------------------------------
# Structured logging (mandatory format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str):
    print(json.dumps({
        "type": "[START]",
        "task": task,
        "env": env,
        "model": model,
        "timestamp": time.time(),
    }), flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None):
    entry = {
        "type": "[STEP]",
        "step": step,
        "action": action,
        "reward": reward,
        "done": done,
        "timestamp": time.time(),
    }
    if error:
        entry["error"] = error
    print(json.dumps(entry), flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    print(json.dumps({
        "type": "[END]",
        "success": success,
        "steps": steps,
        "score": score,
        "rewards": rewards,
        "timestamp": time.time(),
    }), flush=True)


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def get_model_action(
    client: OpenAI,
    step_num: int,
    observation: Dict[str, Any],
    last_reward: float,
    history: List[str],
) -> Dict[str, Any]:
    """Ask the LLM what action to take next."""

    # Determine phase urgency
    if step_num <= 2:
        phase_msg = "PHASE: INVESTIGATE — check_status and check_dependencies first."
    elif step_num <= 5:
        phase_msg = "PHASE: INVESTIGATE — check_logs on the most broken services."
    elif step_num <= 7:
        phase_msg = "⚠️ PHASE: DIAGNOSE & FIX — You MUST submit a diagnose command NOW, then start fixing."
    else:
        phase_msg = "🔴 PHASE: FIX — STOP investigating. Apply fixes NOW or you will run out of steps!"

    # Build context from observation
    user_prompt = f"""Step {step_num}/20 | Reward: {last_reward:+.4f} | {phase_msg}
Time elapsed: {observation.get('time_elapsed_minutes', 0)} min | Severity: {observation.get('incident_severity', 'unknown')}

Service Status: {json.dumps(observation.get('services_status', {}))}

Alerts: {'; '.join(observation.get('active_alerts', ['None']))}

Last Output (summary):
{observation.get('output', 'No output')[:1500]}

Hint: {observation.get('hint', '')}

History: {'; '.join(history[-3:])}

Respond with ONE JSON object — your next action."""

    max_retries = 5
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            text = (completion.choices[0].message.content or "").strip()

            # Parse JSON from response (handle markdown code blocks)
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            action = json.loads(text)
            return action

        except json.JSONDecodeError:
            print(f"[DEBUG] Failed to parse model response as JSON: {text[:200]}", flush=True)
            return {"command": "check_status"}
        except Exception as exc:
            err_str = str(exc)
            if "429" in err_str and attempt < max_retries - 1:
                wait = min(5 * (2 ** attempt), 30)
                print(f"[DEBUG] Rate limited, retrying in {wait}s (attempt {attempt+1}/{max_retries})", flush=True)
                time.sleep(wait)
                continue
            print(f"[DEBUG] Model request failed: {exc}", flush=True)
            return {"command": "check_status"}


# ---------------------------------------------------------------------------
# Environment interaction (via HTTP)
# ---------------------------------------------------------------------------

import requests

def env_reset(base_url: str, task_id: str) -> Dict[str, Any]:
    resp = requests.post(f"{base_url}/reset", json={"task_id": task_id})
    resp.raise_for_status()
    return resp.json()


def env_step(base_url: str, action: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(f"{base_url}/step", json=action)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, base_url: str, task_id: str) -> float:
    """Run inference on a single task. Returns the final score."""

    log_start(task=task_id, env="incident-response-env", model=MODEL_NAME)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        # Reset environment
        result = env_reset(base_url, task_id)
        observation = result["observation"]
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.get("done", False):
                break

            # Get action from LLM
            action = get_model_action(client, step, observation, last_reward, history)

            # Execute action
            result = env_step(base_url, action)
            observation = result["observation"]
            reward = result.get("reward", 0.0)
            done = result.get("done", False)

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            # Log step
            action_str = json.dumps(action)
            log_step(step=step, action=action_str, reward=reward, done=done)

            # Track history for context
            history.append(
                f"Step {step}: {action.get('command', '?')} "
                f"target={action.get('target', '')} → reward {reward:+.4f}"
            )

            if done:
                break

        # Get final score from environment if available (preferred — includes penalties)
        if "info" in result and "final_score" in result["info"]:
            score = result["info"]["final_score"]
        elif rewards:
            # Fallback: use cumulative sum (including negatives) so penalties count
            score = min(max(sum(rewards), 0.0), 1.0)
        else:
            score = 0.0

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} error: {exc}", flush=True)

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


def main():
    """Run baseline inference on all tasks."""
    if not API_KEY:
        print("ERROR: Set HF_TOKEN or OPENAI_API_KEY environment variable", flush=True)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print(f"{'='*60}", flush=True)
    print(f"IT Incident Response Environment — Baseline Inference", flush=True)
    print(f"Model: {MODEL_NAME}", flush=True)
    print(f"API: {API_BASE_URL}", flush=True)
    print(f"Environment: {ENV_BASE_URL}", flush=True)
    print(f"{'='*60}", flush=True)

    scores = {}
    for task_id in TASKS:
        print(f"\n{'─'*40}", flush=True)
        print(f"Running task: {task_id}", flush=True)
        print(f"{'─'*40}", flush=True)

        score = run_task(client, ENV_BASE_URL, task_id)
        scores[task_id] = score
        print(f"\n✅ Task '{task_id}' score: {score:.4f}", flush=True)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print(f"RESULTS SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    for task_id, score in scores.items():
        emoji = "🟢" if score >= 0.7 else "🟡" if score >= 0.4 else "🔴"
        print(f"  {emoji} {task_id:10s}: {score:.4f}", flush=True)
    avg = sum(scores.values()) / len(scores) if scores else 0
    print(f"\n  📊 Average:   {avg:.4f}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
