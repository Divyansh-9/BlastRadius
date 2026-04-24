"""
Cold-Start SFT Data Generator
==============================
PURPOSE:
This script generates expert Chain-of-Thought (CoT) trajectories for the
Cold-Start SFT phase (Stage 1 of the DeepSeek R1 recipe).

WHY THIS STAGE EXISTS:
Small models (1.5B) attempting GRPO from scratch often suffer "entropy
collapse" — they start outputting identical responses and training stalls.
By first fine-tuning on ~500 expert demonstrations, the model learns:
1. The correct OUTPUT FORMAT (<think>...</think><action>...</action>)
2. The REASONING STYLE (step-by-step causal analysis)
3. The DOMAIN VOCABULARY (service names, SRE terminology)

HOW IT WORKS:
─────────────
1. We instantiate the BlastRadius environment directly (no HTTP server)
2. For each episode, we use a "teacher" model (GPT-4/Claude via API)
   to play through the scenario with detailed chain-of-thought
3. The teacher's responses are saved in the exact format our training
   expects: {role, system_prompt, user_prompt, response} per turn
4. Output is JSONL — one line per training example

USAGE:
──────
  # Using OpenAI API as teacher
  export TEACHER_API_KEY="sk-..."
  export TEACHER_API_BASE="https://api.openai.com/v1"
  export TEACHER_MODEL="gpt-4o-mini"
  python -m agent.generate_sft_data --episodes 50 --output sft_data/

  # Using a local model as teacher (cheaper but lower quality)
  export TEACHER_API_BASE="http://localhost:8000/v1"
  export TEACHER_MODEL="Qwen/Qwen2.5-7B-Instruct"
  python -m agent.generate_sft_data --episodes 50 --output sft_data/
"""

import json
import os
import sys
import time
import argparse
import random
from pathlib import Path
from typing import Dict, Any, List

from openai import OpenAI

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from incident_env.server.incident_environment import IncidentEnvironment
from incident_env.models import IncidentAction
from agent.prompts import (
    SCOUT_SYSTEM_PROMPT,
    COMMANDER_SYSTEM_PROMPT,
)


# ─────────────────────────────────────────────────────────────
# Teacher Model Configuration
# ─────────────────────────────────────────────────────────────

TEACHER_API_BASE = os.environ.get("TEACHER_API_BASE", "https://api.openai.com/v1")
TEACHER_API_KEY = os.environ.get("TEACHER_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "gpt-4o-mini")


# ─────────────────────────────────────────────────────────────
# Expert Episode Runner
# ─────────────────────────────────────────────────────────────

class ExpertEpisodeRunner:
    """
    Runs episodes using a powerful teacher model to generate
    expert-quality trajectories in our exact training format.
    """

    def __init__(self):
        self.client = OpenAI(base_url=TEACHER_API_BASE, api_key=TEACHER_API_KEY)
        self.env = IncidentEnvironment()

    def _teacher_call(self, system_prompt: str, user_prompt: str) -> str:
        """Call the teacher model with retry logic."""
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=TEACHER_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.7,  # Some diversity for training data
                    max_tokens=768,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                if "429" in str(e):
                    time.sleep(5 * (attempt + 1))
                    continue
                print(f"  [TEACHER ERROR] {e}")
                return ""
        return ""

    def run_expert_episode(self, task_id: str) -> List[Dict[str, Any]]:
        """
        Run one full episode with the teacher model, producing
        training examples in our exact dual-role format.

        Returns a list of training examples, each with:
        - role: "scout" or "commander"
        - system_prompt: the role's system prompt
        - user_prompt: what the model sees as input
        - response: the teacher's chain-of-thought response
        - reward: the environment's reward for that step
        - task_id: which scenario
        """
        training_examples = []
        history: List[str] = []

        # Reset environment directly (no HTTP)
        obs = self.env.reset(task_id=task_id)
        observation = obs if isinstance(obs, dict) else obs.__dict__ if hasattr(obs, '__dict__') else {"output": str(obs)}

        # Try to get the observation dict properly
        state = self.env.state
        if isinstance(state, dict):
            observation = state
        elif hasattr(state, '__dict__'):
            observation = state.__dict__

        step_num = 0
        done = False
        last_reward = 0.0

        while not done and step_num < 20:
            step_num += 1

            # ── SCOUT TURN ──
            # Build the same prompt structure the student model will see
            scout_user_prompt = self._build_scout_prompt(observation, history)
            scout_response = self._teacher_call(SCOUT_SYSTEM_PROMPT, scout_user_prompt)

            # Extract triage from the teacher's response
            triage = self._extract_triage(scout_response)

            training_examples.append({
                "role": "scout",
                "system_prompt": SCOUT_SYSTEM_PROMPT,
                "user_prompt": scout_user_prompt,
                "response": scout_response,
                "task_id": task_id,
                "step": step_num,
            })

            # ── COMMANDER TURN ──
            cmdr_user_prompt = self._build_commander_prompt(
                triage, step_num, last_reward, history
            )
            cmdr_response = self._teacher_call(COMMANDER_SYSTEM_PROMPT, cmdr_user_prompt)

            # Parse the action
            action_dict = self._parse_action(cmdr_response)

            training_examples.append({
                "role": "commander",
                "system_prompt": COMMANDER_SYSTEM_PROMPT,
                "user_prompt": cmdr_user_prompt,
                "response": cmdr_response,
                "task_id": task_id,
                "step": step_num,
            })

            # ── EXECUTE ACTION ──
            try:
                action = IncidentAction(
                    command=action_dict.get("command", "check_status"),
                    target=action_dict.get("target", None),
                    parameters=action_dict.get("parameters", {}),
                )
                result = self.env.step(action)

                # Handle different return types
                if isinstance(result, dict):
                    last_reward = result.get("reward", 0.0)
                    done = result.get("done", False)
                    observation = result.get("observation", observation)
                elif hasattr(result, 'reward'):
                    last_reward = result.reward
                    done = getattr(result, 'done', False)
                    new_state = self.env.state
                    observation = new_state if isinstance(new_state, dict) else getattr(new_state, '__dict__', observation)
                else:
                    last_reward = 0.0

                # Tag the reward onto the last two training examples
                training_examples[-1]["reward"] = last_reward
                training_examples[-2]["reward"] = last_reward

            except Exception as e:
                print(f"  [ENV ERROR] Step {step_num}: {e}")
                done = True

            # Update history
            cmd = action_dict.get("command", "?")
            tgt = action_dict.get("target", "")
            history.append(f"Step {step_num}: {cmd}({tgt}) → reward={last_reward:+.4f}")

        return training_examples

    def _build_scout_prompt(self, observation: Dict, history: List[str]) -> str:
        """Build the exact same prompt format the student will see."""
        # Handle observation as dict or object
        if isinstance(observation, dict):
            services = observation.get("services_status", observation.get("output", "N/A"))
            alerts = observation.get("active_alerts", [])
            time_elapsed = observation.get("time_elapsed_minutes", 0)
            severity = observation.get("incident_severity", "unknown")
            output = observation.get("output", "")
        else:
            services = str(observation)[:500]
            alerts = []
            time_elapsed = 0
            severity = "unknown"
            output = str(observation)[:500]

        return f"""ENVIRONMENT OBSERVATION:
Services: {json.dumps(services, indent=1) if isinstance(services, (dict, list)) else str(services)[:600]}
Alerts: {json.dumps(alerts) if isinstance(alerts, list) else str(alerts)}
Time Elapsed: {time_elapsed} min
Severity: {severity}
Output: {str(output)[:1200]}

Recent History: {'; '.join(history[-3:]) if history else 'Episode start'}"""

    def _build_commander_prompt(
        self, triage: str, step_num: int, last_reward: float, history: List[str]
    ) -> str:
        if step_num <= 2:
            phase = "🔍 INVESTIGATE — Build situational awareness first."
        elif step_num <= 5:
            phase = "🔍 DEEP INVESTIGATE — Check logs/dependencies of suspect services."
        elif step_num <= 8:
            phase = "⚠️ DIAGNOSE — Submit your root cause analysis NOW."
        else:
            phase = "🔴 FIX — Apply fixes immediately. Time is running out!"

        return f"""Step {step_num}/25 | Last Reward: {last_reward:+.4f} | {phase}

[SCOUT TRIAGE REPORT]
{triage}

[EPISODE HISTORY]
{chr(10).join(history[-5:]) if history else 'No actions taken yet.'}

Based on the Scout's triage and episode phase, choose your next action.
Respond with <think>your reasoning</think> then <action>JSON</action>."""

    def _extract_triage(self, response: str) -> str:
        """Extract triage from between tags, with fallback."""
        import re
        match = re.search(r"<triage>(.*?)</triage>", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response[:500]

    def _parse_action(self, response: str) -> Dict:
        """Parse action JSON from commander response."""
        import re

        # Try <action> tags
        match = re.search(r"<action>(.*?)</action>", response, re.DOTALL)
        text = match.group(1).strip() if match else response

        # Try markdown code blocks
        if "```" in text:
            parts = text.split("```")
            if len(parts) >= 2:
                code = parts[1]
                if code.startswith("json"):
                    code = code[4:]
                text = code.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            brace_match = re.search(r'\{[^{}]*\}', text)
            if brace_match:
                try:
                    return json.loads(brace_match.group())
                except json.JSONDecodeError:
                    pass
            return {"command": "check_status"}


# ─────────────────────────────────────────────────────────────
# Main: Generate Dataset
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate Cold-Start SFT data for BlastRadius")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes to generate")
    parser.add_argument("--output", default="sft_data", help="Output directory")
    parser.add_argument("--tasks", nargs="+", default=["easy", "medium", "hard"],
                        help="Scenario task IDs to cycle through")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    output_file = os.path.join(args.output, "expert_trajectories.jsonl")

    runner = ExpertEpisodeRunner()
    total_examples = 0

    print(f"Generating {args.episodes} expert episodes → {output_file}")
    print(f"Teacher: {TEACHER_MODEL} @ {TEACHER_API_BASE}")
    print(f"Tasks: {args.tasks}")
    print()

    with open(output_file, "w") as f:
        for ep in range(args.episodes):
            task_id = args.tasks[ep % len(args.tasks)]
            print(f"Episode {ep+1}/{args.episodes} [{task_id}]...", end=" ", flush=True)

            try:
                examples = runner.run_expert_episode(task_id)
                for ex in examples:
                    f.write(json.dumps(ex) + "\n")
                total_examples += len(examples)
                print(f"✓ {len(examples)} examples (total: {total_examples})")
            except Exception as e:
                print(f"✗ {e}")
                continue

    print(f"\n{'='*60}")
    print(f"  Generated {total_examples} training examples across {args.episodes} episodes")
    print(f"  Saved to: {output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
