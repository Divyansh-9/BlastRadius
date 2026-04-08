"""
Grading engine for the incident response environment.

Computes per-step rewards and final episode scores.
Includes causal chain evaluation — the key differentiator.

Reward ranges are clamped to [0.0, 1.0] for final scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GradeResult:
    """Result of grading a single step or final episode."""
    reward: float = 0.0
    breakdown: Dict[str, float] = field(default_factory=dict)
    feedback: str = ""


@dataclass
class ScenarioGradingConfig:
    """
    Grading configuration for a specific scenario.

    Defines the ground truth that the grader evaluates against.
    """
    root_cause_service: str = ""
    root_cause_description: str = ""
    ground_truth_causal_chain: List[str] = field(default_factory=list)
    correct_fix_actions: List[Dict[str, str]] = field(default_factory=list)
    correct_fix_order: List[str] = field(default_factory=list)
    useful_investigation_targets: List[str] = field(default_factory=list)
    max_optimal_steps: int = 6
    max_total_reward: float = 1.0


class Grader:
    """
    Scores agent performance with rich, continuous reward signals.

    Reward Sources
    --------------
    +0.05  Investigating a useful target (checking logs/metrics of relevant service)
    +0.15  Correctly identifying root cause via diagnose command
    +0.10  Correct causal chain (partial credit for partial accuracy)
    +0.20  Each correct fix applied
    +0.05  Speed bonus (solving in fewer steps)
    -0.02  Investigating irrelevant service
    -0.05  Wrong fix attempt (restart/rollback/scale wrong target)
    -0.15  Causing collateral damage (wrong fix order)
    """

    def __init__(self, config: ScenarioGradingConfig):
        self._config = config
        self._investigated_services: set = set()
        self._diagnosis_submitted: bool = False
        self._fixes_applied: List[str] = []
        self._collateral_count: int = 0
        self._cumulative_reward: float = 0.0
        self._step_rewards: List[float] = []
        self._status_check_count: int = 0  # track check_status calls

    def grade_step(
        self,
        command: str,
        target: str,
        params: Dict[str, Any],
        action_succeeded: bool,
        services_now_healthy: List[str],
        all_resolved: bool,
        step_number: int,
        collateral_damage: int,
    ) -> GradeResult:
        """
        Grade a single step and return the reward.

        Parameters
        ----------
        command            : The command the agent executed
        target             : Target service name
        params             : Additional parameters
        action_succeeded   : Whether the action actually fixed something
        services_now_healthy: List of currently healthy services
        all_resolved       : Whether all services are now healthy
        step_number        : Current step number
        collateral_damage  : Total collateral damage events so far

        Returns
        -------
        GradeResult with reward, breakdown, and feedback
        """
        reward = 0.0
        breakdown = {}
        feedback_parts = []

        # ─── Investigation rewards ───
        if command in ("check_logs", "check_metrics", "check_status"):
            if command == "check_status":
                # check_status is always slightly useful (free action) — reward first 2 calls
                self._status_check_count += 1
                if self._status_check_count <= 2:
                    reward += 0.02
                    breakdown["status_check"] = 0.02
                    feedback_parts.append("Good: Checking overall system status.")
            elif target in self._config.useful_investigation_targets:
                if target not in self._investigated_services:
                    reward += 0.05
                    breakdown["useful_investigation"] = 0.05
                    feedback_parts.append(f"Good: Investigating {target} is relevant.")
                    self._investigated_services.add(target)
            else:
                reward -= 0.02
                breakdown["irrelevant_investigation"] = -0.02
                feedback_parts.append(f"Wasted time: {target} is not directly relevant.")

        # ─── Diagnosis rewards ───
        elif command == "diagnose":
            diag_reward, diag_breakdown, diag_feedback = self._grade_diagnosis(params)
            reward += diag_reward
            breakdown.update(diag_breakdown)
            feedback_parts.append(diag_feedback)

        # ─── Fix action rewards ───
        elif command in ("restart_service", "rollback_deploy", "scale_service"):
            if action_succeeded:
                if target not in self._fixes_applied:
                    # First time this service is fixed — full reward
                    reward += 0.20
                    breakdown["correct_fix"] = 0.20
                    feedback_parts.append(f"Excellent: {command} on {target} fixed the service.")
                    self._fixes_applied.append(target)
                else:
                    # Already rewarded for this target — no double credit
                    feedback_parts.append(f"Note: {target} was already fixed.")
            else:
                # Distinguish: already-healthy (wasted step, 0) vs genuinely wrong fix (-0.05)
                if target in self._fixes_applied:
                    # Service already fixed — just a wasted step, not penalized
                    feedback_parts.append(f"Wasted step: {target} is already healthy.")
                else:
                    reward -= 0.05
                    breakdown["wrong_fix"] = -0.05
                    feedback_parts.append(f"Failed: {command} on {target} did not resolve the issue.")

        # ─── Collateral damage penalty ───
        new_damage = collateral_damage - self._collateral_count
        if new_damage > 0:
            penalty = new_damage * -0.15
            reward += penalty
            breakdown["collateral_damage"] = penalty
            feedback_parts.append(f"DAMAGE: {new_damage} additional service(s) affected by wrong action order.")
            self._collateral_count = collateral_damage

        # ─── All resolved bonus ───
        if all_resolved:
            # Speed bonus: fewer steps = more reward
            optimal = self._config.max_optimal_steps
            if step_number <= optimal:
                speed_bonus = 0.10
            elif step_number <= optimal * 1.5:
                speed_bonus = 0.05
            else:
                speed_bonus = 0.00
            reward += speed_bonus
            breakdown["speed_bonus"] = speed_bonus
            breakdown["resolution_bonus"] = 0.05
            reward += 0.05
            feedback_parts.append(f"🎉 All services resolved in {step_number} steps!")

        # Track
        self._cumulative_reward += reward
        self._step_rewards.append(reward)

        return GradeResult(
            reward=round(reward, 4),
            breakdown=breakdown,
            feedback=" | ".join(feedback_parts) if feedback_parts else "No notable effect.",
        )

    def _grade_diagnosis(self, params: Dict[str, Any]) -> tuple:
        """Grade a diagnosis submission with causal chain evaluation."""
        reward = 0.0
        breakdown = {}
        feedback_parts = []

        if self._diagnosis_submitted:
            return -0.02, {"duplicate_diagnosis": -0.02}, "Diagnosis already submitted."
        self._diagnosis_submitted = True

        # Root cause identification
        agent_root_cause = params.get("root_cause", "")
        if agent_root_cause == self._config.root_cause_service:
            reward += 0.15
            breakdown["root_cause_correct"] = 0.15
            feedback_parts.append("✅ Root cause correctly identified!")
        else:
            reward -= 0.03
            breakdown["root_cause_wrong"] = -0.03
            feedback_parts.append(
                f"❌ Wrong root cause: you said '{agent_root_cause}', "
                f"actual is '{self._config.root_cause_service}'."
            )

        # Causal chain evaluation
        agent_chain = params.get("causal_chain", [])
        if agent_chain and self._config.ground_truth_causal_chain:
            truth = self._config.ground_truth_causal_chain
            # Calculate overlap score
            correct_steps = sum(
                1 for step in agent_chain
                if any(truth_step.lower() in step.lower() or step.lower() in truth_step.lower()
                       for truth_step in truth)
            )
            chain_accuracy = correct_steps / max(len(truth), 1)

            chain_reward = round(0.10 * chain_accuracy, 4)
            reward += chain_reward
            breakdown["causal_chain_accuracy"] = chain_reward
            feedback_parts.append(
                f"Causal chain: {correct_steps}/{len(truth)} steps correct "
                f"({chain_accuracy:.0%} accuracy)"
            )

        # Confidence calibration bonus
        confidence = params.get("confidence", 0.5)
        actual_accuracy = 1.0 if agent_root_cause == self._config.root_cause_service else 0.0
        calibration_error = abs(confidence - actual_accuracy)
        if calibration_error < 0.2:
            reward += 0.03
            breakdown["confidence_calibrated"] = 0.03
            feedback_parts.append("Confidence well-calibrated.")

        return reward, breakdown, " | ".join(feedback_parts)

    def get_final_score(self) -> GradeResult:
        """
        Compute final episode score normalized to [0.0, 1.0].
        """
        raw = self._cumulative_reward
        # Normalize: max theoretical reward is ~1.0 depending on scenario
        score = max(0.0, min(1.0, raw / self._config.max_total_reward))

        breakdown = {
            "raw_cumulative": round(raw, 4),
            "normalized_score": round(score, 4),
            "steps_taken": len(self._step_rewards),
            "correct_fixes": len(self._fixes_applied),
            "diagnosis_submitted": self._diagnosis_submitted,
            "collateral_damage": self._collateral_count,
        }

        if score >= 0.8:
            feedback = "🏆 Excellent incident response!"
        elif score >= 0.5:
            feedback = "👍 Good response with room for improvement."
        elif score >= 0.2:
            feedback = "⚠️ Partial resolution — key issues remaining."
        else:
            feedback = "❌ Incident not resolved effectively."

        return GradeResult(
            reward=round(score, 4),
            breakdown=breakdown,
            feedback=feedback,
        )

    @property
    def cumulative_reward(self) -> float:
        return self._cumulative_reward

    @property
    def step_rewards(self) -> List[float]:
        return list(self._step_rewards)
