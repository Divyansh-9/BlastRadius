"""
Grading engine for the incident response environment.

Computes per-step rewards and final episode scores.
Includes causal chain evaluation — the key differentiator.

Reward ranges are clamped to [0.0, 1.0] for final scores.

v2.0 — TF-IDF cosine similarity for causal chains, configurable
reward magnitudes, smooth speed bonus, symmetric confidence
calibration.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────
# Lightweight TF-IDF Cosine Similarity (no external dependency)
# ─────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", text.lower())


def _tf(tokens: List[str]) -> Dict[str, float]:
    """Term frequency: count / total."""
    counts = Counter(tokens)
    total = len(tokens) or 1
    return {t: c / total for t, c in counts.items()}


def _idf(documents: List[List[str]]) -> Dict[str, float]:
    """Inverse document frequency across a corpus."""
    n = len(documents) or 1
    df: Dict[str, int] = {}
    for doc in documents:
        for token in set(doc):
            df[token] = df.get(token, 0) + 1
    return {t: math.log((n + 1) / (d + 1)) + 1 for t, d in df.items()}


def _tfidf_vector(tokens: List[str], idf_map: Dict[str, float]) -> Dict[str, float]:
    """Build a TF-IDF vector for a single document."""
    tf = _tf(tokens)
    return {t: tf_val * idf_map.get(t, 1.0) for t, tf_val in tf.items()}


def _cosine_similarity(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors."""
    common = set(v1) & set(v2)
    if not common:
        return 0.0
    dot = sum(v1[k] * v2[k] for k in common)
    mag1 = math.sqrt(sum(val ** 2 for val in v1.values()))
    mag2 = math.sqrt(sum(val ** 2 for val in v2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def compute_chain_similarity(
    agent_chain: List[str],
    truth_chain: List[str],
    similarity_threshold: float = 0.20,
) -> Tuple[float, int, int]:
    """
    Compare agent's causal chain against ground truth using TF-IDF
    cosine similarity.

    Returns (accuracy, matched_count, truth_count).

    Each agent step is matched to the best ground truth step.
    A match counts if cosine similarity >= threshold.
    Each truth step can only be matched once (greedy best-first).
    """
    if not agent_chain or not truth_chain:
        return 0.0, 0, max(len(truth_chain), 1)

    # Build corpus from both chains for IDF
    all_docs = [_tokenize(s) for s in agent_chain + truth_chain]
    idf_map = _idf(all_docs)

    agent_vectors = [_tfidf_vector(_tokenize(s), idf_map) for s in agent_chain]
    truth_vectors = [_tfidf_vector(_tokenize(s), idf_map) for s in truth_chain]

    # Compute similarity matrix
    similarities = []
    for ai, av in enumerate(agent_vectors):
        for ti, tv in enumerate(truth_vectors):
            sim = _cosine_similarity(av, tv)
            if sim >= similarity_threshold:
                similarities.append((sim, ai, ti))

    # Greedy matching: highest similarity first, no reuse
    similarities.sort(reverse=True)
    matched_agent = set()
    matched_truth = set()
    matched_count = 0

    for sim, ai, ti in similarities:
        if ai not in matched_agent and ti not in matched_truth:
            matched_agent.add(ai)
            matched_truth.add(ti)
            matched_count += 1

    accuracy = matched_count / len(truth_chain)
    return accuracy, matched_count, len(truth_chain)


# ─────────────────────────────────────────────────────────────
# Reward Configuration (eliminates all magic numbers)
# ─────────────────────────────────────────────────────────────

@dataclass
class RewardConfig:
    """
    All reward magnitudes in one place.
    No magic numbers anywhere else in this file.
    """
    # Investigation
    status_check_reward: float = 0.02
    max_status_checks_rewarded: int = 2
    useful_investigation: float = 0.05
    irrelevant_investigation: float = -0.02

    # Diagnosis
    root_cause_correct: float = 0.15
    root_cause_wrong: float = -0.03
    causal_chain_max: float = 0.10
    confidence_calibrated: float = 0.03
    confidence_miscalibrated: float = -0.03
    confidence_calibration_tolerance: float = 0.2
    duplicate_diagnosis: float = -0.02

    # Fixes
    correct_fix: float = 0.20
    wrong_fix: float = -0.05
    collateral_damage_per_event: float = -0.15

    # Episode completion
    resolution_bonus: float = 0.05
    speed_bonus_max: float = 0.10

    # Causal chain similarity
    chain_similarity_threshold: float = 0.20


# Default config instance
DEFAULT_REWARD_CONFIG = RewardConfig()


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

    v2.0 Changes:
    - TF-IDF cosine similarity for causal chain evaluation
    - All reward values from RewardConfig (no magic numbers)
    - Smooth linear speed bonus (not step function)
    - Symmetric confidence calibration (penalizes overconfident wrong)
    - Duplicate diagnosis returns 0 (not penalty for re-submitting correct)
    """

    def __init__(
        self,
        config: ScenarioGradingConfig,
        reward_config: Optional[RewardConfig] = None,
    ):
        self._config = config
        self._rc = reward_config or DEFAULT_REWARD_CONFIG
        self._investigated_services: set = set()
        self._diagnosis_submitted: bool = False
        self._diagnosis_was_correct: bool = False
        self._fixes_applied: List[str] = []
        self._collateral_count: int = 0
        self._cumulative_reward: float = 0.0
        self._step_rewards: List[float] = []
        self._status_check_count: int = 0
        self._fix_attempts: Dict[str, int] = {}  # anti-cheat: track per-service
        self._revision_used: bool = False  # Bug #3: explicitly init for snapshot safety

    # ── Snapshot Support (Bug #4: GRPO grader state cloning) ──

    def save_snapshot(self) -> Dict:
        """Serialize all mutable grader state for GRPO environment cloning."""
        return {
            "investigated": list(self._investigated_services),
            "diagnosis_submitted": self._diagnosis_submitted,
            "diagnosis_correct": self._diagnosis_was_correct,
            "revision_used": self._revision_used,
            "fixes_applied": list(self._fixes_applied),
            "collateral_count": self._collateral_count,
            "cumulative_reward": self._cumulative_reward,
            "step_rewards": list(self._step_rewards),
            "status_check_count": self._status_check_count,
            "fix_attempts": dict(self._fix_attempts),
        }

    def restore_snapshot(self, snap: Dict):
        """Restore grader state from a snapshot dict."""
        self._investigated_services = set(snap.get("investigated", []))
        self._diagnosis_submitted = snap.get("diagnosis_submitted", False)
        self._diagnosis_was_correct = snap.get("diagnosis_correct", False)
        self._revision_used = snap.get("revision_used", False)
        self._fixes_applied = list(snap.get("fixes_applied", []))
        self._collateral_count = snap.get("collateral_count", 0)
        self._cumulative_reward = snap.get("cumulative_reward", 0.0)
        self._step_rewards = list(snap.get("step_rewards", []))
        self._status_check_count = snap.get("status_check_count", 0)
        self._fix_attempts = dict(snap.get("fix_attempts", {}))

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
        rc = self._rc

        # ─── Investigation rewards ───
        if command in ("check_logs", "check_metrics", "check_status", "check_dependencies"):
            if command == "check_status":
                self._status_check_count += 1
                if self._status_check_count <= rc.max_status_checks_rewarded:
                    reward += rc.status_check_reward
                    breakdown["status_check"] = rc.status_check_reward
                    feedback_parts.append("Good: Checking overall system status.")
            elif command == "check_dependencies":
                # Reward once for checking dependency graph
                if "_deps_checked" not in self._investigated_services:
                    reward += rc.status_check_reward
                    breakdown["dependency_check"] = rc.status_check_reward
                    feedback_parts.append("Good: Understanding service dependencies.")
                    self._investigated_services.add("_deps_checked")
            elif target in self._config.useful_investigation_targets:
                if target not in self._investigated_services:
                    reward += rc.useful_investigation
                    breakdown["useful_investigation"] = rc.useful_investigation
                    feedback_parts.append(f"Good: Investigating {target} is relevant.")
                    self._investigated_services.add(target)
                else:
                    # Re-investigation: same penalty as irrelevant to discourage step waste
                    reward += rc.irrelevant_investigation
                    breakdown["irrelevant_investigation"] = rc.irrelevant_investigation
                    feedback_parts.append(f"Already investigated {target}. Wasted step.")
            elif target:
                reward += rc.irrelevant_investigation
                breakdown["irrelevant_investigation"] = rc.irrelevant_investigation
                feedback_parts.append(f"Wasted time: {target} is not directly relevant.")

        # ─── Diagnosis rewards ───
        elif command == "diagnose":
            diag_reward, diag_breakdown, diag_feedback = self._grade_diagnosis(params)
            reward += diag_reward
            breakdown.update(diag_breakdown)
            feedback_parts.append(diag_feedback)

        # ─── Fix action rewards ───
        elif command in ("restart_service", "rollback_deploy", "scale_service"):
            # Track fix attempts per service (anti-cheat)
            self._fix_attempts[target] = self._fix_attempts.get(target, 0) + 1

            if action_succeeded:
                if target not in self._fixes_applied:
                    reward += rc.correct_fix
                    breakdown["correct_fix"] = rc.correct_fix
                    feedback_parts.append(f"Excellent: {command} on {target} fixed the service.")
                    self._fixes_applied.append(target)
                else:
                    feedback_parts.append(f"Note: {target} was already fixed.")
            else:
                if target in self._fixes_applied:
                    feedback_parts.append(f"Wasted step: {target} is already healthy.")
                else:
                    reward += rc.wrong_fix
                    breakdown["wrong_fix"] = rc.wrong_fix
                    feedback_parts.append(f"Failed: {command} on {target} did not resolve the issue.")

            # Anti-cheat: penalize excessive fix attempts on same service
            attempts = self._fix_attempts[target]
            if attempts > 2:
                spam_penalty = -0.01 * (attempts - 2)
                reward += spam_penalty
                breakdown["fix_spam_penalty"] = spam_penalty
                feedback_parts.append(f"Warning: Repeated fix attempts on {target} (attempt #{attempts}).")

        # ─── Collateral damage penalty ───
        new_damage = collateral_damage - self._collateral_count
        if new_damage > 0:
            penalty = new_damage * rc.collateral_damage_per_event
            reward += penalty
            breakdown["collateral_damage"] = penalty
            feedback_parts.append(f"DAMAGE: {new_damage} additional service(s) affected by wrong action order.")
            self._collateral_count = collateral_damage

        # ─── All resolved bonus ───
        if all_resolved:
            # Smooth linear speed bonus
            optimal = self._config.max_optimal_steps
            if step_number <= optimal:
                speed_bonus = rc.speed_bonus_max
            elif step_number >= optimal * 2:
                speed_bonus = 0.0
            else:
                # Linear interpolation: bonus decreases from max to 0
                progress = (step_number - optimal) / optimal
                speed_bonus = round(rc.speed_bonus_max * (1.0 - progress), 4)

            reward += speed_bonus
            breakdown["speed_bonus"] = speed_bonus
            breakdown["resolution_bonus"] = rc.resolution_bonus
            reward += rc.resolution_bonus
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
        rc = self._rc

        if self._diagnosis_submitted:
            # Don't penalize re-submission of a CORRECT diagnosis
            if self._diagnosis_was_correct:
                return 0.0, {}, "Diagnosis already submitted (correct). No change."
            # Bug #2 fix: Allow one revision attempt at 50% reward weight
            if not self._revision_used:
                self._revision_used = True
                self._diagnosis_submitted = False  # Reset to allow re-grade
                # Bug H: Guard against exceptions leaving _diagnosis_submitted=False
                try:
                    r, b, f = self._grade_diagnosis_inner(params)
                except Exception:
                    self._diagnosis_submitted = True  # restore on failure
                    raise
                return round(r * 0.5, 4), {k: round(v * 0.5, 4) for k, v in b.items()}, f"[REVISED x0.5] {f}"
            return rc.duplicate_diagnosis, {"duplicate_diagnosis": rc.duplicate_diagnosis}, "No more revisions allowed."
        return self._grade_diagnosis_inner(params)

    def _grade_diagnosis_inner(self, params: Dict[str, Any]) -> tuple:
        """Core diagnosis grading logic. Separated for revision support."""
        reward = 0.0
        breakdown = {}
        feedback_parts = []
        rc = self._rc

        self._diagnosis_submitted = True

        # Root cause identification
        agent_root_cause = params.get("root_cause", "")
        if agent_root_cause == self._config.root_cause_service:
            reward += rc.root_cause_correct
            breakdown["root_cause_correct"] = rc.root_cause_correct
            feedback_parts.append("✅ Root cause correctly identified!")
            self._diagnosis_was_correct = True
        else:
            reward += rc.root_cause_wrong
            breakdown["root_cause_wrong"] = rc.root_cause_wrong
            feedback_parts.append(
                f"❌ Wrong root cause: you said '{agent_root_cause}', "
                f"actual is '{self._config.root_cause_service}'."
            )

        # Causal chain evaluation (TF-IDF cosine similarity)
        agent_chain = params.get("causal_chain", [])
        if agent_chain and self._config.ground_truth_causal_chain:
            truth = self._config.ground_truth_causal_chain

            chain_accuracy, matched, total = compute_chain_similarity(
                agent_chain, truth, rc.chain_similarity_threshold
            )

            chain_reward = round(rc.causal_chain_max * chain_accuracy, 4)
            reward += chain_reward
            breakdown["causal_chain_accuracy"] = chain_reward
            feedback_parts.append(
                f"Causal chain: {matched}/{total} steps matched "
                f"({chain_accuracy:.0%} semantic accuracy)"
            )

        # Symmetric confidence calibration
        # Bug N: Clamp confidence to [0, 1] — reject nonsensical values
        confidence = max(0.0, min(1.0, float(params.get("confidence", 0.5))))
        actual_accuracy = 1.0 if agent_root_cause == self._config.root_cause_service else 0.0
        calibration_error = abs(confidence - actual_accuracy)
        if calibration_error < rc.confidence_calibration_tolerance:
            reward += rc.confidence_calibrated
            breakdown["confidence_calibrated"] = rc.confidence_calibrated
            feedback_parts.append("Confidence well-calibrated.")
        elif confidence > 0.7 and actual_accuracy == 0.0:
            # Penalize overconfident wrong answers (symmetric calibration)
            reward += rc.confidence_miscalibrated
            breakdown["confidence_miscalibrated"] = rc.confidence_miscalibrated
            feedback_parts.append("⚠️ Overconfident wrong diagnosis penalized.")

        return reward, breakdown, " | ".join(feedback_parts)

    def get_final_score(self) -> GradeResult:
        """
        Compute final episode score normalized to [0.0, 1.0].
        """
        raw = self._cumulative_reward
        # Normalize: max theoretical reward is scenario-specific
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
