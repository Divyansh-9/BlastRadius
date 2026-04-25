"""
Core Incident Response Environment.

Implements the OpenEnv interface: reset(), step(), state.
Orchestrates the service graph, temporal evolution, log/metrics
generation, and grading.
"""

from __future__ import annotations

import copy
import random
import uuid
import hashlib
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from incident_env.models import (
    ACTION_TIME_COSTS,
    VALID_COMMANDS,
    IncidentAction,
    IncidentObservation,
    IncidentState,
)
from incident_env.server.engine.grader import Grader
from incident_env.server.engine.infrastructure import ServiceGraph
from incident_env.server.engine.log_generator import generate_logs
from incident_env.server.engine.metrics_generator import generate_metrics_report
from incident_env.server.scenarios import SCENARIOS
from incident_env.server.scenarios.base import BaseScenario


class IncidentEnvironment:
    """
    IT Incident Response Environment.

    The agent is dropped into a production incident and must:
    1. Investigate (check logs, metrics, status, dependencies)
    2. Diagnose (submit root cause + causal chain hypothesis)
    3. Remediate (restart, rollback, scale — in correct order)

    Time ticks forward with each action, and failures cascade.
    """

    def __init__(self):
        self._state: IncidentState = IncidentState()
        self._graph: Optional[ServiceGraph] = None
        self._scenario: Optional[BaseScenario] = None
        self._grader: Optional[Grader] = None
        self._eval_mode: bool = False
        self._obf_map: Dict[str, str] = {}
        self._action_history: List[tuple] = []  # (command, target) pairs for repetition detection
        self._diagnosis_attempts: int = 0  # escalating penalty counter

    def _obfuscate(self, data: Any) -> Any:
        if not self._eval_mode or not self._obf_map:
            return data
            
        if isinstance(data, str):
            text = data
            for real, obf in self._obf_map.items():
                text = text.replace(real, obf)
            return text
            
        if isinstance(data, dict):
            return {
                self._obf_map.get(k, k): self._obfuscate(v)  # Bug #7: recurse into values too
                for k, v in data.items()
            }
            
        if isinstance(data, list):
            return [self._obfuscate(item) for item in data]  # recurse into items as strings
            
        return data

    def _deobfuscate(self, target: str) -> str:
        if not self._eval_mode:
            return target
        for real, obf in self._obf_map.items():
            if target == obf:
                return real
        return target

    # -----------------------------------------------------------------
    # Snapshot Support (Fix #2: GRPO environment cloning)
    # -----------------------------------------------------------------

    def save_snapshot(self) -> Dict[str, Any]:
        """
        Capture the full mutable state of the environment.
        Used by GRPO to freeze state at step N, then restore it
        independently for each of G=4 candidate completions.
        """
        # Use task_difficulty (e.g. "easy") which maps to SCENARIOS keys,
        # NOT scenario_id (e.g. "easy_db_pool") which is internal.
        return {
            "task_id": self._state.task_difficulty if self._state else "easy",
            "state": copy.deepcopy(asdict(self._state)),
            "graph_snapshot": self._graph.save_snapshot() if self._graph else {},
            "grader_snapshot": self._grader.save_snapshot() if self._grader else {},
            "diagnosis_attempts": self._diagnosis_attempts,
            "action_history": list(self._action_history),
        }

    def restore_snapshot(self, snapshot: Dict[str, Any]):
        """
        Restore environment to a previously saved snapshot.
        The scenario/graph structure must already be initialized via reset().
        """
        # Restore scenario first
        task_id = snapshot.get("task_id", "easy")
        scenario_cls = SCENARIOS.get(task_id)
        if scenario_cls is None:
            raise ValueError(f"Cannot restore: unknown task_id '{task_id}'")

        self._scenario = scenario_cls()
        self._graph = self._scenario.build_service_graph()
        self._eval_mode = False
        self._obf_map = {}

        # Restore graph mutable state
        if self._graph and snapshot.get("graph_snapshot"):
            self._graph.restore_snapshot(snapshot["graph_snapshot"])

        # Restore grader
        grading_config = self._scenario.get_grading_config()
        self._grader = Grader(grading_config)
        # Bug #4: Restore grader internal state (investigation, diagnosis, rewards, etc.)
        if snapshot.get("grader_snapshot"):
            self._grader.restore_snapshot(snapshot["grader_snapshot"])

        # Restore episode state
        saved_state = snapshot.get("state", {})
        self._state = IncidentState(
            episode_id=saved_state.get("episode_id", str(uuid.uuid4())),
            step_count=saved_state.get("step_count", 0),
            scenario_id=saved_state.get("scenario_id", task_id),
            task_difficulty=saved_state.get("task_difficulty", "easy"),
            max_steps=saved_state.get("max_steps", 25),
            total_reward=saved_state.get("total_reward", 0.0),
            done=saved_state.get("done", False),
            is_resolved=saved_state.get("is_resolved", False),
            wrong_diagnoses=saved_state.get("wrong_diagnoses", 0),  # Bug #5: restore 3-strike counter
            root_cause_identified=saved_state.get("root_cause_identified", False),
        )

        self._diagnosis_attempts = snapshot.get("diagnosis_attempts", 0)
        self._action_history = list(snapshot.get("action_history", []))

    # -----------------------------------------------------------------
    # OpenEnv API: reset()
    # -----------------------------------------------------------------

    def reset(self, task_id: str = "easy", eval_mode: bool = False) -> Dict[str, Any]:
        """
        Initialize a new incident episode.

        Parameters
        ----------
        task_id : "easy" | "medium" | "hard"

        Returns
        -------
        Dict with observation, reward, done, info
        """
        # Build scenario
        scenario_cls = SCENARIOS.get(task_id)
        if scenario_cls is None:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose from: {list(SCENARIOS.keys())}")

        self._scenario = scenario_cls()
        self._graph = self._scenario.build_service_graph()
        self._eval_mode = eval_mode
        self._obf_map = {}
        
        self._action_history = []
        self._diagnosis_attempts = 0

        if self._eval_mode:
            for node_name in self._graph.service_names():
                slug = hashlib.md5((node_name + str(uuid.uuid4())).encode()).hexdigest()[:6]
                self._obf_map[node_name] = f"srv-{slug}"
            # Metric noise: jitter all current metrics by ±10% to prevent pattern recognition
            for svc in self._graph.get_all_services().values():
                for key in list(svc.current_metrics.keys()):
                    original = svc.current_metrics[key]
                    if isinstance(original, (int, float)) and original != 0:
                        jitter = random.uniform(0.9, 1.1)
                        svc.current_metrics[key] = round(original * jitter, 2)
                
        grading_config = self._scenario.get_grading_config()
        self._grader = Grader(grading_config)

        # Initialize state
        self._state = IncidentState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            scenario_id=self._scenario.scenario_id,
            task_difficulty=self._scenario.difficulty,
            max_steps=25,
        )

        # Build initial observation
        obs = IncidentObservation(
            output=self._obfuscate(self._scenario.get_initial_alert_message()),
            services_status=self._obfuscate(self._graph.get_status_summary()),
            active_alerts=self._obfuscate(self._graph.get_active_alerts()),
            time_elapsed_minutes=0,
            incident_severity=self._graph.get_incident_severity(),
            services_at_risk=self._obfuscate(self._graph.get_services_at_risk()),
            hint="" if self._eval_mode else self._obfuscate("Start by checking the status of all services."),
        )

        return {
            "observation": asdict(obs),
            "reward": 0.0,
            "done": False,
            "info": {"task_id": task_id, "episode_id": self._state.episode_id},
        }

    # -----------------------------------------------------------------
    # OpenEnv API: step()
    # -----------------------------------------------------------------

    def step(self, action: IncidentAction) -> Dict[str, Any]:
        """
        Execute an action and return the next observation + reward.

        Parameters
        ----------
        action : IncidentAction with command, target, parameters

        Returns
        -------
        Dict with observation, reward, done, info
        """
        if self._graph is None or self._grader is None or self._scenario is None:
            return self._error_response("Environment not initialized. Call reset() first.")

        if self._state.done:
            return self._error_response("Episode is already complete. Call reset() to start a new one.")

        # Fix #5: Handle _parse_failure sentinel from parse_action_json
        command = action.command.lower().strip()
        if command == "_parse_failure":
            self._state.step_count += 1
            obs = IncidentObservation(
                output="ERROR: Agent produced unparseable output. No action taken.",
                services_status=self._obfuscate(self._graph.get_status_summary()),
                active_alerts=self._obfuscate(self._graph.get_active_alerts()),
                time_elapsed_minutes=self._graph.time_minutes,
                incident_severity=self._graph.get_incident_severity(),
            )
            return {
                "observation": asdict(obs),
                "reward": -0.05,
                "done": False,
                "info": {"error": "parse_failure", "step_reward": -0.05},
            }

        # Validate command
        if command not in VALID_COMMANDS:
            return self._error_response(
                f"Unknown command '{command}'. Valid commands: {', '.join(sorted(VALID_COMMANDS))}"
            )

        # Advance time based on action cost
        time_cost = ACTION_TIME_COSTS.get(command, 1)
        if time_cost > 0:
            cascades = self._graph.tick(time_cost)
        else:
            cascades = []

        self._state.step_count += 1
        self._state.time_elapsed_minutes = self._graph.time_minutes

        # Execute the command
        output, action_succeeded = self._execute_command(command, self._deobfuscate(action.target), action.parameters)

        # Add cascade notifications to output
        if cascades:
            cascade_text = "\n\n📡 CASCADE ALERT:\n" + "\n".join(
                f"  ⚠️ {c['target']} → {c['new_status']} (from {c['source']})"
                for c in cascades
            )
            output += cascade_text
            
        output = self._obfuscate(output)

        # Track action
        self._state.actions_taken.append({
            "step": self._state.step_count,
            "command": command,
            "target": action.target,
            "time_cost": time_cost,
            "succeeded": action_succeeded,
        })

        # Check if resolved
        all_resolved = self._graph.is_fully_resolved()
        self._state.services_resolved = self._graph.get_resolved_services()
        self._state.collateral_damage = self._graph.count_collateral_damage()

        # Grade this step
        grade = self._grader.grade_step(
            command=command,
            target=action.target,
            params=action.parameters,
            action_succeeded=action_succeeded,
            services_now_healthy=self._state.services_resolved,
            all_resolved=all_resolved,
            step_number=self._state.step_count,
            collateral_damage=self._state.collateral_damage,
        )

        # Sync grader's cumulative reward FIRST (Bug #3 fix)
        self._state.total_reward = self._grader.cumulative_reward
        self._state.step_rewards = self._grader.step_rewards
        
        # Set root_cause_identified when grader confirms correct diagnosis (Bug #7 fix)
        if "root_cause_correct" in grade.breakdown:
            self._state.root_cause_identified = True

        # Anti-cheat: diagnosis penalty escalation
        # These mutations happen AFTER the grader sync, so they stick (Bug #3 fix)
        if command == "diagnose":
            self._diagnosis_attempts += 1
            if "root_cause_wrong" in grade.breakdown:
                self._state.wrong_diagnoses += 1
                if self._state.wrong_diagnoses > 1:
                    escalation = -0.03 * (2 ** (self._state.wrong_diagnoses - 2))
                    self._state.total_reward += escalation
                if self._state.wrong_diagnoses >= 3:
                    self._state.done = True
                    self._state.total_reward -= 0.5
                    grade.feedback = "Episode Terminated: Maximum incorrect diagnoses reached (Anti-Cheat)."

        # Anti-cheat: action repetition damping
        action_key = (command, self._deobfuscate(action.target) if action.target else "")
        repeat_count = sum(1 for prev in self._action_history if prev == action_key)
        if repeat_count >= 3 and command not in ("check_status", "diagnose"):
            damping = -0.01 * (repeat_count - 2)
            self._state.total_reward += damping
        self._action_history.append(action_key)

        # Fix #8: Check if done — distinguish timeout from resolution
        truncated = self._state.step_count >= self._state.max_steps and not all_resolved
        done = all_resolved or self._state.step_count >= self._state.max_steps or self._state.done
        self._state.done = done
        self._state.is_resolved = all_resolved

        # Build observation
        obs = IncidentObservation(
            output=output,
            services_status=self._obfuscate(self._graph.get_status_summary()),
            active_alerts=self._obfuscate(self._graph.get_active_alerts()),
            time_elapsed_minutes=self._graph.time_minutes,
            incident_severity=self._graph.get_incident_severity(),
            services_at_risk=self._obfuscate(self._graph.get_services_at_risk()),
            hint="" if self._eval_mode else self._obfuscate(grade.feedback),
        )

        # If done, append final score info
        info: Dict[str, Any] = {
            "step_reward": grade.reward,
            "reward_breakdown": grade.breakdown,
            "is_resolved": all_resolved,
            "truncated": truncated,
        }
        if done:
            final = self._grader.get_final_score()
            info["final_score"] = final.reward
            info["final_breakdown"] = final.breakdown
            info["final_feedback"] = final.feedback

        return {
            "observation": asdict(obs),
            "reward": grade.reward,
            "done": done,
            "info": info,
        }

    # -----------------------------------------------------------------
    # OpenEnv API: state
    # -----------------------------------------------------------------

    @property
    def state(self) -> Dict[str, Any]:
        """Return current episode state."""
        return asdict(self._state)

    # -----------------------------------------------------------------
    # Command execution
    # -----------------------------------------------------------------

    def _execute_command(
        self, command: str, target: str, params: Dict
    ) -> tuple:
        """
        Execute an agent command against the infrastructure.
        Returns (output_text, success_bool).
        """
        if command == "check_status":
            return self._cmd_check_status(), False

        if command == "check_logs":
            return self._cmd_check_logs(target), False

        if command == "check_metrics":
            return self._cmd_check_metrics(target), False

        if command == "check_dependencies":
            return self._cmd_check_dependencies(), False

        if command == "diagnose":
            return self._cmd_diagnose(params), False

        if command == "restart_service":
            text, success = self._graph.restart_service(target)
            return text, success

        if command == "rollback_deploy":
            text, success = self._graph.rollback_deploy(target)
            return text, success

        if command == "scale_service":
            text, success = self._graph.scale_service(target, params)
            return text, success

        return f"Unknown command: {command}", False

    def _cmd_check_status(self) -> str:
        """Show status of all services."""
        lines = ["=== System Status Dashboard ===", ""]
        for name, svc in self._graph.get_all_services().items():
            icon = {"healthy": "🟢", "degraded": "🟡", "down": "🔴", "restarting": "🔄"}.get(
                svc.status.value, "⚪"
            )
            lines.append(f"  {icon} {svc.display_name:<25} [{svc.status.value.upper()}]")
            if svc.status.value != "healthy" and svc.failure_description:
                lines.append(f"     └─ {svc.failure_description}")
        lines.append("")
        lines.append(f"Time elapsed: {self._graph.time_minutes} minutes since incident start")
        lines.append(f"Severity: {self._graph.get_incident_severity()}")

        at_risk = self._graph.get_services_at_risk()
        if at_risk:
            lines.append(f"\n⚠️ Services at risk of cascading failure: {', '.join(at_risk)}")

        return "\n".join(lines)

    def _cmd_check_logs(self, target: str) -> str:
        """Show logs for a specific service."""
        svc = self._graph.get_service(target)
        if svc is None:
            return (
                f"ERROR: Unknown service '{target}'.\n"
                f"Available services: {', '.join(self._graph.service_names())}"
            )
        return generate_logs(svc, self._graph.time_minutes)

    def _cmd_check_metrics(self, target: str) -> str:
        """Show metrics dashboard for a specific service."""
        svc = self._graph.get_service(target)
        if svc is None:
            return (
                f"ERROR: Unknown service '{target}'.\n"
                f"Available services: {', '.join(self._graph.service_names())}"
            )
        return generate_metrics_report(svc, self._graph.time_minutes)

    def _cmd_check_dependencies(self) -> str:
        """Show the service dependency graph."""
        return self._graph.get_dependency_text()

    def _cmd_diagnose(self, params: Dict) -> str:
        """Agent submits a diagnosis with root cause + causal chain."""
        root_cause = params.get("root_cause", "")
        causal_chain = params.get("causal_chain", [])
        confidence = params.get("confidence", 0.5)

        if not root_cause:
            return (
                "DIAGNOSIS INCOMPLETE: You must provide 'root_cause' in parameters.\n"
                "Example: {\"root_cause\": \"database\", "
                "\"causal_chain\": [\"db pool exhausted\", \"api timeouts\"], "
                "\"confidence\": 0.8}"
            )

        self._state.agent_diagnosis = {
            "root_cause": root_cause,
            "causal_chain": causal_chain,
            "confidence": confidence,
        }
        self._state.root_cause_service = root_cause

        return (
            f"📋 Diagnosis recorded:\n"
            f"  Root cause: {root_cause}\n"
            f"  Causal chain: {' → '.join(causal_chain) if causal_chain else 'not provided'}\n"
            f"  Confidence: {confidence:.0%}\n"
            f"\nProceeding with remediation based on this diagnosis."
        )

    def _error_response(self, message: str) -> Dict[str, Any]:
        """Return an error response."""
        severity = self._graph.get_incident_severity() if self._graph else ""
        obs = IncidentObservation(
            output=f"ERROR: {message}",
            incident_severity=severity,
        )
        return {
            "observation": asdict(obs),
            "reward": 0.0,
            "done": self._state.done,
            "info": {"error": message},
        }
