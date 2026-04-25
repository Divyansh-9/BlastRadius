"""
Typed models for the IT Incident Response Environment.

Defines the Action, Observation, and State dataclasses that form
the contract between the agent and the environment.

Enhanced with:
- Temporal evolution tracking
- Causal chain diagnosis support
- Information cost model metadata
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Action — what the agent can do
# ---------------------------------------------------------------------------

@dataclass
class IncidentAction:
    """
    An action the agent can take during incident response.

    Commands & Time Costs
    ---------------------
    check_status       (0 min) : View health status of all services
    check_logs         (2 min) : View recent log entries for a target service
    check_metrics      (1 min) : View CPU/mem/latency/errors for a target service
    check_dependencies (1 min) : View the service dependency graph
    diagnose           (0 min) : Declare root cause + causal chain hypothesis
    restart_service    (3 min) : Restart a specific service (risky)
    rollback_deploy    (5 min) : Roll back last deployment on a service (slow but safe)
    scale_service      (2 min) : Scale resources for a service
    """

    command: str
    target: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)


# Time cost for each command (in simulated minutes)
ACTION_TIME_COSTS: Dict[str, int] = {
    "check_status": 0,
    "check_logs": 2,
    "check_metrics": 1,
    "check_dependencies": 1,
    "diagnose": 0,
    "restart_service": 3,
    "rollback_deploy": 5,
    "scale_service": 2,
}

VALID_COMMANDS = set(ACTION_TIME_COSTS.keys())


# ---------------------------------------------------------------------------
# Observation — what the agent sees
# ---------------------------------------------------------------------------

@dataclass
class IncidentObservation:
    """
    The observation returned after every action.

    Fields
    ------
    output                : Human-readable text output of the command
    services_status       : {service_name: "healthy"|"degraded"|"down"}
    active_alerts         : Currently firing alert descriptions
    time_elapsed_minutes  : Simulated minutes since incident start
    incident_severity     : P1/P2/P3 severity level
    services_at_risk      : Services trending toward failure
    hint                  : Optional guiding context
    """

    output: str = ""
    services_status: Dict[str, str] = field(default_factory=dict)
    active_alerts: List[str] = field(default_factory=list)
    time_elapsed_minutes: int = 0
    incident_severity: str = ""
    services_at_risk: List[str] = field(default_factory=list)
    hint: str = ""


# ---------------------------------------------------------------------------
# State — full episode state (superset of observation)
# ---------------------------------------------------------------------------

@dataclass
class IncidentState:
    """
    Complete internal state of an incident episode.

    Tracks all metadata needed for grading, replay, and debugging.
    Includes temporal evolution tracking and causal chain data.
    """

    episode_id: str = ""
    step_count: int = 0
    scenario_id: str = ""
    task_difficulty: str = ""           # easy | medium | hard

    # Resolution tracking
    services_resolved: List[str] = field(default_factory=list)
    root_cause_identified: bool = False
    root_cause_service: str = ""
    is_resolved: bool = False

    # Reward tracking
    total_reward: float = 0.0
    step_rewards: List[float] = field(default_factory=list)

    # Action history
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)

    # Temporal state
    time_elapsed_minutes: int = 0
    collateral_damage: int = 0          # Services broken by wrong actions

    # Causal reasoning
    agent_diagnosis: Optional[Dict[str, Any]] = None
    diagnosis_accuracy: float = 0.0
    wrong_diagnoses: int = 0

    # Episode bounds
    max_steps: int = 25
    done: bool = False
