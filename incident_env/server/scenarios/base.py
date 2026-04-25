"""
Base scenario class.

Each scenario defines:
- Initial service configuration (what's broken and how)
- Cascade rules (how failures spread over time)
- Grading config (ground truth for evaluation)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from incident_env.server.engine.infrastructure import ServiceGraph
from incident_env.server.engine.grader import ScenarioGradingConfig


class BaseScenario(ABC):
    """Abstract base for all incident scenarios."""

    @property
    @abstractmethod
    def scenario_id(self) -> str:
        """Unique scenario identifier."""
        ...

    @property
    @abstractmethod
    def difficulty(self) -> str:
        """easy | medium | hard"""
        ...

    @property
    @abstractmethod
    def title(self) -> str:
        """Human-readable scenario title."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Brief description shown to the agent."""
        ...

    @abstractmethod
    def build_service_graph(self) -> ServiceGraph:
        """Construct the initial service graph with failure states."""
        ...

    @abstractmethod
    def get_grading_config(self) -> ScenarioGradingConfig:
        """Return the grading configuration with ground truth."""
        ...

    def get_initial_alert_message(self) -> str:
        """The alert message the agent sees when the incident starts."""
        return (
            f"🚨 INCIDENT ALERT — {self.title}\n"
            f"Severity: {'P1' if self.difficulty == 'hard' else 'P2'}\n"
            f"Description: {self.description}\n"
            f"\nYou are the on-call SRE. Diagnose the issue and restore all services.\n"
            f"Available commands: check_status, check_logs, check_metrics, "
            f"check_dependencies, diagnose, restart_service, rollback_deploy, scale_service\n"
            f"\n⏱️  Time is ticking — failures may spread while you investigate."
        )
