"""
Hard Scenario: DB Replica Failover Split-Brain

Situation:
- Primary DB failed over to replica automatically, but the replica wasn't fully synced.
- The old Primary comes back online and there's a split brain scenario. Applications see stale data.
- Root cause: replication-mgr (split-brain).
- Fix: stop/rollback db-primary (the dead one) -> apply authoritative promote to db-replica -> restart app-server.

Temporal evolution:
- If unfixed after 4 min: queue-worker reads stale data.
"""

from incident_env.server.engine.infrastructure import (
    CascadeRule,
    ServiceGraph,
    ServiceNode,
    ServiceStatus,
)
from incident_env.server.engine.grader import ScenarioGradingConfig
from incident_env.server.scenarios.base import BaseScenario


class DbFailoverScenario(BaseScenario):

    @property
    def scenario_id(self) -> str:
        return "hard_db_failover"

    @property
    def difficulty(self) -> str:
        return "hard"

    @property
    def title(self) -> str:
        return "Database Split-Brain Failover"

    @property
    def description(self) -> str:
        return (
            "Consistency errors are triggering data corruption alerts. "
            "Users report they save data but it disappears on refresh. "
            "The infrastructure monitoring shows recent failover events."
        )

    def build_service_graph(self) -> ServiceGraph:
        services = [
            ServiceNode(
                name="replication-mgr",
                display_name="DB Replication Manager",
                status=ServiceStatus.DEGRADED,
                dependencies=["db-primary", "db-replica"],
                port=2379,
                healthy_metrics={
                    "latency_p50_ms": 2.0,
                },
                current_metrics={
                    "latency_p50_ms": 150.0,
                },
                log_pattern="degraded",
                failure_description="SPLIT BRAIN DETECTED: Multiple masters accepting writes.",
                is_root_cause=True,
                fixable_by=["restart"], # Represents forcing a topology recalculation
                fix_order=2,
            ),
            ServiceNode(
                name="db-primary",
                display_name="Database Node (Old Primary)",
                status=ServiceStatus.DEGRADED,
                dependencies=[],
                port=5432,
                healthy_metrics={
                    "error_rate_percent": 0.0,
                },
                current_metrics={
                    "error_rate_percent": 50.0,
                },
                log_pattern="degraded",
                failure_description="Stale timeline. Network partition recovered but state out of sync.",
                is_root_cause=False,
                fixable_by=["rollback"], # Represents taking it offline safely
                fix_order=1,
            ),
            ServiceNode(
                name="db-replica",
                display_name="Database Node (New Promoted Primary)",
                status=ServiceStatus.HEALTHY,
                dependencies=[],
                port=5433,
            ),
            ServiceNode(
                name="app-server",
                display_name="Application Server",
                status=ServiceStatus.DEGRADED,
                dependencies=["replication-mgr"],
                port=3000,
                healthy_metrics={
                    "error_rate_percent": 0.1,
                },
                current_metrics={
                    "error_rate_percent": 25.0,
                },
                log_pattern="degraded",
                failure_description="ConstraintViolation: duplicate key value / row not found.",
                is_root_cause=False,
                fixable_by=["restart"], # To force new connection pool
                fix_order=3,
            ),
            ServiceNode(
                name="queue-worker",
                display_name="Asynchronous Job Worker",
                status=ServiceStatus.HEALTHY,
                dependencies=["app-server"],
                port=3001,
            ),
        ]

        cascade_rules = [
            CascadeRule(
                source="replication-mgr",
                target="queue-worker",
                delay_minutes=4,
                target_status=ServiceStatus.DEGRADED,
            ),
        ]

        return ServiceGraph(services, cascade_rules)

    def get_grading_config(self) -> ScenarioGradingConfig:
        return ScenarioGradingConfig(
            root_cause_service="replication-mgr",
            root_cause_description="Split-brain database topology with multiple masters",
            ground_truth_causal_chain=[
                "old primary partitioned and replica promoted",
                "old primary rejoined network causing split brain",
                "app-server writes randomly to both nodes causing consistency errors",
            ],
            correct_fix_actions=[
                {"command": "rollback_deploy", "target": "db-primary"}, # Step down old master
                {"command": "restart_service", "target": "replication-mgr"}, # Fix topology
                {"command": "restart_service", "target": "app-server"}, # Flush bad connection pool
            ],
            correct_fix_order=["db-primary", "replication-mgr", "app-server"],
            useful_investigation_targets=["replication-mgr", "db-primary", "app-server"],
            max_optimal_steps=8,
            max_total_reward=0.85,
        )
