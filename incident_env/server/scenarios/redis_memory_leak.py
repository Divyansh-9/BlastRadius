"""
Easy Scenario: Redis Memory Leak & OOM

Situation:
- Defective deployment causes session cache without TTLs.
- Redis server consumes all RAM and is repeatedly OOM killed by kernel.
- The session-store depends on it and fails.
- Fix: Restart redis to clear memory, rollback session-store bad deploy.

Temporal evolution:
- If unfixed after 3 min: session-store fails and web-app degrades.
"""

from incident_env.server.engine.infrastructure import (
    CascadeRule,
    ServiceGraph,
    ServiceNode,
    ServiceStatus,
)
from incident_env.server.engine.grader import ScenarioGradingConfig
from incident_env.server.scenarios.base import BaseScenario


class RedisMemoryLeakScenario(BaseScenario):

    @property
    def scenario_id(self) -> str:
        return "easy_redis_oom"

    @property
    def difficulty(self) -> str:
        return "easy"

    @property
    def title(self) -> str:
        return "Redis OOM Catastrophe"

    @property
    def description(self) -> str:
        return (
            "The system is randomly logging out users. "
            "Session validation latency is through the roof. "
            "Cache layers seem unresponsive. Diagnose and stabilize the system."
        )

    def build_service_graph(self) -> ServiceGraph:
        services = [
            ServiceNode(
                name="session-store",
                display_name="Session Manager",
                status=ServiceStatus.DEGRADED,
                dependencies=["redis-cache"],
                port=4000,
                healthy_metrics={
                    "cpu_percent": 20.0,
                    "memory_percent": 30.0,
                    "latency_p50_ms": 5.0,
                },
                current_metrics={
                    "cpu_percent": 5.0,
                    "memory_percent": 30.0,
                    "latency_p50_ms": 3500.0,
                    "error_rate_percent": 40.0,
                },
                log_pattern="degraded",
                failure_description="Timeouts connecting to upstream cache",
                is_root_cause=False,
                fixable_by=["rollback"],
                fix_order=2,
            ),
            ServiceNode(
                name="redis-cache",
                display_name="Redis Session Cache",
                status=ServiceStatus.DEGRADED,
                dependencies=[],
                port=6379,
                healthy_metrics={
                    "memory_percent": 40.0,
                    "latency_p50_ms": 1.0,
                },
                current_metrics={
                    "memory_percent": 99.9,
                    "latency_p50_ms": 8000.0,
                    "error_rate_percent": 100.0,
                },
                log_pattern="oom_killed",
                failure_description="OOM Killed by kernel. Unbounded memory growth.",
                is_root_cause=True,
                fixable_by=["restart"],
                fix_order=1,
            ),
            ServiceNode(
                name="web-app",
                display_name="Main Web App",
                status=ServiceStatus.HEALTHY,
                dependencies=["session-store"],
                port=8080,
            ),
        ]

        cascade_rules = [
            CascadeRule(
                source="redis-cache",
                target="session-store",
                delay_minutes=3,
                target_status=ServiceStatus.DOWN,
            ),
            CascadeRule(
                source="session-store",
                target="web-app",
                delay_minutes=4,
                target_status=ServiceStatus.DEGRADED,
            ),
        ]

        return ServiceGraph(services, cascade_rules)

    def get_grading_config(self) -> ScenarioGradingConfig:
        return ScenarioGradingConfig(
            root_cause_service="redis-cache",
            root_cause_description="Redis unbounded memory growth leading to OOM",
            ground_truth_causal_chain=[
                "redis memory leak",
                "redis OOM limits hit",
                "session-store drops connections causing logouts",
            ],
            correct_fix_actions=[
                {"command": "restart_service", "target": "redis-cache"},
                {"command": "rollback_deploy", "target": "session-store"},
            ],
            correct_fix_order=["redis-cache", "session-store"],
            useful_investigation_targets=["redis-cache", "session-store"],
            max_optimal_steps=6,
            max_total_reward=0.80,
        )
