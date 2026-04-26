"""
Easy Scenario: Database Connection Pool Exhaustion

Situation:
- The database service has exhausted its connection pool (100/100 connections)
- API gateway is returning 503s because it can't get DB connections
- Fix is straightforward: scale the database connection pool

Temporal evolution:
- If unfixed after 4 min: API gateway degrades
- If unfixed after 8 min: API gateway goes DOWN

This scenario tests basic investigation and fix skills.
Expected baseline score: 0.7-0.9
"""

from incident_env.server.engine.infrastructure import (
    CascadeRule,
    ServiceGraph,
    ServiceNode,
    ServiceStatus,
)
from incident_env.server.engine.grader import ScenarioGradingConfig
from incident_env.server.scenarios.base import BaseScenario


class EasyScenario(BaseScenario):

    @property
    def scenario_id(self) -> str:
        return "easy_db_pool"

    @property
    def difficulty(self) -> str:
        return "easy"

    @property
    def title(self) -> str:
        return "Database Connection Pool Exhaustion"

    @property
    def description(self) -> str:
        return (
            "Users are reporting slow page loads and intermittent 503 errors. "
            "The on-call dashboard shows the database service with elevated latency. "
            "Investigate and resolve the issue before it impacts more services."
        )

    def build_service_graph(self) -> ServiceGraph:
        services = [
            ServiceNode(
                name="api-gateway",
                display_name="API Gateway",
                status=ServiceStatus.DEGRADED,
                dependencies=["database"],
                port=8080,
                healthy_metrics={
                    "cpu_percent": 20.0,
                    "memory_percent": 40.0,
                    "latency_p50_ms": 15.0,
                    "latency_p99_ms": 50.0,
                    "error_rate_percent": 0.1,
                    "requests_per_sec": 300.0,
                    "active_connections": 60,
                },
                current_metrics={
                    "cpu_percent": 45.0,
                    "memory_percent": 55.0,
                    "latency_p50_ms": 800.0,
                    "latency_p99_ms": 5000.0,
                    "error_rate_percent": 12.5,
                    "requests_per_sec": 180.0,
                    "active_connections": 95,
                },
                log_pattern="degraded",
                failure_description="Intermittent 503 errors — database connection timeouts",
                # This is a victim, not the root cause
                is_root_cause=False,
                fixable_by=["restart"],
                fix_order=2,  # Must fix DB first
            ),
            ServiceNode(
                name="database",
                display_name="PostgreSQL Database",
                status=ServiceStatus.DEGRADED,
                dependencies=[],
                port=5432,
                healthy_metrics={
                    "cpu_percent": 25.0,
                    "memory_percent": 50.0,
                    "latency_p50_ms": 5.0,
                    "latency_p99_ms": 20.0,
                    "error_rate_percent": 0.0,
                    "requests_per_sec": 500.0,
                    "active_connections": 45,
                },
                current_metrics={
                    "cpu_percent": 85.0,
                    "memory_percent": 78.0,
                    "latency_p50_ms": 200.0,
                    "latency_p99_ms": 8000.0,
                    "error_rate_percent": 8.0,
                    "requests_per_sec": 120.0,
                    "active_connections": 100,
                },
                log_pattern="db_pool_exhaustion",
                failure_description="Connection pool exhausted: 100/100 active connections",
                is_root_cause=True,
                fixable_by=["scale"],
                fix_params={"max_connections": 200},
                fix_order=1,
            ),
            ServiceNode(
                name="auth-service",
                display_name="Auth Service",
                status=ServiceStatus.HEALTHY,
                dependencies=["database"],
                port=8081,
            ),
            ServiceNode(
                name="payment-service",
                display_name="Payment Service",
                status=ServiceStatus.HEALTHY,
                dependencies=["auth-service", "database"],
                port=8082,
            ),
        ]

        cascade_rules = [
            # If DB is degraded for 4 min, API gateway degrades further
            CascadeRule(
                source="database",
                target="api-gateway",
                delay_minutes=4,
                target_status=ServiceStatus.DOWN,
            ),
            # If DB is degraded for 6 min, auth starts struggling
            CascadeRule(
                source="database",
                target="auth-service",
                delay_minutes=6,
                target_status=ServiceStatus.DEGRADED,
            ),
        ]

        return ServiceGraph(services, cascade_rules)

    def get_grading_config(self) -> ScenarioGradingConfig:
        return ScenarioGradingConfig(
            root_cause_service="database",
            root_cause_description="Connection pool exhausted at 100/100 connections",
            ground_truth_causal_chain=[
                "database connection pool exhausted",
                "API gateway cannot acquire connections",
                "users see 503 errors and slow responses",
            ],
            correct_fix_actions=[
                {"command": "scale_service", "target": "database"},
            ],
            correct_fix_order=["database"],
            useful_investigation_targets=["database", "api-gateway"],
            max_optimal_steps=5,
            # max_total_reward computed analytically by Grader.__init__
        )
