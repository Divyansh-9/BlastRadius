"""
Medium Scenario: Bad Deployment Cascade

Situation:
- Auth service deployed v2.4.0 twelve minutes ago with broken JWT signing
- Payment service is FAILING because it can't validate auth tokens
- Red herring: payment logs say "auth token validation failed" — tempts
  agent to restart payment (which won't help)
- Correct fix: rollback auth-service deployment

Temporal evolution:
- If unfixed after 4 min: worker-queue starts backing up
- If unfixed after 7 min: cache-layer starts failing (can't refresh auth)
- If unfixed after 10 min: API gateway degrades (auth dependency)

This scenario tests root cause analysis vs. symptom chasing.
Expected baseline score: 0.4-0.6
"""

from incident_env.server.engine.infrastructure import (
    CascadeRule,
    ServiceGraph,
    ServiceNode,
    ServiceStatus,
)
from incident_env.server.engine.grader import ScenarioGradingConfig
from incident_env.server.scenarios.base import BaseScenario


class MediumScenario(BaseScenario):

    @property
    def scenario_id(self) -> str:
        return "medium_bad_deploy"

    @property
    def difficulty(self) -> str:
        return "medium"

    @property
    def title(self) -> str:
        return "Bad Deployment Cascade"

    @property
    def description(self) -> str:
        return (
            "Critical alert: Payment processing is DOWN. Users cannot complete "
            "purchases. Multiple services showing elevated error rates. "
            "The payment team says they haven't changed anything. "
            "Something upstream may be causing this. Find the root cause."
        )

    def build_service_graph(self) -> ServiceGraph:
        services = [
            ServiceNode(
                name="api-gateway",
                display_name="API Gateway",
                status=ServiceStatus.HEALTHY,
                dependencies=["auth-service"],
                port=8080,
            ),
            ServiceNode(
                name="auth-service",
                display_name="Auth Service",
                status=ServiceStatus.DOWN,
                dependencies=["database"],
                port=8081,
                is_root_cause=True,
                failure_description="JWT signing broken after v2.4.0 deployment",
                has_recent_deploy=True,
                deploy_minutes_ago=12,
                deploy_version="v2.4.0",
                previous_version="v2.3.0",
                fixable_by=["rollback"],
                fix_order=1,
                log_pattern="bad_deploy_auth",
                healthy_metrics={
                    "cpu_percent": 18.0,
                    "memory_percent": 30.0,
                    "latency_p50_ms": 8.0,
                    "latency_p99_ms": 25.0,
                    "error_rate_percent": 0.05,
                    "requests_per_sec": 400.0,
                    "active_connections": 30,
                },
                current_metrics={
                    "cpu_percent": 65.0,
                    "memory_percent": 55.0,
                    "latency_p50_ms": 500.0,
                    "latency_p99_ms": 5000.0,
                    "error_rate_percent": 95.0,
                    "requests_per_sec": 400.0,
                    "active_connections": 120,
                },
            ),
            ServiceNode(
                name="payment-service",
                display_name="Payment Service",
                status=ServiceStatus.DOWN,
                dependencies=["auth-service", "database"],
                port=8082,
                is_root_cause=False,  # VICTIM!
                failure_description="Cannot process payments — auth token validation failing",
                log_pattern="auth_victim",
                # Restarting payment won't help — it depends on auth
                fixable_by=["restart"],
                fix_order=2,  # Can only be fixed AFTER auth is fixed
                healthy_metrics={
                    "cpu_percent": 22.0,
                    "memory_percent": 45.0,
                    "latency_p50_ms": 20.0,
                    "latency_p99_ms": 80.0,
                    "error_rate_percent": 0.02,
                    "requests_per_sec": 200.0,
                    "active_connections": 50,
                },
                current_metrics={
                    "cpu_percent": 10.0,
                    "memory_percent": 40.0,
                    "latency_p50_ms": 0.0,
                    "latency_p99_ms": 0.0,
                    "error_rate_percent": 100.0,
                    "requests_per_sec": 0.0,
                    "active_connections": 200,
                },
            ),
            ServiceNode(
                name="database",
                display_name="PostgreSQL Database",
                status=ServiceStatus.HEALTHY,
                dependencies=[],
                port=5432,
            ),
            ServiceNode(
                name="worker-queue",
                display_name="Worker Queue",
                status=ServiceStatus.HEALTHY,
                dependencies=["auth-service", "database"],
                port=8083,
                log_pattern="normal",
            ),
            ServiceNode(
                name="cache-layer",
                display_name="Redis Cache",
                status=ServiceStatus.HEALTHY,
                dependencies=["auth-service"],
                port=6379,
                log_pattern="normal",
            ),
        ]

        cascade_rules = [
            # Worker queue backs up after 4 min of auth being down
            CascadeRule(
                source="auth-service",
                target="worker-queue",
                delay_minutes=4,
                target_status=ServiceStatus.DEGRADED,
            ),
            # Cache fails after 7 min (can't refresh auth tokens)
            CascadeRule(
                source="auth-service",
                target="cache-layer",
                delay_minutes=7,
                target_status=ServiceStatus.DEGRADED,
            ),
            # API gateway degrades after 10 min
            CascadeRule(
                source="auth-service",
                target="api-gateway",
                delay_minutes=10,
                target_status=ServiceStatus.DEGRADED,
            ),
        ]

        return ServiceGraph(services, cascade_rules)

    def get_grading_config(self) -> ScenarioGradingConfig:
        return ScenarioGradingConfig(
            root_cause_service="auth-service",
            root_cause_description="Bad deployment v2.4.0 broke JWT signing",
            ground_truth_causal_chain=[
                "auth-service deployed v2.4.0 with broken JWT signing config",
                "auth tokens are malformed or fail verification",
                "payment-service cannot validate user sessions",
                "all payment processing fails",
                "worker-queue backs up with unprocessable auth-dependent jobs",
            ],
            correct_fix_actions=[
                {"command": "rollback_deploy", "target": "auth-service"},
                {"command": "restart_service", "target": "payment-service"},
            ],
            correct_fix_order=["auth-service", "payment-service"],
            useful_investigation_targets=[
                "auth-service", "payment-service", "worker-queue",
            ],
            max_optimal_steps=8,
            # max_total_reward computed analytically by Grader.__init__
        )
