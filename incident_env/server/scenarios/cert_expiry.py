"""
Medium Scenario: Internal Certificate Expiry

Situation:
- An internal TLS cert expired, causing mTLS failures between microservices.
- External proxy still works, but internal connections fail silently or throw 502s.
- Root cause: cert-manager cache/expiry.
- Fix: Restart cert-manager (forces renewal) -> restart internal-gateway to pick it up.

Temporal evolution:
- If unfixed after 6 min, notification_svc completely fails.
"""

from incident_env.server.engine.infrastructure import (
    CascadeRule,
    ServiceGraph,
    ServiceNode,
    ServiceStatus,
)
from incident_env.server.engine.grader import ScenarioGradingConfig
from incident_env.server.scenarios.base import BaseScenario


class CertExpiryScenario(BaseScenario):

    @property
    def scenario_id(self) -> str:
        return "medium_cert_expiry"

    @property
    def difficulty(self) -> str:
        return "medium"

    @property
    def title(self) -> str:
        return "Internal mTLS Certificate Expiry"

    @property
    def description(self) -> str:
        return (
            "API routes are responding with 502 Bad Gateway. "
            "Customer-facing portals load but user actions fail on the backend. "
            "There are reports of SSL handshake errors in internal telemetry."
        )

    def build_service_graph(self) -> ServiceGraph:
        services = [
            ServiceNode(
                name="api-gateway",
                display_name="External API Gateway",
                status=ServiceStatus.DEGRADED,
                dependencies=["internal-gateway"],
                port=443,
                healthy_metrics={
                    "cpu_percent": 30.0,
                    "error_rate_percent": 0.1,
                },
                current_metrics={
                    "cpu_percent": 25.0,
                    "error_rate_percent": 65.0,  # Throwing 502s to users
                },
                log_pattern="degraded",
                failure_description="502 Bad Gateway from upstream servers",
                is_root_cause=False,
                fixable_by=["restart"],
                fix_order=3,
            ),
            ServiceNode(
                name="internal-gateway",
                display_name="Internal Service Mesh Proxy",
                status=ServiceStatus.DEGRADED,
                dependencies=["cert-manager", "user-service"],
                port=8443,
                healthy_metrics={
                    "cpu_percent": 40.0,
                    "error_rate_percent": 0.1,
                },
                current_metrics={
                    "cpu_percent": 15.0,
                    "error_rate_percent": 99.0,
                },
                log_pattern="degraded",
                failure_description="x509: certificate has expired or is not yet valid",
                is_root_cause=False,
                fixable_by=["restart"],
                fix_order=2,
            ),
            ServiceNode(
                name="cert-manager",
                display_name="Certificate Authority Manager",
                status=ServiceStatus.DEGRADED,
                dependencies=[],
                port=9090,
                healthy_metrics={
                    "cpu_percent": 5.0,
                    "error_rate_percent": 0.0,
                },
                current_metrics={
                    "cpu_percent": 80.0, # Spinning trying to renew but failing due to wedged process
                    "error_rate_percent": 100.0,
                },
                log_pattern="cert_expiry",
                failure_description="Failed to automatically rotate cluster wildcard certificate",
                is_root_cause=True,
                fixable_by=["restart"],
                fix_order=1,
            ),
            ServiceNode(
                name="user-service",
                display_name="User Profiling Service",
                status=ServiceStatus.HEALTHY,
                dependencies=[],
                port=8081,
            ),
            ServiceNode(
                name="notification-svc",
                display_name="Push Notifications",
                status=ServiceStatus.HEALTHY,
                dependencies=["cert-manager"],
                port=8082,
            ),
        ]

        cascade_rules = [
            CascadeRule(
                source="cert-manager",
                target="notification-svc",
                delay_minutes=6,
                target_status=ServiceStatus.DOWN,
            ),
        ]

        return ServiceGraph(services, cascade_rules)

    def get_grading_config(self) -> ScenarioGradingConfig:
        return ScenarioGradingConfig(
            root_cause_service="cert-manager",
            root_cause_description="Internal service mesh certificate expired",
            ground_truth_causal_chain=[
                "cert-manager failed to renew",
                "internal-gateway encounters x509 expiration",
                "api-gateway loses upstream connection and returns 502",
            ],
            correct_fix_actions=[
                {"command": "restart_service", "target": "cert-manager"},
                {"command": "restart_service", "target": "internal-gateway"},
            ],
            correct_fix_order=["cert-manager", "internal-gateway"],
            useful_investigation_targets=["internal-gateway", "cert-manager"],
            max_optimal_steps=7,
            max_total_reward=0.77,
        )
