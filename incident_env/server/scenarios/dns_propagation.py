"""
Easy Scenario: DNS Propagation Failure

Situation:
- A DNS TTL was set too low (5 minutes) after a migration.
- Many users are hitting the old stale load balancer routing to dead servers.
- The web frontend is degrading due to connection drops.
- Root cause is the dns-resolver cache.
- Fix: Flush dns cache (restart load-balancer)

Temporal evolution:
- If unfixed after 5 min: Web-frontend degrades and drops 50% traffic.
"""

from incident_env.server.engine.infrastructure import (
    CascadeRule,
    ServiceGraph,
    ServiceNode,
    ServiceStatus,
)
from incident_env.server.engine.grader import ScenarioGradingConfig
from incident_env.server.scenarios.base import BaseScenario


class DnsPropagationScenario(BaseScenario):

    @property
    def scenario_id(self) -> str:
        return "easy_dns_propagation"

    @property
    def difficulty(self) -> str:
        return "easy"

    @property
    def title(self) -> str:
        return "Stale DNS TTL Propagation"

    @property
    def description(self) -> str:
        return (
            "Users report that the web app is sporadically loading. "
            "Traffic dropped sharply at edge nodes right after an infrastructure migration. "
            "Investigate load balancing and DNS resolution."
        )

    def build_service_graph(self) -> ServiceGraph:
        services = [
            ServiceNode(
                name="web-frontend",
                display_name="Web Frontend",
                status=ServiceStatus.DEGRADED,
                dependencies=["api-backend"],
                port=3000,
                healthy_metrics={
                    "cpu_percent": 15.0,
                    "memory_percent": 30.0,
                    "latency_p50_ms": 25.0,
                    "error_rate_percent": 0.05,
                    "requests_per_sec": 500.0,
                },
                current_metrics={
                    "cpu_percent": 10.0,  # CPU is actually low because traffic is lost
                    "memory_percent": 30.0,
                    "latency_p50_ms": 3000.0,
                    "error_rate_percent": 45.0,
                    "requests_per_sec": 220.0,
                },
                log_pattern="degraded",
                failure_description="50% of traffic is lost due to DNS timeouts",
                is_root_cause=False,
                fixable_by=["restart"],
                fix_order=2,
            ),
            ServiceNode(
                name="load-balancer",
                display_name="Edge Load Balancer",
                status=ServiceStatus.DEGRADED,
                dependencies=["web-frontend"],
                port=80,
                healthy_metrics={
                    "cpu_percent": 10.0,
                    "error_rate_percent": 0.01,
                    "requests_per_sec": 1000.0,
                },
                current_metrics={
                    "cpu_percent": 25.0,
                    "error_rate_percent": 30.0,
                    "requests_per_sec": 600.0,
                },
                log_pattern="degraded",
                failure_description="Routing table contains dead IP addresses",
                is_root_cause=False,
                fixable_by=["restart"],
                fix_order=1,
            ),
            ServiceNode(
                name="dns-resolver",
                display_name="Internal DNS Cache",
                status=ServiceStatus.DEGRADED,
                dependencies=[],
                port=53,
                healthy_metrics={
                    "cpu_percent": 5.0,
                    "error_rate_percent": 0.0,
                    "requests_per_sec": 2000.0,
                    "active_connections": 10,
                },
                current_metrics={
                    "cpu_percent": 5.0,
                    "error_rate_percent": 0.0,
                    "requests_per_sec": 2000.0,
                    "active_connections": 10,
                },
                log_pattern="dns_stale_cache",  # Needs matching text in log_generator.py naturally
                failure_description="Serving stale IP resolutions despite upstream changes",
                is_root_cause=True,
                fixable_by=["restart", "rollback"],
                fix_order=1,
            ),
            ServiceNode(
                name="api-backend",
                display_name="API Backend",
                status=ServiceStatus.HEALTHY,
                dependencies=[],
                port=8080,
            ),
        ]

        cascade_rules = [
            CascadeRule(
                source="dns-resolver",
                target="web-frontend",
                delay_minutes=5,
                target_status=ServiceStatus.DOWN,
            ),
        ]

        return ServiceGraph(services, cascade_rules)

    def get_grading_config(self) -> ScenarioGradingConfig:
        return ScenarioGradingConfig(
            root_cause_service="dns-resolver",
            root_cause_description="Stale DNS cache with low TTL causing bad routing",
            ground_truth_causal_chain=[
                "stale dns cache",
                "load balancer routes to dead IPs",
                "frontend traffic drops heavily",
            ],
            correct_fix_actions=[
                {"command": "restart_service", "target": "dns-resolver"},
            ],
            correct_fix_order=["dns-resolver"],
            useful_investigation_targets=["dns-resolver", "load-balancer", "web-frontend"],
            max_optimal_steps=5,
            max_total_reward=0.77,
        )
