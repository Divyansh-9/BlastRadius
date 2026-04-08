"""
Hard Scenario: WAF Regex Catastrophe

Situation:
- A bad WAF (Web Application Firewall) regex rule with excessive backtracking was deployed
- CPU spikes to 100% across the edge firewall, causing massive queuing
- All upstream services show high CPU (waiting on IO/event loop starvation) making it look like a DDoS
- Root cause: waf-engine (bad deploy)
- Fix: Rollback waf-engine -> Restart edge-proxy -> Restart origin-server

Temporal evolution:
- If unfixed after 2 min, edge-proxy is DOWN
- If unfixed after 5 min, origin-server is DOWN
"""

from incident_env.server.engine.infrastructure import (
    CascadeRule,
    ServiceGraph,
    ServiceNode,
    ServiceStatus,
)
from incident_env.server.engine.grader import ScenarioGradingConfig
from incident_env.server.scenarios.base import BaseScenario


class RegexCatastropheScenario(BaseScenario):

    @property
    def scenario_id(self) -> str:
        return "hard_regex_catastrophe"

    @property
    def difficulty(self) -> str:
        return "hard"

    @property
    def title(self) -> str:
        return "WAF Regex Catastrophe"

    @property
    def description(self) -> str:
        return (
            "CPU usage is pegged at 100% across multiple infrastructure layers. "
            "Traffic is dropping severely, resembling a massive DDoS attack. "
            "Edge nodes are timing out and dropping 99% of requests."
        )

    def build_service_graph(self) -> ServiceGraph:
        services = [
            ServiceNode(
                name="edge-proxy",
                display_name="Edge Traffic Proxy",
                status=ServiceStatus.DEGRADED,
                dependencies=["waf-engine", "origin-server"],
                port=80,
                healthy_metrics={
                    "cpu_percent": 15.0,
                    "latency_p50_ms": 2.0,
                    "error_rate_percent": 0.01,
                },
                current_metrics={
                    "cpu_percent": 99.9, # Event loop starvation waiting on WAF
                    "latency_p50_ms": 15000.0,
                    "error_rate_percent": 85.0,
                },
                log_pattern="degraded",
                failure_description="Timeouts proxying to origin. Thread pool exhausted.",
                is_root_cause=False,
                fixable_by=["restart"],
                fix_order=2,
            ),
            ServiceNode(
                name="waf-engine",
                display_name="Web Application Firewall (WAF)",
                status=ServiceStatus.DEGRADED,
                dependencies=[],
                port=8080,
                healthy_metrics={
                    "cpu_percent": 25.0,
                    "latency_p50_ms": 1.0,
                },
                current_metrics={
                    "cpu_percent": 100.0,
                    "latency_p50_ms": 25000.0,
                    "error_rate_percent": 95.0,
                },
                log_pattern="degraded",
                failure_description="ReDoS (Regex Denial of Service): catastrophic backtracking on new ruleset.",
                is_root_cause=True,
                fixable_by=["rollback"],
                fix_order=1,
            ),
            ServiceNode(
                name="origin-server",
                display_name="Origin API Server",
                status=ServiceStatus.DEGRADED,
                dependencies=[],
                port=443,
                healthy_metrics={
                    "cpu_percent": 30.0,
                },
                current_metrics={
                    "cpu_percent": 90.0, # High CPU from TCP connection queuing
                },
                log_pattern="degraded",
                failure_description="Dropping connections: accept queue overflow.",
                is_root_cause=False,
                fixable_by=["restart"],
                fix_order=3,
            ),
            ServiceNode(
                name="static-cdn",
                display_name="Static Assets CDN",
                status=ServiceStatus.HEALTHY,
                dependencies=[],
                port=444,
            ),
            ServiceNode(
                name="log-pipeline",
                display_name="Telemetry Pipeline",
                status=ServiceStatus.DEGRADED,
                dependencies=["edge-proxy"],
                port=5044,
                healthy_metrics={"cpu_percent": 10.0},
                current_metrics={"cpu_percent": 100.0},
                log_pattern="degraded",
                failure_description="Unable to parse malformed traffic patterns.",
                is_root_cause=False,
                fixable_by=["restart"],
                fix_order=4,
            ),
        ]

        cascade_rules = [
            CascadeRule(
                source="waf-engine",
                target="edge-proxy",
                delay_minutes=2,
                target_status=ServiceStatus.DOWN,
            ),
            CascadeRule(
                source="edge-proxy",
                target="origin-server",
                delay_minutes=5,
                target_status=ServiceStatus.DOWN,
            ),
        ]

        return ServiceGraph(services, cascade_rules)

    def get_grading_config(self) -> ScenarioGradingConfig:
        return ScenarioGradingConfig(
            root_cause_service="waf-engine",
            root_cause_description="Catastrophic regex backtracking in WAF ruleset causing CPU starvation",
            ground_truth_causal_chain=[
                "waf-engine regex pegging CPU to 100%",
                "edge-proxy thread pool queues up waiting for WAF",
                "origin-server socket queue overflows from stale TCP connections",
            ],
            correct_fix_actions=[
                {"command": "rollback_deploy", "target": "waf-engine"},
                {"command": "restart_service", "target": "edge-proxy"},
                {"command": "restart_service", "target": "origin-server"},
            ],
            correct_fix_order=["waf-engine", "edge-proxy", "origin-server"],
            useful_investigation_targets=["waf-engine", "edge-proxy", "origin-server"],
            max_optimal_steps=8,
            max_total_reward=0.85,
        )
