"""
Medium Scenario: Kubernetes Pod Eviction Storm

Situation:
- A noisy neighbor pod uses too much memory.
- The Kubelet begins evicting pods rapidly, overloading other nodes.
- API and worker pods are killed.
- Root cause: noisy-pod configuration.
- Fix: Scale down noisy-pod -> restart k8s-scheduler -> restart api-pods.

Temporal evolution:
- If unfixed after 4 min, worker-pods get evicted.
"""

from incident_env.server.engine.infrastructure import (
    CascadeRule,
    ServiceGraph,
    ServiceNode,
    ServiceStatus,
)
from incident_env.server.engine.grader import ScenarioGradingConfig
from incident_env.server.scenarios.base import BaseScenario


class K8sEvictionScenario(BaseScenario):

    @property
    def scenario_id(self) -> str:
        return "medium_k8s_eviction"

    @property
    def difficulty(self) -> str:
        return "medium"

    @property
    def title(self) -> str:
        return "Kubernetes Pod Eviction Storm"

    @property
    def description(self) -> str:
        return (
            "Multiple services are randomly restarting. "
            "P99 latency is highly erratic. Node memory pressure alerts are firing across the cluster. "
            "Identify the root cause of the resource exhaustion and stabilize the cluster."
        )

    def build_service_graph(self) -> ServiceGraph:
        services = [
            ServiceNode(
                name="api-pods",
                display_name="API Gateway Pods",
                status=ServiceStatus.DEGRADED,
                dependencies=["k8s-scheduler", "node-pool"],
                port=8080,
                healthy_metrics={
                    "cpu_percent": 30.0,
                    "memory_percent": 45.0,
                },
                current_metrics={
                    "cpu_percent": 90.0,
                    "memory_percent": 10.0,
                    "error_rate_percent": 35.0,
                },
                log_pattern="degraded",
                failure_description="SIGKILL received. Pod evicted due to node memory pressure.",
                is_root_cause=False,
                fixable_by=["restart"],
                fix_order=3,
            ),
            ServiceNode(
                name="node-pool",
                display_name="Worker Node Pool",
                status=ServiceStatus.DEGRADED,
                dependencies=["noisy-pod"],
                port=10250,
                healthy_metrics={
                    "memory_percent": 60.0,
                },
                current_metrics={
                    "memory_percent": 99.9,
                },
                log_pattern="degraded",
                failure_description="MemoryPressure condition true. Attempting to reclaim resources.",
                is_root_cause=False,
                fixable_by=[],
                fix_order=0,
            ),
            ServiceNode(
                name="noisy-pod",
                display_name="Data Ingestion Job",
                status=ServiceStatus.DEGRADED,
                dependencies=[],
                port=5050,
                healthy_metrics={
                    "memory_percent": 20.0,
                },
                current_metrics={
                    "memory_percent": 100.0,
                },
                log_pattern="degraded",
                failure_description="Loading entire dataset into memory. No limits configured.",
                is_root_cause=True,
                fixable_by=["scale"],
                fix_params={"instances": 0}, # Must scale down to 0 to stop the bleeding
                fix_order=1,
            ),
            ServiceNode(
                name="k8s-scheduler",
                display_name="Kubernetes Scheduler",
                status=ServiceStatus.DEGRADED,
                dependencies=["node-pool"],
                port=10251,
                healthy_metrics={
                    "cpu_percent": 10.0,
                },
                current_metrics={
                    "cpu_percent": 100.0,
                },
                log_pattern="degraded",
                failure_description="Failed to schedule pods: no nodes available with sufficient memory.",
                is_root_cause=False,
                fixable_by=["restart"],
                fix_order=2,
            ),
            ServiceNode(
                name="worker-pods",
                display_name="Background Workers",
                status=ServiceStatus.HEALTHY,
                dependencies=["k8s-scheduler", "node-pool"],
                port=8081,
            ),
        ]

        cascade_rules = [
            CascadeRule(
                source="node-pool",
                target="worker-pods",
                delay_minutes=4,
                target_status=ServiceStatus.DEGRADED,
            ),
        ]

        return ServiceGraph(services, cascade_rules)

    def get_grading_config(self) -> ScenarioGradingConfig:
        return ScenarioGradingConfig(
            root_cause_service="noisy-pod",
            root_cause_description="Unbounded memory usage in data ingestion pod causing node pressure",
            ground_truth_causal_chain=[
                "noisy-pod exhausts memory",
                "node-pool triggers eviction",
                "api-pods get SIGKILL and scheduler thrashes",
            ],
            correct_fix_actions=[
                {"command": "scale_service", "target": "noisy-pod"},
                {"command": "restart_service", "target": "k8s-scheduler"},
                {"command": "restart_service", "target": "api-pods"},
            ],
            correct_fix_order=["noisy-pod", "k8s-scheduler", "api-pods"],
            useful_investigation_targets=["node-pool", "noisy-pod", "api-pods"],
            max_optimal_steps=8,
            max_total_reward=0.75,
        )
