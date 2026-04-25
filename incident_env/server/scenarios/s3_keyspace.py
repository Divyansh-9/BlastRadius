"""
Hard Scenario: AWS S3 Metadata Index Overflow

Situation:
- A batch job is mass deleting objects.
- It exceeds the metadata index capacity, causing it to fall behind. Read operations time out.
- Writes still work but queue infinitely.
- Root cause: batch-processor
- Fix: Stop batch processor -> Scale metadata_index -> restart api-layer.

Temporal evolution:
- If unfixed after 3 min: api-layer DOWN.
- If unfixed after 6 min: backup-service DEGRADED.
"""

from incident_env.server.engine.infrastructure import (
    CascadeRule,
    ServiceGraph,
    ServiceNode,
    ServiceStatus,
)
from incident_env.server.engine.grader import ScenarioGradingConfig
from incident_env.server.scenarios.base import BaseScenario


class S3KeyspaceScenario(BaseScenario):

    @property
    def scenario_id(self) -> str:
        return "hard_s3_keyspace_overflow"

    @property
    def difficulty(self) -> str:
        return "hard"

    @property
    def title(self) -> str:
        return "Object Storage Keyspace Overflow"

    @property
    def description(self) -> str:
        return (
            "API read latency is spiking massively for object storage endpoints. "
            "Write operations appear to be succeeding but slowly. "
            "Internal alerts fire for metadata index saturation."
        )

    def build_service_graph(self) -> ServiceGraph:
        services = [
            ServiceNode(
                name="batch-processor",
                display_name="Mass Cleanup Batch Job",
                status=ServiceStatus.DEGRADED,
                dependencies=[],
                port=8080,
                healthy_metrics={
                    "requests_per_sec": 50.0,
                },
                current_metrics={
                    "requests_per_sec": 50000.0,
                },
                log_pattern="degraded",
                failure_description="Aggressively issuing DELETE operations. Rate limits bypassed.",
                is_root_cause=True,
                fixable_by=["rollback"], # Stop the job
                fix_order=1,
            ),
            ServiceNode(
                name="metadata-index",
                display_name="Storage Metadata Indexer",
                status=ServiceStatus.DEGRADED,
                dependencies=["batch-processor"],
                port=9200,
                healthy_metrics={
                    "cpu_percent": 30.0,
                    "latency_p50_ms": 1.0,
                },
                current_metrics={
                    "cpu_percent": 100.0,
                    "latency_p50_ms": 12000.0,
                },
                log_pattern="degraded",
                failure_description="Write queue backlog exceeding hard limits. Reads timing out.",
                is_root_cause=False,
                fixable_by=["scale"],
                fix_params={"instances": 5},
                fix_order=2,
            ),
            ServiceNode(
                name="object-store",
                display_name="Blob Storage Engine",
                status=ServiceStatus.HEALTHY, # Storage is fine, index is broken
                dependencies=["metadata-index"],
                port=9000,
            ),
            ServiceNode(
                name="api-layer",
                display_name="Customer API Layer",
                status=ServiceStatus.DEGRADED,
                dependencies=["object-store"],
                port=443,
                healthy_metrics={
                    "error_rate_percent": 0.0,
                },
                current_metrics={
                    "error_rate_percent": 60.0,
                },
                log_pattern="degraded",
                failure_description="Upstream storage index timeouts processing GET requests.",
                is_root_cause=False,
                fixable_by=["restart"],
                fix_order=3,
            ),
            ServiceNode(
                name="backup-service",
                display_name="Nightly Snapshot Service",
                status=ServiceStatus.HEALTHY,
                dependencies=["object-store"],
                port=8111,
            ),
        ]

        cascade_rules = [
            CascadeRule(
                source="metadata-index",
                target="api-layer",
                delay_minutes=3,
                target_status=ServiceStatus.DOWN,
            ),
            CascadeRule(
                source="api-layer",
                target="backup-service",
                delay_minutes=6,
                target_status=ServiceStatus.DEGRADED,
            ),
        ]

        return ServiceGraph(services, cascade_rules)

    def get_grading_config(self) -> ScenarioGradingConfig:
        return ScenarioGradingConfig(
            root_cause_service="batch-processor",
            root_cause_description="Runaway batch deletion exceeding index bounds",
            ground_truth_causal_chain=[
                "batch-processor issues 50k deletes/sec",
                "metadata-index queue backs up causing read starvation",
                "api-layer times out trying to read objects",
            ],
            correct_fix_actions=[
                {"command": "rollback_deploy", "target": "batch-processor"},
                {"command": "scale_service", "target": "metadata-index"},
                {"command": "restart_service", "target": "api-layer"},
            ],
            correct_fix_order=["batch-processor", "metadata-index", "api-layer"],
            useful_investigation_targets=["batch-processor", "metadata-index", "api-layer"],
            max_optimal_steps=8,
            max_total_reward=0.77,
        )
