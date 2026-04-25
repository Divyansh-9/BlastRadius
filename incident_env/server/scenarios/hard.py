"""
Hard Scenario: Thundering Herd After CDN Cache Invalidation

Situation:
- CDN cache was invalidated (routine operation, NOT the root cause)
- All traffic now hits the load balancer directly (cache miss storm)
- Load balancer overwhelmed → API gateway crushed → database connection storm
- MISLEADING: CDN metrics spike looks like CDN is broken (it's not — it's
  doing exactly what it should during a cache miss)
- REAL root cause: API gateway needs to be scaled to handle the surge
- Fix ORDER matters:
  1. First: scale API gateway (absorb traffic)
  2. Then: scale database (handle connection surge)  
  3. Finally: warm CDN cache (reduce ongoing traffic to backend)

Wrong order: Scaling database first causes thundering herd on API gateway → crash

Temporal evolution:
- If unfixed after 3 min: database starts degrading (conn storm)
- If unfixed after 5 min: auth-service degrades (can't reach DB)
- If unfixed after 8 min: payment-service goes DOWN
- If unfixed after 12 min: everything is DOWN

This scenario tests causal reasoning under pressure with misleading signals.
Expected baseline score: 0.1-0.3
"""

from incident_env.server.engine.infrastructure import (
    CascadeRule,
    ServiceGraph,
    ServiceNode,
    ServiceStatus,
)
from incident_env.server.engine.grader import ScenarioGradingConfig
from incident_env.server.scenarios.base import BaseScenario


class HardScenario(BaseScenario):

    @property
    def scenario_id(self) -> str:
        return "hard_thundering_herd"

    @property
    def difficulty(self) -> str:
        return "hard"

    @property
    def title(self) -> str:
        return "Thundering Herd After CDN Cache Invalidation"

    @property
    def description(self) -> str:
        return (
            "🔴 P1 INCIDENT: Multiple services cascading. API gateway overwhelmed, "
            "database under extreme load, payment processing failing. "
            "CDN metrics show massive traffic spike. "
            "Four services affected and spreading. Fix them in the right order "
            "or risk making things worse."
        )

    def build_service_graph(self) -> ServiceGraph:
        services = [
            # CDN 1
            ServiceNode(
                name="cdn-1",
                display_name="CDN / Edge Cache (us-east)",
                status=ServiceStatus.HEALTHY,
                dependencies=[],
                port=443,
                log_pattern="cdn_cache_miss",
                healthy_metrics={
                    "cpu_percent": 10.0,
                    "memory_percent": 20.0,
                    "latency_p50_ms": 2.0,
                    "latency_p99_ms": 10.0,
                    "error_rate_percent": 0.0,
                    "requests_per_sec": 2500.0,
                    "active_connections": 100,
                },
                current_metrics={
                    "cpu_percent": 65.0,
                    "memory_percent": 55.0,
                    "latency_p50_ms": 150.0,
                    "latency_p99_ms": 800.0,
                    "error_rate_percent": 2.0,
                    "requests_per_sec": 2500.0,
                    "active_connections": 2400,
                },
                failure_description="Cache miss rate 87% — EXPECTED BEHAVIOR during cache invalidation, NOT the root cause",
            ),
            
            # CDN 2 (Per User Request for two servers)
            ServiceNode(
                name="cdn-2",
                display_name="CDN / Edge Cache (eu-west)",
                status=ServiceStatus.HEALTHY,
                dependencies=[],
                port=443,
                log_pattern="cdn_cache_miss",
                healthy_metrics={
                    "cpu_percent": 12.0,
                    "memory_percent": 22.0,
                    "latency_p50_ms": 2.5,
                    "latency_p99_ms": 12.0,
                    "error_rate_percent": 0.0,
                    "requests_per_sec": 2500.0,
                    "active_connections": 100,
                },
                current_metrics={
                    "cpu_percent": 68.0,
                    "memory_percent": 58.0,
                    "latency_p50_ms": 160.0,
                    "latency_p99_ms": 850.0,
                    "error_rate_percent": 2.5,
                    "requests_per_sec": 2500.0,
                    "active_connections": 2400,
                },
                failure_description="Cache miss rate 88% — all traffic hitting origin",
            ),

            # Load Balancer — overwhelmed by the traffic surge
            ServiceNode(
                name="load-balancer",
                display_name="Load Balancer",
                status=ServiceStatus.DEGRADED,
                dependencies=["cdn-1", "cdn-2"],
                port=80,
                log_pattern="lb_overwhelmed",
                failure_description="Connection queue depth 2500+ — dropping requests",
                is_root_cause=False,
                healthy_metrics={
                    "cpu_percent": 15.0,
                    "memory_percent": 25.0,
                    "latency_p50_ms": 1.0,
                    "latency_p99_ms": 5.0,
                    "error_rate_percent": 0.01,
                    "requests_per_sec": 1000.0,
                    "active_connections": 100,
                },
                current_metrics={
                    "cpu_percent": 92.0,
                    "memory_percent": 78.0,
                    "latency_p50_ms": 500.0,
                    "latency_p99_ms": 10000.0,
                    "error_rate_percent": 35.0,
                    "requests_per_sec": 4500.0,
                    "active_connections": 10000,
                },
                fixable_by=["scale"],
                fix_order=2,
            ),

            # API Gateway — crushed by load
            ServiceNode(
                name="api-gateway",
                display_name="API Gateway",
                status=ServiceStatus.DOWN,
                dependencies=["load-balancer"],
                port=8080,
                log_pattern="thundering_herd",
                failure_description="Thread pool exhausted — OOM killer triggered",
                is_root_cause=True,  # This is where the fix needs to start
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
                    "cpu_percent": 0.0,
                    "memory_percent": 0.0,
                    "latency_p50_ms": 0.0,
                    "latency_p99_ms": 0.0,
                    "error_rate_percent": 100.0,
                    "requests_per_sec": 0.0,
                    "active_connections": 0,
                },
                fixable_by=["scale"],
                fix_params={"instances": 4, "memory_gb": 16},
                fix_order=1,  # MUST fix first
            ),

            # Database — connection storm from retries
            ServiceNode(
                name="database",
                display_name="PostgreSQL Database",
                status=ServiceStatus.DEGRADED,
                dependencies=[],
                port=5432,
                log_pattern="db_pool_exhaustion",
                failure_description="Connection storm: 200+ concurrent connections from retries",
                is_root_cause=False,
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
                    "cpu_percent": 88.0,
                    "memory_percent": 82.0,
                    "latency_p50_ms": 500.0,
                    "latency_p99_ms": 12000.0,
                    "error_rate_percent": 15.0,
                    "requests_per_sec": 100.0,
                    "active_connections": 200,
                },
                fixable_by=["scale"],
                fix_params={"max_connections": 500},
                fix_order=3,  # Fix AFTER api-gateway
            ),

            # Auth — degraded because DB is slow
            ServiceNode(
                name="auth-service",
                display_name="Auth Service",
                status=ServiceStatus.HEALTHY,  # Starts healthy, cascades later
                dependencies=["database"],
                port=8081,
            ),

            # Payment — will cascade if unfixed
            ServiceNode(
                name="payment-service",
                display_name="Payment Service",
                status=ServiceStatus.HEALTHY,
                dependencies=["auth-service", "database", "api-gateway"],
                port=8082,
            ),
        ]

        cascade_rules = [
            # Database degrades further after 3 min of LB being overwhelmed
            CascadeRule(
                source="load-balancer",
                target="database",
                delay_minutes=3,
                target_status=ServiceStatus.DOWN,
            ),
            # Auth starts failing after 5 min (DB dependency)
            CascadeRule(
                source="database",
                target="auth-service",
                delay_minutes=5,
                target_status=ServiceStatus.DEGRADED,
            ),
            # Payment goes down after 8 min (cascading from auth + db)
            CascadeRule(
                source="auth-service",
                target="payment-service",
                delay_minutes=8,
                target_status=ServiceStatus.DOWN,
            ),
            # If LB is degraded 12 min, auth goes DOWN entirely
            CascadeRule(
                source="database",
                target="auth-service",
                delay_minutes=12,
                target_status=ServiceStatus.DOWN,
            ),
        ]

        return ServiceGraph(services, cascade_rules)

    def get_grading_config(self) -> ScenarioGradingConfig:
        return ScenarioGradingConfig(
            root_cause_service="api-gateway",
            root_cause_description=(
                "CDN cache invalidation caused traffic surge → API gateway "
                "overwhelmed and OOM killed → connection storm to database"
            ),
            ground_truth_causal_chain=[
                "CDN cache invalidation caused 87% cache miss rate",
                "all user traffic forwarded directly to load balancer",
                "load balancer connection queue overwhelmed (2500+ queued)",
                "API gateway thread pool exhausted and OOM killed",
                "database hit with connection storm from retry floods",
                "auth and payment services cascade failing",
            ],
            correct_fix_actions=[
                {"command": "scale_service", "target": "api-gateway"},
                {"command": "scale_service", "target": "load-balancer"},
                {"command": "scale_service", "target": "database"},
            ],
            correct_fix_order=["api-gateway", "load-balancer", "database"],
            useful_investigation_targets=[
                "api-gateway", "load-balancer", "database",
                # cdn intentionally excluded: it's a red herring (healthy but misleading metrics)
            ],
            max_optimal_steps=12,
            max_total_reward=1.22,
        )
