"""
Realistic log generator for the incident response environment.

Produces log entries that look like real production service logs,
with timestamps, severity levels, service context, and error details
that match the current state of each service.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Dict, List

from incident_env.server.engine.infrastructure import ServiceNode, ServiceStatus


# ---------------------------------------------------------------------------
# Log templates by pattern
# ---------------------------------------------------------------------------

_LOG_TEMPLATES: Dict[str, List[str]] = {
    # Normal operation
    "normal": [
        "[{ts}] INFO  [{svc}] Request handled successfully | latency={lat}ms | status=200",
        "[{ts}] INFO  [{svc}] Health check passed | uptime=99.97%",
        "[{ts}] DEBUG [{svc}] Connection pool stats: active={conn}/100 | idle=55",
        "[{ts}] INFO  [{svc}] Processed batch of {batch} items | duration={dur}ms",
    ],

    # Database connection pool exhaustion
    "db_pool_exhaustion": [
        "[{ts}] ERROR [{svc}] Connection pool exhausted: active_connections=100/100 | waiting_threads=47",
        "[{ts}] WARN  [{svc}] Connection acquisition timeout after 30000ms | pool_size=100",
        "[{ts}] ERROR [{svc}] java.sql.SQLTransientConnectionException: HikariPool-1 - Connection is not available",
        "[{ts}] ERROR [{svc}] Query execution failed: could not obtain connection within 30s | query=SELECT * FROM users",
        "[{ts}] WARN  [{svc}] Pool stats: total=100, active=100, idle=0, waiting=52",
        "[{ts}] ERROR [{svc}] Healthcheck FAILED: database connection timeout after 5000ms",
    ],

    # Bad deployment (auth service)
    "bad_deploy_auth": [
        "[{ts}] ERROR [{svc}] JWT signature verification failed: invalid key format in v2.4.0",
        "[{ts}] ERROR [{svc}] Token generation error: RSA key pair mismatch after deployment",
        "[{ts}] WARN  [{svc}] Auth middleware rejecting requests: 0 valid tokens issued in last 60s",
        "[{ts}] ERROR [{svc}] POST /api/v1/auth/token 500 Internal Server Error | trace_id=abc123",
        "[{ts}] ERROR [{svc}] Deployed version v2.4.0 has incompatible JWT signing config",
        "[{ts}] INFO  [{svc}] Deploy event: v2.3.0 → v2.4.0 at {deploy_ts} by CI/CD pipeline",
    ],

    # Downstream victim (payment failing because of auth)
    "auth_victim": [
        "[{ts}] ERROR [{svc}] Auth token validation failed: upstream auth-service returned 500",
        "[{ts}] WARN  [{svc}] Cannot verify user session — auth dependency unavailable",
        "[{ts}] ERROR [{svc}] POST /api/v1/payments/process 401 Unauthorized | reason=invalid_token",
        "[{ts}] ERROR [{svc}] 47 payment requests failed in last 60s: auth_validation_error",
        "[{ts}] WARN  [{svc}] Circuit breaker OPEN for auth-service dependency | failures=50/50",
    ],

    # Thundering herd / load spike
    "thundering_herd": [
        "[{ts}] WARN  [{svc}] Incoming request rate surged: {rps} req/s (normal: 250 req/s)",
        "[{ts}] ERROR [{svc}] Thread pool exhausted: active_threads=200/200 | queued=1500",
        "[{ts}] ERROR [{svc}] Request rejected: server overloaded | status=503",
        "[{ts}] WARN  [{svc}] Memory pressure: heap usage at 94% | GC pause 850ms",
        "[{ts}] ERROR [{svc}] Timeout waiting for downstream response: 30000ms exceeded",
        "[{ts}] CRITICAL [{svc}] OOM killer triggered: process consuming 7.8GB/8GB",
    ],

    # CDN cache miss storm
    "cdn_cache_miss": [
        "[{ts}] INFO  [{svc}] Cache MISS rate elevated: 87% (normal: 5%)",
        "[{ts}] WARN  [{svc}] Origin pull rate: {rps} req/s to backend (normal: 12 req/s)",
        "[{ts}] INFO  [{svc}] Cache invalidation event completed at {deploy_ts}",
        "[{ts}] INFO  [{svc}] Serving stale content for 23% of requests while revalidating",
        "[{ts}] WARN  [{svc}] Edge node eu-west-1 reporting elevated origin traffic",
    ],

    # Load balancer overwhelmed
    "lb_overwhelmed": [
        "[{ts}] ERROR [{svc}] Backend pool health: 1/4 instances healthy",
        "[{ts}] WARN  [{svc}] Connection queue depth: 2500 (threshold: 500)",
        "[{ts}] ERROR [{svc}] 502 Bad Gateway: all backend instances timing out",
        "[{ts}] WARN  [{svc}] Active connections: 10000 (limit: 10000) — dropping new connections",
        "[{ts}] ERROR [{svc}] Health check failures for api-gateway-{inst}: 5 consecutive",
    ],

    # Recovery log
    "recovery": [
        "[{ts}] INFO  [{svc}] Service restarted successfully | pid={pid}",
        "[{ts}] INFO  [{svc}] Health check passed | status=200 | latency={lat}ms",
        "[{ts}] INFO  [{svc}] Connection pool initialized: 100 connections ready",
        "[{ts}] INFO  [{svc}] Accepting traffic | status=HEALTHY",
    ],

    # Rollback success
    "rollback_success": [
        "[{ts}] INFO  [{svc}] Deployment rollback initiated: v2.4.0 → v2.3.0",
        "[{ts}] INFO  [{svc}] Previous version restored successfully",
        "[{ts}] INFO  [{svc}] Health check passed after rollback | status=200",
        "[{ts}] INFO  [{svc}] All endpoints responding normally",
    ],

    # Scale success
    "scale_success": [
        "[{ts}] INFO  [{svc}] Horizontal scale-up complete: 2 → 4 instances",
        "[{ts}] INFO  [{svc}] Connection pool expanded: 100 → 200 max connections",
        "[{ts}] INFO  [{svc}] Load balanced across 4 healthy instances",
        "[{ts}] INFO  [{svc}] Resource allocation adjusted — service stabilized",
    ],

    # Auto-recovery after upstream fix (cascade victim self-healing)
    "auto_recovery": [
        "[{ts}] INFO  [{svc}] Service auto-recovered: upstream dependency restored",
        "[{ts}] INFO  [{svc}] Health check passed after upstream fix | status=200 | latency={lat}ms",
        "[{ts}] INFO  [{svc}] Connection pool re-established to upstream | active={conn}/100",
        "[{ts}] INFO  [{svc}] Resuming normal operation after cascade recovery | uptime_restored",
    ],

    # Worker queue backup
    "queue_backup": [
        "[{ts}] WARN  [{svc}] Queue depth: {depth} messages (normal: 50)",
        "[{ts}] ERROR [{svc}] Consumer lag: {lag}s behind producer",
        "[{ts}] WARN  [{svc}] Processing rate dropped: {rate} msg/s (normal: 500 msg/s)",
        "[{ts}] ERROR [{svc}] Dead letter queue growing: {dlq} unprocessable messages",
    ],

    # Cache failure
    "cache_failure": [
        "[{ts}] ERROR [{svc}] Redis connection refused: ECONNREFUSED 10.0.1.5:6379",
        "[{ts}] WARN  [{svc}] Cache fallback to database — expect elevated latency",
        "[{ts}] ERROR [{svc}] Cache hit rate: 0% (normal: 95%) — all requests hitting DB",
        "[{ts}] WARN  [{svc}] Memory eviction rate: 500 keys/s — possible memory pressure",
    ],

    # Generic degraded
    "degraded": [
        "[{ts}] WARN  [{svc}] Elevated error rate: {err}% of requests failing",
        "[{ts}] WARN  [{svc}] p99 latency: {lat}ms (SLO threshold: 200ms)",
        "[{ts}] ERROR [{svc}] Intermittent failures detected: {failures} in last 60s",
        "[{ts}] WARN  [{svc}] Dependency {dep} responding slowly: avg {dep_lat}ms",
    ],

    # Generic down
    "down": [
        "[{ts}] CRITICAL [{svc}] Service UNREACHABLE — all health checks failing",
        "[{ts}] ERROR [{svc}] Process exited with code 137 (OOM killed)",
        "[{ts}] CRITICAL [{svc}] No response on port {port} for 120 seconds",
        "[{ts}] ERROR [{svc}] Connection refused: Is the service running?",
    ],
}


def generate_logs(
    service: ServiceNode,
    env_time_minutes: int,
    num_entries: int = 8,
    base_time: datetime | None = None,
) -> str:
    """
    Generate realistic log entries for a service based on its current state.

    Parameters
    ----------
    service       : The service to generate logs for
    env_time_minutes : Current environment time in minutes
    num_entries   : Number of log entries to generate
    base_time     : Base datetime for timestamps (defaults to now)

    Returns
    -------
    Formatted multi-line log string
    """
    if base_time is None:
        base_time = datetime(2026, 4, 4, 3, 0, 0)  # 3:00 AM — prime incident time

    # Pick log template based on service state
    pattern = service.log_pattern

    # If no specific pattern but service is degraded/down, use generic
    if pattern == "normal" and service.status == ServiceStatus.DEGRADED:
        pattern = "degraded"
    elif pattern == "normal" and service.status == ServiceStatus.DOWN:
        pattern = "down"

    templates = _LOG_TEMPLATES.get(pattern, _LOG_TEMPLATES["normal"])

    entries = []
    for i in range(num_entries):
        # Timestamp progresses through the log window
        offset_seconds = (env_time_minutes * 60) - (num_entries - i) * random.randint(5, 30)
        offset_seconds = max(0, offset_seconds)
        ts = base_time + timedelta(seconds=offset_seconds)
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S.") + f"{random.randint(0, 999):03d}"

        template = random.choice(templates)
        entry = template.format(
            ts=ts_str,
            svc=service.name,
            lat=random.randint(5, 2000) if service.status != ServiceStatus.HEALTHY else random.randint(5, 50),
            conn=random.randint(80, 100) if service.status != ServiceStatus.HEALTHY else random.randint(20, 50),
            batch=random.randint(10, 500),
            dur=random.randint(50, 5000),
            pid=random.randint(1000, 9999),
            port=service.port,
            rps=random.randint(500, 3000),
            err=f"{service.current_metrics.get('error_rate_percent', 0.1):.1f}",
            failures=random.randint(20, 200),
            dep=random.choice(service.dependencies) if service.dependencies else "unknown",
            dep_lat=random.randint(500, 5000),
            deploy_ts=(base_time + timedelta(minutes=env_time_minutes - service.deploy_minutes_ago)).strftime("%H:%M:%S"),
            inst=random.randint(1, 4),
            depth=random.randint(500, 5000),
            lag=random.randint(10, 120),
            rate=random.randint(10, 100),
            dlq=random.randint(50, 500),
        )
        entries.append(entry)

    header = f"=== Logs for {service.display_name} ({service.name}) | Last {num_entries} entries ==="
    return header + "\n\n" + "\n".join(entries)
