"""
Metrics generator for the incident response environment.

Produces realistic metrics snapshots that an SRE would see
in a monitoring dashboard (Datadog/Grafana style).
"""

from __future__ import annotations

from typing import Dict

from incident_env.server.engine.infrastructure import ServiceNode, ServiceStatus


def generate_metrics_report(service: ServiceNode, env_time_minutes: int) -> str:
    """
    Generate a human-readable metrics report for a service.

    Looks like a Datadog/Grafana dashboard snapshot.
    """
    m = service.current_metrics
    status_icon = {
        ServiceStatus.HEALTHY: "🟢 HEALTHY",
        ServiceStatus.DEGRADED: "🟡 DEGRADED",
        ServiceStatus.DOWN: "🔴 DOWN",
        ServiceStatus.RESTARTING: "🔄 RESTARTING",
    }.get(service.status, "⚪ UNKNOWN")

    lines = [
        f"=== Metrics Dashboard: {service.display_name} ({service.name}) ===",
        f"Status: {status_icon}",
        f"Time: T+{env_time_minutes} min since incident start",
        "",
        "─── Resource Utilization ────────────────────────",
        f"  CPU Usage:        {m.get('cpu_percent', 0):6.1f}%  {'▓' * int(m.get('cpu_percent', 0) / 5)}{'░' * (20 - int(m.get('cpu_percent', 0) / 5))}",
        f"  Memory Usage:     {m.get('memory_percent', 0):6.1f}%  {'▓' * int(m.get('memory_percent', 0) / 5)}{'░' * (20 - int(m.get('memory_percent', 0) / 5))}",
        f"  Active Conns:     {m.get('active_connections', 0):6.0f}",
        "",
        "─── Latency ────────────────────────────────────",
        f"  p50:              {m.get('latency_p50_ms', 0):6.1f} ms",
        f"  p99:              {m.get('latency_p99_ms', 0):6.1f} ms",
        f"  {'⚠️  p99 exceeds 200ms SLO!' if m.get('latency_p99_ms', 0) > 200 else '✅  Within SLO (< 200ms)'}",
        "",
        "─── Traffic ────────────────────────────────────-",
        f"  Requests/sec:     {m.get('requests_per_sec', 0):6.1f}",
        f"  Error Rate:       {m.get('error_rate_percent', 0):6.2f}%",
        f"  {'🔴 ERROR RATE CRITICAL!' if m.get('error_rate_percent', 0) > 5 else '🟡 Elevated' if m.get('error_rate_percent', 0) > 1 else '✅  Normal'}",
        "",
    ]

    # Add deployment info if relevant
    if service.has_recent_deploy:
        lines.extend([
            "─── Recent Deployment ──────────────────────────",
            f"  Version:          {service.deploy_version}",
            f"  Deployed:         {service.deploy_minutes_ago} minutes ago",
            f"  Previous:         {service.previous_version}",
            f"  {'⚠️  RECENT DEPLOY — may be related to incident' if service.deploy_minutes_ago < 30 else ''}",
            "",
        ])

    # Add dependency info
    if service.dependencies:
        lines.extend([
            "─── Dependencies ───────────────────────────────",
            f"  Depends on: {', '.join(service.dependencies)}",
            "",
        ])

    return "\n".join(lines)


def get_metrics_dict(service: ServiceNode) -> Dict:
    """Return raw metrics as a dict (for structured responses)."""
    return {
        "service": service.name,
        "status": service.status.value,
        **service.current_metrics,
        "has_recent_deploy": service.has_recent_deploy,
        "deploy_version": service.deploy_version if service.has_recent_deploy else None,
    }
