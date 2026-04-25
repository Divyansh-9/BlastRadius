"""
Infrastructure simulation engine.

Models a service dependency graph as a pure Python state machine.
No actual containers or networking — just the INFORMATION an SRE would see.

Enhanced with:
- Temporal state evolution (failures spread over time)
- Information cost model (actions cost simulated minutes)
- Cascading damage propagation
- Fix ordering constraints
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class ServiceStatus(str, Enum):
    """Possible health states for a service."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"
    RESTARTING = "restarting"


@dataclass
class CascadeRule:
    """
    Defines how failures propagate between services over time.

    After `delay_minutes` of the source being unhealthy,
    the target transitions to `target_status`.
    """
    source: str
    target: str
    delay_minutes: int
    target_status: ServiceStatus = ServiceStatus.DEGRADED
    triggered: bool = False


@dataclass
class ServiceNode:
    """A single service in the infrastructure graph."""

    name: str
    display_name: str = ""
    status: ServiceStatus = ServiceStatus.HEALTHY
    dependencies: List[str] = field(default_factory=list)

    # Root cause metadata
    is_root_cause: bool = False
    failure_description: str = ""

    # Fix constraints
    fixable_by: List[str] = field(default_factory=list)
    fix_params: Dict = field(default_factory=dict)
    fix_order: int = 0  # Lower = must be fixed first

    # Deployment info
    has_recent_deploy: bool = False
    deploy_minutes_ago: int = 120
    deploy_version: str = "v2.3.1"
    previous_version: str = "v2.3.0"

    # Metrics
    port: int = 8080
    healthy_metrics: Dict = field(default_factory=lambda: {
        "cpu_percent": 15.0,
        "memory_percent": 35.0,
        "latency_p50_ms": 12.0,
        "latency_p99_ms": 45.0,
        "error_rate_percent": 0.1,
        "requests_per_sec": 250.0,
        "active_connections": 45,
    })
    current_metrics: Dict = field(default_factory=dict)

    # Log pattern key
    log_pattern: str = "normal"

    # Temporal tracking
    unhealthy_since_minute: int = -1  # -1 = currently healthy

    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.name.replace("-", " ").replace("_", " ").title()
        if not self.current_metrics:
            self.current_metrics = copy.deepcopy(self.healthy_metrics)


class ServiceGraph:
    """
    The full infrastructure graph — services + cascade rules.

    Key feature: temporal evolution. Call `tick(minutes)` to advance
    simulated time and propagate failures through cascade rules.
    """

    def __init__(
        self,
        services: List[ServiceNode],
        cascade_rules: Optional[List[CascadeRule]] = None,
    ):
        self._services: Dict[str, ServiceNode] = {s.name: s for s in services}
        self._cascade_rules: List[CascadeRule] = cascade_rules or []
        self._fix_history: List[Dict] = []
        self._time_minutes: int = 0
        self._damage_events: List[Dict] = []

        # Record initial unhealthy times
        for svc in self._services.values():
            if svc.status != ServiceStatus.HEALTHY:
                svc.unhealthy_since_minute = 0

    # ---------------------------------------------------------------
    # Snapshot Support (for GRPO offline evaluation)
    # ---------------------------------------------------------------

    def save_snapshot(self) -> Dict:
        """
        Serialize the full graph state into a plain dict.
        Used by GRPO to freeze the environment at a specific step,
        then restore it independently for each of G=4 completions.
        """
        return {
            "services": {
                name: {
                    "status": svc.status.value,
                    "current_metrics": copy.deepcopy(svc.current_metrics),
                    "unhealthy_since_minute": svc.unhealthy_since_minute,
                    "log_pattern": svc.log_pattern,
                    "has_recent_deploy": svc.has_recent_deploy,
                    "deploy_version": svc.deploy_version,
                    "previous_version": svc.previous_version,
                }
                for name, svc in self._services.items()
            },
            "cascade_rules": [
                {"source": r.source, "target": r.target, "triggered": r.triggered}
                for r in self._cascade_rules
            ],
            "time_minutes": self._time_minutes,
            "fix_history": copy.deepcopy(self._fix_history),
            "damage_events": copy.deepcopy(self._damage_events),
        }

    def restore_snapshot(self, snapshot: Dict):
        """
        Restore graph state from a snapshot dict.
        This must be called AFTER __init__ (i.e., the graph structure
        already exists from the scenario). We only restore mutable state.
        """
        for name, svc_state in snapshot.get("services", {}).items():
            svc = self._services.get(name)
            if svc is None:
                continue
            svc.status = ServiceStatus(svc_state["status"])
            svc.current_metrics = copy.deepcopy(svc_state["current_metrics"])
            svc.unhealthy_since_minute = svc_state["unhealthy_since_minute"]
            svc.log_pattern = svc_state["log_pattern"]
            svc.has_recent_deploy = svc_state["has_recent_deploy"]
            # Bug K: Restore deploy versions for replay fidelity
            svc.deploy_version = svc_state.get("deploy_version", svc.deploy_version)
            svc.previous_version = svc_state.get("previous_version", svc.previous_version)

        for i, rule_state in enumerate(snapshot.get("cascade_rules", [])):
            if i < len(self._cascade_rules):
                self._cascade_rules[i].triggered = rule_state["triggered"]

        self._time_minutes = snapshot.get("time_minutes", 0)
        self._fix_history = copy.deepcopy(snapshot.get("fix_history", []))
        self._damage_events = copy.deepcopy(snapshot.get("damage_events", []))

    # ---------------------------------------------------------------
    # Queries
    # ---------------------------------------------------------------

    def get_service(self, name: str) -> Optional[ServiceNode]:
        return self._services.get(name)

    def get_all_services(self) -> Dict[str, ServiceNode]:
        return dict(self._services)

    def get_status_summary(self) -> Dict[str, str]:
        return {n: s.status.value for n, s in self._services.items()}

    def get_active_alerts(self) -> List[str]:
        alerts = []
        for svc in self._services.values():
            if svc.status == ServiceStatus.DOWN:
                alerts.append(
                    f"🔴 CRITICAL [{svc.display_name}]: {svc.failure_description or 'Service unreachable'}"
                )
            elif svc.status == ServiceStatus.DEGRADED:
                alerts.append(
                    f"🟡 WARNING [{svc.display_name}]: Elevated error rate — "
                    f"{svc.current_metrics.get('error_rate_percent', 0):.1f}% errors, "
                    f"p99 latency {svc.current_metrics.get('latency_p99_ms', 0):.0f}ms"
                )
        return alerts

    def get_services_at_risk(self) -> List[str]:
        """Services that are HEALTHY but have unhealthy dependencies."""
        at_risk = []
        for svc in self._services.values():
            if svc.status == ServiceStatus.HEALTHY:
                for dep in svc.dependencies:
                    dep_svc = self._services.get(dep)
                    if dep_svc and dep_svc.status != ServiceStatus.HEALTHY:
                        at_risk.append(svc.name)
                        break
        return at_risk

    def get_dependency_map(self) -> Dict[str, List[str]]:
        return {n: list(s.dependencies) for n, s in self._services.items()}

    def get_dependency_text(self) -> str:
        """Human-readable dependency graph."""
        lines = ["=== Service Dependency Graph ===", ""]
        for name, svc in self._services.items():
            status_icon = {
                ServiceStatus.HEALTHY: "🟢",
                ServiceStatus.DEGRADED: "🟡",
                ServiceStatus.DOWN: "🔴",
                ServiceStatus.RESTARTING: "🔄",
            }.get(svc.status, "⚪")
            deps = ", ".join(svc.dependencies) if svc.dependencies else "none"
            lines.append(f"  {status_icon} {svc.display_name} ({svc.name})")
            lines.append(f"     └─ depends on: [{deps}]")
        return "\n".join(lines)

    def service_names(self) -> List[str]:
        return list(self._services.keys())

    @property
    def time_minutes(self) -> int:
        return self._time_minutes

    # ---------------------------------------------------------------
    # Temporal Evolution (THE KEY DIFFERENTIATOR)
    # ---------------------------------------------------------------

    def tick(self, minutes: int):
        """
        Advance simulated time by `minutes`.
        Evaluates cascade rules and propagates failures.
        Returns list of newly triggered cascades.
        """
        self._time_minutes += minutes
        newly_triggered = []

        for rule in self._cascade_rules:
            if rule.triggered:
                continue

            source = self._services.get(rule.source)
            if source is None or source.status == ServiceStatus.HEALTHY:
                continue

            # Check if enough time has passed since source went unhealthy
            if source.unhealthy_since_minute < 0:
                continue

            elapsed = self._time_minutes - source.unhealthy_since_minute
            if elapsed >= rule.delay_minutes:
                target = self._services.get(rule.target)
                if target and target.status == ServiceStatus.HEALTHY:
                    target.status = rule.target_status
                    target.unhealthy_since_minute = self._time_minutes
                    self._apply_degraded_metrics(target)
                    rule.triggered = True
                    newly_triggered.append({
                        "source": rule.source,
                        "target": rule.target,
                        "new_status": rule.target_status.value,
                        "at_minute": self._time_minutes,
                    })
                elif target and target.status == ServiceStatus.DEGRADED and rule.target_status == ServiceStatus.DOWN:
                    target.status = ServiceStatus.DOWN
                    self._apply_down_metrics(target)
                    rule.triggered = True
                    newly_triggered.append({
                        "source": rule.source,
                        "target": rule.target,
                        "new_status": ServiceStatus.DOWN.value,
                        "at_minute": self._time_minutes,
                    })

        self._damage_events.extend(newly_triggered)
        return newly_triggered

    def _apply_degraded_metrics(self, svc: ServiceNode):
        """Apply degraded-state metrics to a service."""
        svc.current_metrics = copy.deepcopy(svc.healthy_metrics)
        svc.current_metrics["cpu_percent"] = min(svc.healthy_metrics["cpu_percent"] * 2.5, 95.0)
        svc.current_metrics["memory_percent"] = min(svc.healthy_metrics["memory_percent"] * 1.8, 92.0)
        svc.current_metrics["latency_p50_ms"] = svc.healthy_metrics["latency_p50_ms"] * 4
        svc.current_metrics["latency_p99_ms"] = svc.healthy_metrics["latency_p99_ms"] * 8
        svc.current_metrics["error_rate_percent"] = min(svc.healthy_metrics["error_rate_percent"] * 50, 25.0)
        svc.current_metrics["requests_per_sec"] = svc.healthy_metrics["requests_per_sec"] * 0.6
        # Bug L: Signal connection pressure in degraded state
        svc.current_metrics["active_connections"] = min(
            int(svc.healthy_metrics.get("active_connections", 45) * 2.2), 100
        )

    def _apply_down_metrics(self, svc: ServiceNode):
        """Apply down-state metrics to a service."""
        svc.current_metrics = {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "latency_p50_ms": 0.0,
            "latency_p99_ms": 0.0,
            "error_rate_percent": 100.0,
            "requests_per_sec": 0.0,
            "active_connections": 0,
        }

    # ---------------------------------------------------------------
    # Fix Actions
    # ---------------------------------------------------------------

    def restart_service(self, name: str) -> Tuple[str, bool]:
        """
        Attempt to restart a service.
        Returns (result_text, success_bool).
        """
        svc = self._services.get(name)
        if svc is None:
            return f"ERROR: Unknown service '{name}'. Available: {', '.join(self.service_names())}", False

        if svc.status == ServiceStatus.HEALTHY:
            return f"{svc.display_name} is already healthy. No action needed.", False

        if "restart" in svc.fixable_by:
            ok, blocker = self._check_fix_order(svc)
            if not ok:
                self._apply_cascading_damage(name)
                return (
                    f"⚠️ FAILED: Restarting {svc.display_name} while '{blocker}' is still "
                    f"unhealthy caused a connection storm. Fix upstream dependencies first.\n"
                    f"COLLATERAL DAMAGE: Downstream services degraded further."
                ), False
            svc.status = ServiceStatus.HEALTHY
            svc.current_metrics = copy.deepcopy(svc.healthy_metrics)
            svc.unhealthy_since_minute = -1
            svc.log_pattern = "recovery"
            self._fix_history.append({"action": "restart", "target": name, "minute": self._time_minutes})
            self._auto_recover_dependents()
            return f"✅ {svc.display_name} restarted successfully. Service is now healthy.", True

        # Restart doesn't fix root cause
        if svc.is_root_cause:
            return (
                f"⚠️ {svc.display_name} restarted but crashed again within 30 seconds.\n"
                f"Status: still {svc.status.value}. The underlying issue persists.\n"
                f"Hint: A restart won't fix this — investigate the root cause."
            ), False

        # Cascade victim: check if all upstream dependencies are now healthy
        # If they are, the service can self-recover (root cause cleared)
        all_deps_healthy = all(
            self._services.get(dep, ServiceNode(name=dep, status=ServiceStatus.DOWN)).status == ServiceStatus.HEALTHY
            for dep in svc.dependencies
        )
        if all_deps_healthy and svc.dependencies:
            svc.status = ServiceStatus.HEALTHY
            svc.current_metrics = copy.deepcopy(svc.healthy_metrics)
            svc.unhealthy_since_minute = -1
            svc.log_pattern = "recovery"
            self._fix_history.append({"action": "restart", "target": name, "minute": self._time_minutes})
            self._auto_recover_dependents()
            return (
                f"✅ {svc.display_name} restarted successfully.\n"
                f"All upstream dependencies are now healthy — service recovered."
            ), True

        return (
            f"⚠️ {svc.display_name} restarted but returned to {svc.status.value} "
            f"after 45 seconds. This service depends on unhealthy upstream services.\n"
            f"Treating symptoms won't help — find the root cause."
        ), False

    def rollback_deploy(self, name: str) -> Tuple[str, bool]:
        """Attempt to roll back the last deployment."""
        svc = self._services.get(name)
        if svc is None:
            return f"ERROR: Unknown service '{name}'.", False

        if svc.status == ServiceStatus.HEALTHY:
            return (
                f"{svc.display_name} is already healthy. "
                f"No rollback needed."
            ), False

        if not svc.has_recent_deploy:
            return (
                f"No recent deployment found for {svc.display_name}.\n"
                f"Last deploy: {svc.deploy_minutes_ago} minutes ago ({svc.deploy_version}).\n"
                f"No rollback available — try a different approach."
            ), False

        if "rollback" in svc.fixable_by:
            ok, blocker = self._check_fix_order(svc)
            if not ok:
                self._apply_cascading_damage(name)
                return (
                    f"⚠️ FAILED: Rolling back {svc.display_name} while '{blocker}' "
                    f"is unhealthy caused cascading errors."
                ), False
            svc.status = ServiceStatus.HEALTHY
            svc.current_metrics = copy.deepcopy(svc.healthy_metrics)
            svc.unhealthy_since_minute = -1
            svc.has_recent_deploy = False
            svc.log_pattern = "rollback_success"
            self._fix_history.append({"action": "rollback", "target": name, "minute": self._time_minutes})
            self._auto_recover_dependents()
            return (
                f"✅ Deployment rolled back on {svc.display_name}.\n"
                f"Reverted: {svc.deploy_version} → {svc.previous_version}\n"
                f"Service recovered and healthy."
            ), True

        if svc.has_recent_deploy:
            return (
                f"Deployment on {svc.display_name} rolled back "
                f"({svc.deploy_version} → {svc.previous_version}), "
                f"but service remains {svc.status.value}.\n"
                f"The recent deploy was NOT the cause of this failure."
            ), False

        return f"Rollback had no effect on {svc.display_name}.", False

    def scale_service(self, name: str, params: Dict) -> Tuple[str, bool]:
        """Attempt to scale service resources."""
        svc = self._services.get(name)
        if svc is None:
            return f"ERROR: Unknown service '{name}'.", False

        if svc.status == ServiceStatus.HEALTHY:
            return (
                f"{svc.display_name} is already healthy and scaled. "
                f"No further action needed."
            ), False

        if "scale" in svc.fixable_by:
            ok, blocker = self._check_fix_order(svc)
            if not ok:
                self._apply_cascading_damage(name)
                return (
                    f"⚠️ FAILED: Scaling {svc.display_name} while '{blocker}' "
                    f"is unhealthy — resources allocated but service still failing."
                ), False
            svc.status = ServiceStatus.HEALTHY
            svc.current_metrics = copy.deepcopy(svc.healthy_metrics)
            svc.unhealthy_since_minute = -1
            svc.log_pattern = "scale_success"
            self._fix_history.append({"action": "scale", "target": name, "params": params, "minute": self._time_minutes})
            param_str = ", ".join(f"{k}={v}" for k, v in params.items()) if params else "auto"
            self._auto_recover_dependents()
            return (
                f"✅ {svc.display_name} scaled successfully.\n"
                f"Resources adjusted: {param_str}\n"
                f"Service is now healthy."
            ), True

        return (
            f"Scaled {svc.display_name} resources, but service remains "
            f"{svc.status.value}. Scaling is not the correct fix for this issue."
        ), False

    # ---------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------

    def _check_fix_order(self, svc: ServiceNode) -> Tuple[bool, Optional[str]]:
        """Check if prerequisite services (lower fix_order) are already fixed."""
        if svc.fix_order <= 0:
            return True, None
        for other in self._services.values():
            if (
                other.name != svc.name
                and other.fix_order > 0
                and other.fix_order < svc.fix_order
                and other.status != ServiceStatus.HEALTHY
            ):
                return False, other.name
        return True, None

    def _auto_recover_dependents(self):
        """
        After a successful fix, scan all cascade-victim services (no fixable_by)
        and auto-recover them if ALL their dependencies are now healthy.
        This models real-world self-healing: once the upstream root cause is cleared,
        downstream victim services recover on their own.
        """
        changed = True
        while changed:  # iterate until no more services recover (handles chains)
            changed = False
            for svc in self._services.values():
                if svc.status == ServiceStatus.HEALTHY:
                    continue
                if svc.fixable_by:  # Already handled by explicit fix actions
                    continue
                if not svc.dependencies:
                    continue
                all_deps_healthy = all(
                    self._services.get(dep, ServiceNode(name=dep, status=ServiceStatus.DOWN)).status
                    == ServiceStatus.HEALTHY
                    for dep in svc.dependencies
                )
                if all_deps_healthy:
                    svc.status = ServiceStatus.HEALTHY
                    svc.current_metrics = copy.deepcopy(svc.healthy_metrics)
                    svc.unhealthy_since_minute = -1
                    svc.log_pattern = "auto_recovery"
                    self._fix_history.append({
                        "action": "auto_recovery",
                        "target": svc.name,
                        "minute": self._time_minutes,
                    })
                    # Bug G: Re-arm cascade rules targeting this service
                    for rule in self._cascade_rules:
                        if rule.target == svc.name and rule.triggered:
                            rule.triggered = False
                    changed = True

    def _apply_cascading_damage(self, source_name: str):
        """When a fix fails due to ordering, propagate damage to dependents."""
        for svc in self._services.values():
            if source_name in svc.dependencies:
                if svc.status == ServiceStatus.HEALTHY:
                    svc.status = ServiceStatus.DEGRADED
                    self._apply_degraded_metrics(svc)
                    svc.unhealthy_since_minute = self._time_minutes
                elif svc.status == ServiceStatus.DEGRADED:
                    svc.status = ServiceStatus.DOWN
                    self._apply_down_metrics(svc)
                self._damage_events.append({
                    "type": "collateral_damage",
                    "source": source_name,
                    "target": svc.name,
                    "new_status": svc.status.value,
                    "at_minute": self._time_minutes,
                })

    def is_fully_resolved(self) -> bool:
        return all(s.status == ServiceStatus.HEALTHY for s in self._services.values())

    EXPLICIT_FIX_ACTIONS = {"restart", "rollback", "scale"}

    def get_resolved_services(self) -> List[str]:
        return [
            e["target"] for e in self._fix_history
            if e.get("action") in self.EXPLICIT_FIX_ACTIONS
        ]

    def count_collateral_damage(self) -> int:
        return sum(1 for e in self._damage_events if e.get("type") == "collateral_damage")

    def get_incident_severity(self) -> str:
        """P1 = any service DOWN, P2 = any DEGRADED, P3 = all healthy."""
        statuses = [s.status for s in self._services.values()]
        if ServiceStatus.DOWN in statuses:
            return "P1"
        if ServiceStatus.DEGRADED in statuses:
            return "P2"
        return "P3"
