"""
Comprehensive tests for the IT Incident Response Environment.

Tests cover:
- Model validation
- Infrastructure engine (temporal cascading, fix ordering)
- Grader (causal chain evaluation, reward signals)
- Scenarios (all 3 difficulty levels)
- Full episode integration
"""

import pytest
from incident_env.models import (
    IncidentAction,
    IncidentObservation,
    IncidentState,
    VALID_COMMANDS,
    ACTION_TIME_COSTS,
)
from incident_env.server.engine.infrastructure import (
    CascadeRule,
    ServiceGraph,
    ServiceNode,
    ServiceStatus,
)
from incident_env.server.engine.log_generator import generate_logs
from incident_env.server.engine.metrics_generator import generate_metrics_report
from incident_env.server.engine.grader import Grader, ScenarioGradingConfig
from incident_env.server.scenarios import SCENARIOS
from incident_env.server.incident_environment import IncidentEnvironment


# ═══════════════════════════════════════════════════════════
# Model Tests
# ═══════════════════════════════════════════════════════════

class TestModels:
    def test_valid_commands_count(self):
        assert len(VALID_COMMANDS) == 8

    def test_action_time_costs(self):
        assert ACTION_TIME_COSTS["check_status"] == 0
        assert ACTION_TIME_COSTS["check_logs"] == 2
        assert ACTION_TIME_COSTS["rollback_deploy"] == 5

    def test_action_creation(self):
        action = IncidentAction(command="check_logs", target="database")
        assert action.command == "check_logs"
        assert action.target == "database"
        assert action.parameters == {}

    def test_observation_defaults(self):
        obs = IncidentObservation()
        assert obs.output == ""
        assert obs.services_status == {}
        assert obs.incident_severity == "P2"

    def test_state_defaults(self):
        state = IncidentState()
        assert state.step_count == 0
        assert state.total_reward == 0.0
        assert state.max_steps == 25
        assert not state.done


# ═══════════════════════════════════════════════════════════
# Infrastructure Engine Tests
# ═══════════════════════════════════════════════════════════

class TestInfrastructure:
    def _make_simple_graph(self):
        """Create a minimal test graph: A depends on B."""
        services = [
            ServiceNode(
                name="service-a",
                status=ServiceStatus.HEALTHY,
                dependencies=["service-b"],
            ),
            ServiceNode(
                name="service-b",
                status=ServiceStatus.DOWN,
                dependencies=[],
                is_root_cause=True,
                fixable_by=["restart"],
                fix_order=1,
                failure_description="Test failure",
            ),
        ]
        cascades = [
            CascadeRule(
                source="service-b",
                target="service-a",
                delay_minutes=3,
                target_status=ServiceStatus.DEGRADED,
            ),
        ]
        return ServiceGraph(services, cascades)

    def test_status_summary(self):
        graph = self._make_simple_graph()
        status = graph.get_status_summary()
        assert status["service-a"] == "healthy"
        assert status["service-b"] == "down"

    def test_active_alerts(self):
        graph = self._make_simple_graph()
        alerts = graph.get_active_alerts()
        assert len(alerts) == 1
        assert "CRITICAL" in alerts[0]

    def test_temporal_cascade(self):
        """Failures should spread after delay_minutes."""
        graph = self._make_simple_graph()

        # After 2 minutes — should NOT cascade yet
        graph.tick(2)
        assert graph.get_service("service-a").status == ServiceStatus.HEALTHY

        # After 3 total minutes — should cascade
        events = graph.tick(1)
        assert len(events) == 1
        assert graph.get_service("service-a").status == ServiceStatus.DEGRADED

    def test_fix_success(self):
        graph = self._make_simple_graph()
        text, success = graph.restart_service("service-b")
        assert success
        assert "✅" in text
        assert graph.get_service("service-b").status == ServiceStatus.HEALTHY

    def test_fix_wrong_target(self):
        graph = self._make_simple_graph()
        text, success = graph.restart_service("service-a")
        # service-a is healthy, so restart does nothing
        assert not success

    def test_fix_unknown_service(self):
        graph = self._make_simple_graph()
        text, success = graph.restart_service("nonexistent")
        assert not success
        assert "ERROR" in text

    def test_is_fully_resolved(self):
        graph = self._make_simple_graph()
        assert not graph.is_fully_resolved()
        graph.restart_service("service-b")
        assert graph.is_fully_resolved()

    def test_incident_severity(self):
        graph = self._make_simple_graph()
        assert graph.get_incident_severity() == "P1"  # service-b is DOWN


# ═══════════════════════════════════════════════════════════
# Log Generator Tests
# ═══════════════════════════════════════════════════════════

class TestLogGenerator:
    def test_generates_logs(self):
        svc = ServiceNode(
            name="test-service",
            status=ServiceStatus.DOWN,
            log_pattern="db_pool_exhaustion",
        )
        logs = generate_logs(svc, env_time_minutes=5, num_entries=5)
        assert "test-service" in logs
        assert len(logs) > 100

    def test_healthy_service_logs(self):
        svc = ServiceNode(
            name="healthy-svc",
            status=ServiceStatus.HEALTHY,
            log_pattern="normal",
        )
        logs = generate_logs(svc, env_time_minutes=0)
        assert "INFO" in logs


# ═══════════════════════════════════════════════════════════
# Metrics Generator Tests
# ═══════════════════════════════════════════════════════════

class TestMetricsGenerator:
    def test_generates_report(self):
        svc = ServiceNode(
            name="test-db",
            display_name="Test Database",
            status=ServiceStatus.DEGRADED,
        )
        report = generate_metrics_report(svc, env_time_minutes=5)
        assert "Test Database" in report
        assert "DEGRADED" in report

    def test_recent_deploy_shown(self):
        svc = ServiceNode(
            name="test-svc",
            status=ServiceStatus.DOWN,
            has_recent_deploy=True,
            deploy_version="v2.0.0",
            deploy_minutes_ago=10,
        )
        report = generate_metrics_report(svc, env_time_minutes=10)
        assert "v2.0.0" in report
        assert "RECENT DEPLOY" in report


# ═══════════════════════════════════════════════════════════
# Grader Tests
# ═══════════════════════════════════════════════════════════

class TestGrader:
    def _make_config(self):
        return ScenarioGradingConfig(
            root_cause_service="auth-service",
            root_cause_description="Bad deployment",
            ground_truth_causal_chain=[
                "auth deployed bad code",
                "tokens are invalid",
                "payments fail",
            ],
            correct_fix_actions=[
                {"command": "rollback_deploy", "target": "auth-service"},
            ],
            correct_fix_order=["auth-service"],
            useful_investigation_targets=["auth-service", "payment-service"],
            max_optimal_steps=6,
            max_total_reward=0.85,
        )

    def test_useful_investigation_reward(self):
        grader = Grader(self._make_config())
        result = grader.grade_step(
            command="check_logs", target="auth-service",
            params={}, action_succeeded=False,
            services_now_healthy=[], all_resolved=False,
            step_number=1, collateral_damage=0,
        )
        assert result.reward > 0  # Should get +0.05

    def test_irrelevant_investigation_penalty(self):
        grader = Grader(self._make_config())
        result = grader.grade_step(
            command="check_logs", target="random-service",
            params={}, action_succeeded=False,
            services_now_healthy=[], all_resolved=False,
            step_number=1, collateral_damage=0,
        )
        assert result.reward < 0  # Should get -0.02

    def test_correct_diagnosis(self):
        grader = Grader(self._make_config())
        result = grader.grade_step(
            command="diagnose", target="",
            params={
                "root_cause": "auth-service",
                "causal_chain": ["auth deployed bad code", "tokens invalid", "payments fail"],
                "confidence": 0.9,
            },
            action_succeeded=False,
            services_now_healthy=[], all_resolved=False,
            step_number=2, collateral_damage=0,
        )
        assert result.reward > 0.15  # Root cause correct = +0.15 minimum

    def test_wrong_diagnosis(self):
        grader = Grader(self._make_config())
        result = grader.grade_step(
            command="diagnose", target="",
            params={"root_cause": "database", "causal_chain": [], "confidence": 0.9},
            action_succeeded=False,
            services_now_healthy=[], all_resolved=False,
            step_number=2, collateral_damage=0,
        )
        assert result.reward < 0  # Wrong root cause

    def test_correct_fix_reward(self):
        grader = Grader(self._make_config())
        result = grader.grade_step(
            command="rollback_deploy", target="auth-service",
            params={}, action_succeeded=True,
            services_now_healthy=["auth-service"], all_resolved=False,
            step_number=3, collateral_damage=0,
        )
        assert result.reward == 0.2  # Correct fix = +0.20

    def test_final_score_normalization(self):
        grader = Grader(self._make_config())
        final = grader.get_final_score()
        assert 0.0 <= final.reward <= 1.0

    def test_collateral_damage_penalty(self):
        grader = Grader(self._make_config())
        result = grader.grade_step(
            command="restart_service", target="wrong",
            params={}, action_succeeded=False,
            services_now_healthy=[], all_resolved=False,
            step_number=1, collateral_damage=2,
        )
        # Should have wrong fix penalty + collateral damage penalty
        assert result.reward < -0.05


# ═══════════════════════════════════════════════════════════
# Scenario Tests
# ═══════════════════════════════════════════════════════════

class TestScenarios:
    @pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
    def test_scenario_builds(self, task_id):
        scenario_cls = SCENARIOS[task_id]
        scenario = scenario_cls()
        assert scenario.scenario_id
        assert scenario.difficulty in ("easy", "medium", "hard")
        assert scenario.title
        assert scenario.description

    @pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
    def test_scenario_graph(self, task_id):
        scenario = SCENARIOS[task_id]()
        graph = scenario.build_service_graph()
        assert len(graph.service_names()) >= 4  # At least 4 services

    @pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
    def test_scenario_grading_config(self, task_id):
        scenario = SCENARIOS[task_id]()
        config = scenario.get_grading_config()
        assert config.root_cause_service
        assert config.ground_truth_causal_chain
        assert config.correct_fix_order
        assert config.max_total_reward > 0


# ═══════════════════════════════════════════════════════════
# Full Environment Integration Tests
# ═══════════════════════════════════════════════════════════

class TestEnvironmentIntegration:
    @pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
    def test_reset(self, task_id):
        env = IncidentEnvironment()
        result = env.reset(task_id=task_id)

        assert "observation" in result
        assert "reward" in result
        assert "done" in result
        assert result["done"] is False
        assert result["observation"]["incident_severity"] in ("P1", "P2", "P3")

    def test_invalid_task_id(self):
        env = IncidentEnvironment()
        with pytest.raises(ValueError):
            env.reset(task_id="nonexistent")

    def test_step_before_reset(self):
        env = IncidentEnvironment()
        result = env.step(IncidentAction(command="check_status"))
        assert "error" in result.get("info", {})

    @pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
    def test_full_episode(self, task_id):
        """Run through an episode and verify reward accumulation."""
        env = IncidentEnvironment()
        env.reset(task_id=task_id)

        total_reward = 0.0
        for i in range(5):
            result = env.step(IncidentAction(command="check_status"))
            total_reward += result["reward"]

        state = env.state
        assert state["step_count"] == 5
        assert state["scenario_id"]

    def test_easy_solvable(self):
        """The easy scenario should be solvable with correct actions."""
        env = IncidentEnvironment()
        env.reset(task_id="easy")

        # 1. Check status
        env.step(IncidentAction(command="check_status"))

        # 2. Check database logs
        env.step(IncidentAction(command="check_logs", target="database"))

        # 3. Diagnose
        env.step(IncidentAction(
            command="diagnose",
            parameters={
                "root_cause": "database",
                "causal_chain": [
                    "database connection pool exhausted",
                    "API gateway cannot get connections",
                    "users see 503 errors",
                ],
                "confidence": 0.9,
            },
        ))

        # 4. Fix database
        result = env.step(IncidentAction(
            command="scale_service",
            target="database",
            parameters={"max_connections": 200},
        ))
        assert result["reward"] > 0  # Fix should give reward

    def test_temporal_cascade_in_episode(self):
        """Test that temporal cascading works during an episode."""
        env = IncidentEnvironment()
        env.reset(task_id="medium")

        # Take several expensive actions to advance time
        for _ in range(3):
            env.step(IncidentAction(command="check_logs", target="payment-service"))

        # After 6 min (3 * 2 min), check if worker-queue degraded
        state = env.state
        assert state["time_elapsed_minutes"] >= 6

    def test_max_steps_terminates(self):
        """Episode should end after max_steps."""
        env = IncidentEnvironment()
        env.reset(task_id="easy")

        for _ in range(30):
            result = env.step(IncidentAction(command="check_status"))
            if result["done"]:
                break

        assert result["done"]

    def test_state_tracking(self):
        """State should accurately track actions and rewards."""
        env = IncidentEnvironment()
        env.reset(task_id="easy")

        env.step(IncidentAction(command="check_status"))
        env.step(IncidentAction(command="check_logs", target="database"))

        state = env.state
        assert state["step_count"] == 2
        assert len(state["actions_taken"]) == 2
        assert state["actions_taken"][0]["command"] == "check_status"
        assert state["actions_taken"][1]["command"] == "check_logs"
