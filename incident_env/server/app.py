"""
FastAPI server for the IT Incident Response Environment.

Exposes the OpenEnv HTTP API:
- POST /reset     → Initialize a new episode
- POST /step      → Execute an action
- GET  /state     → Get current episode state
- GET  /health    → Health check
- GET  /info      → Environment metadata
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from incident_env.server.incident_environment import IncidentEnvironment


# ---------------------------------------------------------------------------
# Pydantic request/response models for the HTTP API
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = Field(default="easy", description="Task difficulty: easy | medium | hard")
    eval_mode: bool = Field(default=False, description="Enable strict anti-cheat evaluation mode")


class ActionRequest(BaseModel):
    command: str = Field(..., description="Command to execute")
    target: str = Field(default="", description="Target service name")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")


class ObservationResponse(BaseModel):
    output: str = ""
    services_status: Dict[str, str] = {}
    active_alerts: List[str] = []
    time_elapsed_minutes: int = 0
    incident_severity: str = "P2"
    services_at_risk: List[str] = []
    hint: str = ""


class StepResponse(BaseModel):
    observation: ObservationResponse
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = {}


class StateResponse(BaseModel):
    episode_id: str = ""
    step_count: int = 0
    scenario_id: str = ""
    task_difficulty: str = ""
    services_resolved: List[str] = []
    root_cause_identified: bool = False
    total_reward: float = 0.0
    is_resolved: bool = False
    done: bool = False
    time_elapsed_minutes: int = 0


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="IT Incident Response Environment",
    description=(
        "An OpenEnv-compliant RL environment simulating production incident response. "
        "Agents diagnose cascading infrastructure failures, identify root causes, "
        "and apply fixes in the correct order while failures spread in real-time."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single environment instance (stateful per-episode)
env = IncidentEnvironment()


# ---------------------------------------------------------------------------
# Landing Page
# ---------------------------------------------------------------------------

LANDING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>IT Incident Response Environment</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',sans-serif;background:#0a0e17;color:#e2e8f0;min-height:100vh;overflow-x:hidden}
.bg-grid{position:fixed;inset:0;background-image:linear-gradient(rgba(99,102,241,.05) 1px,transparent 1px),linear-gradient(90deg,rgba(99,102,241,.05) 1px,transparent 1px);background-size:60px 60px;pointer-events:none;z-index:0}
.container{max-width:1000px;margin:0 auto;padding:40px 24px;position:relative;z-index:1}
.hero{text-align:center;padding:48px 0 40px}
.badge{display:inline-flex;align-items:center;gap:6px;background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.3);color:#f87171;font-size:12px;font-weight:600;padding:6px 14px;border-radius:20px;letter-spacing:.5px;text-transform:uppercase;margin-bottom:20px}
.badge .dot{width:7px;height:7px;background:#ef4444;border-radius:50%;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
h1{font-size:42px;font-weight:800;background:linear-gradient(135deg,#f8fafc,#94a3b8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.15;margin-bottom:14px}
.subtitle{font-size:17px;color:#94a3b8;max-width:640px;margin:0 auto;line-height:1.6}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:36px 0}
.card{background:rgba(15,23,42,.7);border:1px solid rgba(99,102,241,.15);border-radius:14px;padding:24px;transition:all .25s}
.card:hover{border-color:rgba(99,102,241,.4);transform:translateY(-2px);box-shadow:0 8px 30px rgba(99,102,241,.1)}
.card-diff{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.8px;margin-bottom:10px;display:flex;align-items:center;gap:6px}
.card-diff.easy{color:#34d399}
.card-diff.medium{color:#fbbf24}
.card-diff.hard{color:#f87171}
.card h3{font-size:16px;font-weight:700;color:#f1f5f9;margin-bottom:8px}
.card p{font-size:13px;color:#64748b;line-height:1.5}
.score{font-family:'JetBrains Mono',monospace;font-size:22px;font-weight:700;margin-top:12px}
.score.easy{color:#34d399}
.score.medium{color:#fbbf24}
.score.hard{color:#f87171}
.section{margin:36px 0}
.section-title{font-size:14px;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:#6366f1;margin-bottom:16px;display:flex;align-items:center;gap:8px}
.endpoints{display:grid;gap:8px}
.ep{display:flex;align-items:center;gap:12px;background:rgba(15,23,42,.6);border:1px solid rgba(99,102,241,.1);border-radius:10px;padding:12px 16px;transition:border-color .2s}
.ep:hover{border-color:rgba(99,102,241,.3)}
.method{font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:600;padding:3px 8px;border-radius:4px;min-width:50px;text-align:center}
.method.get{background:rgba(52,211,153,.15);color:#34d399}
.method.post{background:rgba(99,102,241,.15);color:#818cf8}
.path{font-family:'JetBrains Mono',monospace;font-size:14px;color:#e2e8f0;flex:1}
.desc{font-size:12px;color:#64748b}
.features{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:16px}
.feat{background:rgba(15,23,42,.5);border:1px solid rgba(99,102,241,.08);border-radius:10px;padding:18px;text-align:center}
.feat-icon{font-size:28px;margin-bottom:8px}
.feat-label{font-size:13px;font-weight:600;color:#cbd5e1}
.feat-desc{font-size:11px;color:#64748b;margin-top:4px}
.footer{text-align:center;margin-top:48px;padding-top:24px;border-top:1px solid rgba(99,102,241,.1);color:#475569;font-size:12px}
.footer a{color:#6366f1;text-decoration:none}
@media(max-width:700px){.cards,.features{grid-template-columns:1fr}h1{font-size:28px}}
</style>
</head>
<body>
<div class="bg-grid"></div>
<div class="container">
  <div class="hero">
    <div class="badge"><span class="dot"></span> OpenEnv Compatible</div>
    <h1>IT Incident Response<br>Environment</h1>
    <p class="subtitle">An RL environment that simulates production infrastructure failures.
    Agents diagnose cascading outages, identify root causes via causal reasoning,
    and apply fixes under time pressure as failures spread.</p>
  </div>

  <div class="cards">
    <div class="card">
      <div class="card-diff easy">● Easy</div>
      <h3>DB Pool Exhaustion</h3>
      <p>Connection pool maxed out. API gateway returning 503s. Clear diagnostic signals.</p>
      <div class="score easy">0.70</div>
    </div>
    <div class="card">
      <div class="card-diff medium">● Medium</div>
      <h3>Bad Deployment Cascade</h3>
      <p>Broken JWT deploy on auth service. Payment service logs are a red herring.</p>
      <div class="score medium">0.75</div>
    </div>
    <div class="card">
      <div class="card-diff hard">● Hard</div>
      <h3>Thundering Herd</h3>
      <p>CDN cache miss storm. Misleading signals. Fix order is critical.</p>
      <div class="score hard">0.15</div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">⚡ Key Features</div>
    <div class="features">
      <div class="feat"><div class="feat-icon">🕐</div><div class="feat-label">Temporal Cascading</div><div class="feat-desc">Failures spread while you act</div></div>
      <div class="feat"><div class="feat-icon">🧠</div><div class="feat-label">Causal Chain Grading</div><div class="feat-desc">Agent must explain WHY</div></div>
      <div class="feat"><div class="feat-icon">💰</div><div class="feat-label">Information Cost</div><div class="feat-desc">Each action costs time</div></div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">🔌 API Endpoints</div>
    <div class="endpoints">
      <a href="/health" class="ep" style="text-decoration:none"><span class="method get">GET</span><span class="path">/health</span><span class="desc">Health check</span></a>
      <a href="/info" class="ep" style="text-decoration:none"><span class="method get">GET</span><span class="path">/info</span><span class="desc">Environment metadata</span></a>
      <a href="/tasks" class="ep" style="text-decoration:none"><span class="method get">GET</span><span class="path">/tasks</span><span class="desc">List available scenarios</span></a>
      <a href="/docs" class="ep" style="text-decoration:none"><span class="method get">GET</span><span class="path">/docs</span><span class="desc">Interactive API docs (Swagger)</span></a>
      <div class="ep"><span class="method post">POST</span><span class="path">/reset</span><span class="desc">Initialize new incident episode</span></div>
      <div class="ep"><span class="method post">POST</span><span class="path">/step</span><span class="desc">Execute agent action</span></div>
      <a href="/state" class="ep" style="text-decoration:none"><span class="method get">GET</span><span class="path">/state</span><span class="desc">Current episode state</span></a>
    </div>
  </div>

  <div class="footer">
    Meta PyTorch OpenEnv Hackathon &middot; Powered by FastAPI &middot; <a href="/docs">Swagger Docs</a>
  </div>
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api", response_class=HTMLResponse)
def landing():
    """API overview page."""
    return LANDING_HTML


@app.get("/analysis", response_class=HTMLResponse)
def analysis_page():
    """Post-incident analysis UI."""
    from incident_env.server.analysis_page import ANALYSIS_HTML
    return ANALYSIS_HTML


@app.get("/analysis-data")
def analysis_data():
    """Returns the internal grader and scenario details from the last episode."""
    if not env._scenario:
        return {"error": "No episode run yet."}, 400
        
    final_score = env._grader.get_final_score()
    optimal_config = env._scenario.get_grading_config()
    
    return {
        "scenario": {
            "id": env._scenario.scenario_id,
            "title": env._scenario.title,
            "description": env._scenario.description,
            "difficulty": env._scenario.difficulty,
        },
        "state": env.state,
        "optimal": {
            "root_cause_service": optimal_config.root_cause_service,
            "root_cause_description": optimal_config.root_cause_description,
            "correct_fix_actions": optimal_config.correct_fix_actions,
            "ground_truth_causal_chain": optimal_config.ground_truth_causal_chain,
        },
        "final_score": {
            "reward": final_score.reward,
            "breakdown": final_score.breakdown,
        }
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "environment": "incident-response-env", "version": "1.0.0"}


@app.get("/info")
def info():
    """Environment metadata."""
    return {
        "name": "incident-response-env",
        "description": "IT Incident Response Simulator for SRE/DevOps agents",
        "version": "1.0.0",
        "tasks": ["easy", "medium", "hard"],
        "action_space": {
            "type": "dict",
            "commands": [
                "check_status", "check_logs", "check_metrics",
                "check_dependencies", "diagnose",
                "restart_service", "rollback_deploy", "scale_service",
            ],
        },
        "observation_space": {
            "type": "dict",
            "fields": [
                "output", "services_status", "active_alerts",
                "time_elapsed_minutes", "incident_severity",
                "services_at_risk", "hint",
            ],
        },
    }


@app.post("/reset", response_model=StepResponse)
def reset(request: ResetRequest):
    """
    Initialize a new incident episode.

    Parameters:
    - task_id: "easy" | "medium" | "hard"
    - eval_mode: boolean toggle for anti-cheat
    """
    from incident_env.models import IncidentAction
    result = env.reset(task_id=request.task_id, eval_mode=request.eval_mode)
    return StepResponse(
        observation=ObservationResponse(**result["observation"]),
        reward=result["reward"],
        done=result["done"],
        info=result.get("info", {}),
    )


@app.post("/step", response_model=StepResponse)
def step(request: ActionRequest):
    """
    Execute an action in the environment.

    The agent sends a command (e.g., check_logs, restart_service)
    and receives the updated observation, reward, and done flag.
    """
    from incident_env.models import IncidentAction
    action = IncidentAction(
        command=request.command,
        target=request.target,
        parameters=request.parameters,
    )
    result = env.step(action)
    return StepResponse(
        observation=ObservationResponse(**result["observation"]),
        reward=result["reward"],
        done=result["done"],
        info=result.get("info", {}),
    )


@app.get("/state")
def state():
    """Get current episode state."""
    return env.state


@app.get("/tasks")
def tasks():
    """List available tasks with descriptions."""
    return {
        "tasks": [
            {
                "id": "easy",
                "title": "Database Connection Pool Exhaustion",
                "difficulty": "easy",
                "description": "Single service failure with clear logs. Straightforward fix.",
                "expected_score": "0.7-0.9",
            },
            {
                "id": "medium",
                "title": "Bad Deployment Cascade",
                "difficulty": "medium",
                "description": "Root cause analysis required. Red herring in victim service logs.",
                "expected_score": "0.4-0.6",
            },
            {
                "id": "hard",
                "title": "Thundering Herd After CDN Cache Invalidation",
                "difficulty": "hard",
                "description": "Multi-service cascade with misleading signals. Fix order critical.",
                "expected_score": "0.1-0.3",
            },
        ]
    }
