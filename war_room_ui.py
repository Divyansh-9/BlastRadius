import os
import gradio as gr # type: ignore
import plotly.graph_objects as go # type: ignore
import uvicorn # type: ignore

from incident_env.server.app import app as fast_app
from agent.orchestrator import MATPOOrchestrator

# ---------------------------------------------------------------------------
# Plotly Graph Generation
# ---------------------------------------------------------------------------
def generate_system_graph(observation: dict):
    """
    Generates a stunning dark-mode network graph of the system state.
    """
    services = observation.get("services_status", {})
    if not services:
        # Empty placeholder
        services = {"auth-service": "HEALTHY", "db-primary": "HEALTHY", "redis-cache": "HEALTHY"}
        
    nodes = list(services.keys())
    statuses = list(services.values())
    
    # Map statuses to colors
    color_map = {
        "HEALTHY": "#10b981",    # Emerald green
        "DEGRADED": "#f59e0b",   # Amber
        "DOWN": "#ef4444",       # Red
        "RESTARTING": "#3b82f6"  # Blue
    }
    node_colors = [color_map.get(str(s).upper(), "#6b7280") for s in statuses]
    
    # We will arrange them in a circle for visual flair
    import math
    num_nodes = len(nodes)
    x_coords = []
    y_coords = []
    for i in range(num_nodes):
        angle = 2 * math.pi * i / num_nodes
        x_coords.append(math.cos(angle))
        y_coords.append(math.sin(angle))
        
    # Create the Plotly figure
    fig = go.Figure()
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=x_coords, y=y_coords,
        mode='markers+text',
        marker=dict(
            size=50,
            color=node_colors,
            line=dict(width=2, color='white'),
            symbol='hexagon'
        ),
        text=nodes,
        textposition="top center",
        textfont=dict(color='white', size=14, family="Courier New"),
        hoverinfo='text',
        hovertext=[f"{n}: {s}" for n, s in zip(nodes, statuses)]
    ))
    
    # Add subtle central core
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        marker=dict(size=20, color='#374151', symbol='circle'),
        hoverinfo='none',
        showlegend=False
    ))
    
    # Draw faint links from core to nodes
    for i in range(num_nodes):
        fig.add_trace(go.Scatter(
            x=[0, x_coords[i]], y=[0, y_coords[i]],
            mode='lines',
            line=dict(color='#4b5563', width=1, dash='dot'),
            hoverinfo='none',
            showlegend=False
        ))
        
    fig.update_layout(
        title="Live Infrastructure Topology",
        title_font=dict(color='white', size=20, family="Courier New"),
        paper_bgcolor='#111827',  # Tailwind gray-900
        plot_bgcolor='#111827',
        showlegend=False,
        margin=dict(l=40, r=40, b=40, t=60),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig

# ---------------------------------------------------------------------------
# UI Construction
# ---------------------------------------------------------------------------

custom_css = """
body { background-color: #030712 !important; color: #f9fafb !important; }
.gradio-container { max-width: 1600px !important; }
.terminal-window { 
    background-color: #000000; 
    border: 1px solid #333; 
    border-radius: 8px; 
    padding: 15px; 
    font-family: 'Consolas', 'Courier New', monospace; 
    color: #10b981; 
    height: 600px; 
    overflow-y: auto;
    white-space: pre-wrap;
    box-shadow: inset 0 0 10px rgba(0,0,0,0.8);
}
.cmdr-window { color: #3b82f6; }
h1, h2, h3 { font-family: 'Courier New', monospace; font-weight: bold; }
"""

with gr.Blocks(theme=gr.themes.Base(), css=custom_css) as demo:
    gr.HTML("<h1 style='text-align:center; color:#38bdf8; font-size:3em; margin-bottom:0;'>🔴 THE WAR ROOM</h1>")
    gr.HTML("<p style='text-align:center; color:#9ca3af; font-family:monospace;'>BlastRadius Autonomous SRE Agent (MATPO-GRPO)</p>")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Incident Configuration")
            task_dropdown = gr.Dropdown(choices=["easy", "medium", "hard"], value="medium", label="Scenario Difficulty")
            api_key = gr.Textbox(placeholder="nvapi-...", value=os.environ.get("TEACHER_API_KEY", ""), label="API Key", type="password")
            start_btn = gr.Button("🚀 LAUNCH AUTONOMOUS AGENT", variant="primary", size="lg")
            
            gr.Markdown("---")
            gr.Markdown("### Live Telemetry")
            reward_display = gr.Markdown("## Reward: 0.000")
            status_display = gr.Markdown("### Status: Waiting for launch...")
            
            plot_output = gr.Plot()

        with gr.Column(scale=1):
            gr.Markdown("### 🤖 Scout Module (Triage)")
            scout_terminal = gr.HTML("<div class='terminal-window' id='scout-term'>System Idle...</div>")

        with gr.Column(scale=1):
            gr.Markdown("### 🧠 Commander Module (Action)")
            cmdr_terminal = gr.HTML("<div class='terminal-window cmdr-window' id='cmdr-term'>System Idle...</div>")

    # ---------------------------------------------------------------------------
    # Stream Generator Hook
    # ---------------------------------------------------------------------------
    def trigger_agent(task_id, key):
        # Initial state setup
        yield (
            generate_system_graph({}), 
            "<div class='terminal-window'>Initializing Agent...</div>",
            "<div class='terminal-window cmdr-window'>Awaiting Triage...</div>",
            "## Reward: 0.000",
            "### Status: Running 🟢"
        )
        
        # We need to set the API key for the orchestrator
        os.environ["API_BASE_URL"] = "https://integrate.api.nvidia.com/v1"
        if key:
            os.environ["TEACHER_API_KEY"] = key
            
        orchestrator = MATPOOrchestrator(
            api_base="https://integrate.api.nvidia.com/v1",
            api_key=key or "dummy",
            model_name="meta/llama-3.1-8b-instruct", # Using teacher for demo since GRPO takes hours
            env_base_url="http://127.0.0.1:7860"
        )
        
        try:
            for obs, scout_log, cmdr_log, reward, is_done in orchestrator.run_episode_stream(task_id, max_steps=10):
                # Update UI elements
                fig = generate_system_graph(obs)
                
                # Format terminals
                s_html = f"<div class='terminal-window'>{scout_log}</div>"
                c_html = f"<div class='terminal-window cmdr-window'>{cmdr_log}</div>"
                
                yield (
                    fig, 
                    s_html, 
                    c_html, 
                    f"## Reward: {reward:+.3f}",
                    f"### Status: {'Resolved ✅' if is_done else 'Running 🟢'}"
                )
        except Exception as e:
            yield (
                generate_system_graph({}),
                f"<div class='terminal-window'>ERROR: {str(e)}</div>",
                "<div class='terminal-window cmdr-window'>ERROR</div>",
                "## Reward: ERR",
                "### Status: FAILED 🔴"
            )

    start_btn.click(
        fn=trigger_agent,
        inputs=[task_dropdown, api_key],
        outputs=[plot_output, scout_terminal, cmdr_terminal, reward_display, status_display]
    )

fast_app = gr.mount_gradio_app(fast_app, demo, path="/warroom")

if __name__ == "__main__":
    uvicorn.run(fast_app, host="0.0.0.0", port=7860)
