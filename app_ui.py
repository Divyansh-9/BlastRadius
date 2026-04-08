import json
import gradio as gr
import uvicorn
from fastapi import FastAPI
from incident_env.models import IncidentAction, VALID_COMMANDS
from incident_env.server.app import app as fast_app
from incident_env.client import IncidentEnv

client = IncidentEnv(base_url="http://127.0.0.1:7860")

def format_observation(obs_dict: dict) -> str:
    """Format the observation payload into markdown."""
    text = f"### Simulator Observation\n\n"
    text += f"**Time Elapsed**: {obs_dict.get('time_elapsed_minutes', 0)} minutes\n"
    text += f"**Incident Severity**: {obs_dict.get('incident_severity', 'Unknown')}\n\n"
    
    text += f"#### System Output\n```text\n{obs_dict.get('output', 'No output.')}\n```\n\n"
    
    text += f"#### Active Alerts\n"
    alerts = obs_dict.get('active_alerts', [])
    if alerts:
        for alert in alerts:
            text += f"- 🔴 {alert}\n"
    else:
        text += "*No active alerts.*\n"
        
    at_risk = obs_dict.get('services_at_risk', [])
    if at_risk:
        text += f"\n**Services At Risk**: {', '.join(at_risk)}\n"
        
    hint = obs_dict.get('hint', '')
    if hint:
        text += f"\n> **Hint**: {hint}\n"
        
    return text

def format_state(state_dict: dict) -> str:
    """Format the internal state."""
    text = f"### Episode State\n\n"
    text += f"- **Step Count**: {state_dict.get('step_count', 0)}\n"
    text += f"- **Total Reward**: {state_dict.get('total_reward', 0.0):.3f}\n"
    text += f"- **Resolved**: {'Yes' if state_dict.get('is_resolved') else 'No'}\n"
    text += f"- **Done**: {'Yes' if state_dict.get('done') else 'No'}\n"
    
    resolved_svcs = state_dict.get('services_resolved', [])
    if resolved_svcs:
        text += f"\n**Services Resolved**: {', '.join(resolved_svcs)}\n"
        
    return text

def handle_reset(task_id: str):
    """Callback to reset the environment."""
    try:
        res = client.reset(task_id=task_id.lower())
        obs_md = format_observation(res.observation)
        state_dict = client.state()
        state_md = format_state(state_dict)
        return obs_md, state_md, f"Environment reset to scenario: {task_id}"
    except Exception as e:
        return f"**Error resetting**: {str(e)}", "", ""

def handle_step(command: str, target: str, params_str: str):
    """Callback to process an agent/human action."""
    try:
        params = {}
        if params_str.strip():
            params = json.loads(params_str)
            
        res = client.step(command=command, target=target, parameters=params)
        
        obs_md = format_observation(res.observation)
        state_dict = client.state()
        state_md = format_state(state_dict)
        
        info_str = f"**Last Action Reward**: {res.reward:.3f}\n"
        if 'error' in res.info:
            info_str += f"\n**Error**: {res.info['error']}"
            
        if res.done:
            info_str += "\n# 🏁 EPISODE COMPLETE\n"
            info_str += f"**Final Score**: {res.info.get('final_score', 0):.3f}\n"
            info_str += f"**Feedback**: {res.info.get('final_feedback', '')}\n"
            
        return obs_md, state_md, info_str
    except Exception as e:
        return "**Connection Error**", "**Connection Error**", f"**Step Error**: {str(e)}"

with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# 🚨 SRE Incident Response Simulator")
    gr.Markdown("Agent benchmark environment for debugging cascading production failures. Core engine routes requests via OpenEnv `client.py` API.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Initialize Scenario")
            task_dropdown = gr.Dropdown(choices=["easy", "medium", "hard"], value="easy", label="Task Difficulty")
            reset_btn = gr.Button("Initialize / Reset Environment", variant="primary")
            
            gr.Markdown("### Take Action")
            command_dropdown = gr.Dropdown(choices=list(VALID_COMMANDS), value="check_status", label="Command")
            target_input = gr.Textbox(placeholder="e.g. database, auth-service...", label="Target Service")
            params_input = gr.Textbox(placeholder='{"root_cause": "cpu"}', label="Parameters (JSON)", lines=2)
            step_btn = gr.Button("Execute Action", variant="primary")
            
            action_status = gr.Markdown("")
        
        with gr.Column(scale=2):
            obs_display = gr.Markdown("Initialize environment to see observations...")
            state_display = gr.Markdown("Episode state will appear here.")
            
    reset_btn.click(fn=handle_reset, inputs=[task_dropdown], outputs=[obs_display, state_display, action_status])
    step_btn.click(fn=handle_step, inputs=[command_dropdown, target_input, params_input], outputs=[obs_display, state_display, action_status])

# Mount Gradio securely onto the internal FastAPI loop for 7860
fast_app = gr.mount_gradio_app(fast_app, demo, path="/ui")

if __name__ == "__main__":
    uvicorn.run(fast_app, host="0.0.0.0", port=7860)
