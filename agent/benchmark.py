import os
import time
import argparse
from datetime import datetime
from pathlib import Path

from agent.orchestrator import MATPOOrchestrator

ALL_SCENARIOS = [
    "easy",
    "medium",
    "hard",
    "easy_dns_propagation",
    "easy_redis_oom",
    "medium_cert_expiry",
    "medium_k8s_eviction",
    "hard_regex_catastrophe",
    "hard_db_failover",
    "hard_s3_keyspace_overflow",
]

def generate_html_report(results, model_name, output_path):
    """Generate a beautiful HTML report from the benchmark results."""
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BlastRadius Benchmark Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #0d1117; color: #c9d1d9; margin: 0; padding: 20px; }}
        h1, h2, h3 {{ color: #58a6ff; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        .summary {{ display: flex; gap: 20px; margin-bottom: 30px; }}
        .stat-box {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 20px; flex: 1; text-align: center; }}
        .stat-val {{ font-size: 32px; font-weight: bold; color: #79c0ff; margin-bottom: 5px; }}
        .stat-label {{ font-size: 14px; color: #8b949e; text-transform: uppercase; }}
        table {{ width: 100%; border-collapse: collapse; margin-bottom: 30px; background: #161b22; border: 1px solid #30363d; border-radius: 6px; overflow: hidden; }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #30363d; }}
        th {{ background: #21262d; font-weight: 600; color: #c9d1d9; }}
        tr:last-child td {{ border-bottom: none; }}
        .good {{ color: #3fb950; font-weight: bold; }}
        .mid {{ color: #d29922; font-weight: bold; }}
        .bad {{ color: #f85149; font-weight: bold; }}
        .timestamp {{ color: #8b949e; font-size: 14px; text-align: center; margin-top: 40px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>💥 BlastRadius Benchmark Report</h1>
        <p style="color: #8b949e; margin-bottom: 30px;">Model: <strong>{model_name}</strong></p>
        
        <div class="summary">
            <div class="stat-box">
                <div class="stat-val">{sum(r['score'] for r in results) / len(results):.2f}</div>
                <div class="stat-label">Average Score</div>
            </div>
            <div class="stat-box">
                <div class="stat-val">{sum(1 for r in results if r['resolved'])} / {len(results)}</div>
                <div class="stat-label">Scenarios Resolved</div>
            </div>
            <div class="stat-box">
                <div class="stat-val">{sum(r['steps'] for r in results) / len(results):.1f}</div>
                <div class="stat-label">Avg Steps Taken</div>
            </div>
        </div>

        <h2>Scenario Breakdown</h2>
        <table>
            <thead>
                <tr>
                    <th>Scenario ID</th>
                    <th>Final Score</th>
                    <th>Resolved</th>
                    <th>Steps</th>
                </tr>
            </thead>
            <tbody>
"""

    for r in results:
        score = r['score']
        score_class = "good" if score >= 0.7 else ("mid" if score >= 0.4 else "bad")
        resolved_icon = "✅" if r['resolved'] else "❌"
        
        html += f"""
                <tr>
                    <td style="font-family: monospace;">{r['task_id']}</td>
                    <td class="{score_class}">{score:.4f}</td>
                    <td>{resolved_icon}</td>
                    <td>{r['steps']}</td>
                </tr>"""

    html += f"""
            </tbody>
        </table>
        
        <div class="timestamp">
            Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </div>
    </div>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\\n✅ HTML report saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="BlastRadius Benchmark CLI")
    parser.add_argument("--model", default="meta/llama-3.1-8b-instruct", help="Model name or path to checkpoint")
    parser.add_argument("--scenarios", nargs="+", default="all", help="List of scenario IDs to run, or 'all'")
    parser.add_argument("--output-dir", default="docs/runs", help="Directory to save the report")
    parser.add_argument("--api-base", default=os.environ.get("API_BASE_URL", "http://localhost:8000/v1"), help="LLM API Base URL")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "dummy"), help="API Key")
    parser.add_argument("--env-url", default=os.environ.get("ENV_BASE_URL", "http://127.0.0.1:7860"), help="Env Base URL")
    
    args = parser.parse_args()
    
    if args.scenarios == "all" or args.scenarios == ["all"]:
        scenarios = ALL_SCENARIOS
    else:
        scenarios = args.scenarios

    print(f"\\n{'='*60}")
    print("  BLASTRADIUS AUTO-BENCHMARK")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Target Scenarios: {len(scenarios)}")
    print(f"Environment: {args.env_url}\\n")

    orchestrator = MATPOOrchestrator(
        api_base=args.api_base,
        api_key=args.api_key,
        model_name=args.model,
        env_base_url=args.env_url,
        temperature=0.0, # Greedy for benchmarking
    )

    results = []
    
    # Ensure output dir exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    for i, task_id in enumerate(scenarios, 1):
        print(f"Running [{i}/{len(scenarios)}] {task_id} ...", end="", flush=True)
        start_time = time.time()
        
        try:
            rollout = orchestrator.run_episode(task_id, max_steps=20, verbose=False)
            elapsed = time.time() - start_time
            
            score = rollout.final_score
            resolved = rollout.resolved
            steps = rollout.total_steps
            
            icon = "✅" if score >= 0.7 else ("🟡" if score >= 0.4 else "🔴")
            print(f" done in {elapsed:.1f}s | Score: {score:.4f} {icon} | Resolved: {resolved} | Steps: {steps}")
            
            results.append({
                "task_id": task_id,
                "score": score,
                "resolved": resolved,
                "steps": steps,
                "time_sec": elapsed,
            })
            
        except Exception as e:
            print(f" FAILED: {str(e)}")
            results.append({
                "task_id": task_id,
                "score": 0.0,
                "resolved": False,
                "steps": 0,
                "time_sec": 0,
                "error": str(e)
            })

    # Summary
    print(f"\\n{'='*60}")
    print("  BENCHMARK COMPLETE")
    print(f"{'='*60}")
    avg_score = sum(r['score'] for r in results) / len(results)
    resolved_count = sum(1 for r in results if r['resolved'])
    print(f"Average Score: {avg_score:.4f}")
    print(f"Resolved:      {resolved_count} / {len(results)}")
    
    # Generate HTML report
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(args.output_dir) / f"benchmark_{date_str}.html"
    generate_html_report(results, args.model, report_path)
    
if __name__ == "__main__":
    main()
