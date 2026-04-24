"""
Post-Incident Analysis Page — renders a report of the user's performance,
comparing their actions to the optimal playbook.
"""

ANALYSIS_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Post-Incident Analysis Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{--bg:#0a0e17;--card:#0f172a;--border:rgba(99,102,241,.15);--text:#e2e8f0;--muted:#64748b;--green:#34d399;--yellow:#fbbf24;--red:#f87171;--blue:#818cf8;--indigo:#6366f1}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;display:flex;flex-direction:column;align-items:center}
.bg-grid{position:fixed;inset:0;background-image:linear-gradient(rgba(99,102,241,.04) 1px,transparent 1px),linear-gradient(90deg,rgba(99,102,241,.04) 1px,transparent 1px);background-size:50px 50px;pointer-events:none;z-index:0}

.container{position:relative;z-index:1;max-width:1000px;width:100%;padding:40px 20px;}
.header{display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:30px;padding-bottom:20px;border-bottom:1px solid var(--border);}
.header h1{font-size:28px;font-weight:800;letter-spacing:-0.5px;}
.header p{color:var(--muted);margin-top:8px;}
.btn{font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:600;padding:8px 16px;border-radius:6px;border:1px solid var(--border);background:var(--card);color:var(--text);cursor:pointer;text-decoration:none;transition:all .15s;}
.btn:hover{border-color:var(--indigo);background:rgba(99,102,241,.1);}

.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-bottom:24px;}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:24px;}
.card h2{font-size:16px;font-weight:700;color:var(--indigo);text-transform:uppercase;letter-spacing:1px;margin-bottom:16px;display:flex;align-items:center;gap:8px;}

/* Score Breakdown */
.score-tally{font-family:'JetBrains Mono',monospace;font-size:48px;font-weight:800;text-align:center;margin:20px 0;}
.score-tally.good{color:var(--green)}.score-tally.mid{color:var(--yellow)}.score-tally.low{color:var(--red)}
.breakdown-list{list-style:none;margin-top:20px;}
.breakdown-item{display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px dashed var(--border);font-family:'JetBrains Mono',monospace;font-size:13px;}
.breakdown-item:last-child{border-bottom:none;}
.breakdown-item.pos{color:var(--green)}.breakdown-item.neg{color:var(--red)}.breakdown-item.neu{color:var(--muted)}

/* Timeline & Playbook */
table{width:100%;border-collapse:collapse;font-family:'JetBrains Mono',monospace;font-size:12px;}
th{text-align:left;color:var(--muted);padding-bottom:12px;border-bottom:1px solid var(--border);font-weight:600;font-family:'Inter',sans-serif;font-size:11px;text-transform:uppercase;letter-spacing:1px;}
td{padding:12px 0;border-bottom:1px solid rgba(255,255,255,0.02);}
.col-step{width:50px;color:var(--muted);}
.col-act{font-weight:600;color:var(--text);}
.col-success{width:80px;}

.playbook-step{margin-bottom:12px;padding-left:16px;border-left:2px solid var(--indigo);}
.playbook-cmd{font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:600;color:var(--blue);}
.playbook-target{color:var(--text);}

@media(max-width:768px){.grid{grid-template-columns:1fr;}}
</style>
</head>
<body>
<div class="bg-grid"></div>
<div class="container">
  <div class="header">
    <div>
      <h1 id="scenarioTitle">Loading Analysis...</h1>
      <p id="scenarioDesc">Fetching episode data</p>
    </div>
    <a href="/" class="btn">← Back to Simulator</a>
  </div>

  <div class="grid" id="mainGrid" style="display:none;">
    <!-- Score Card -->
    <div class="card">
      <h2>🏆 Final Score</h2>
      <div id="scoreBig" class="score-tally">0.00</div>
      <p style="text-align:center;color:var(--muted);font-size:13px;" id="resolutionStatus"></p>
      
      <ul class="breakdown-list" id="breakdownList"></ul>
    </div>

    <!-- Optimal Playbook -->
    <div class="card">
      <h2>📖 Ground Truth Playbook</h2>
      <p style="font-size:13px;color:var(--muted);margin-bottom:16px;">The ideal response to this specific incident.</p>
      
      <div style="margin-bottom:20px;">
        <div style="font-size:11px;text-transform:uppercase;color:var(--muted);margin-bottom:8px;letter-spacing:1px;">Root Cause</div>
        <div style="font-size:14px;font-weight:600;padding:12px;background:rgba(255,255,255,0.03);border-radius:6px;border-left:3px solid var(--red);" id="rootCauseDesc"></div>
      </div>
      
      <div style="font-size:11px;text-transform:uppercase;color:var(--muted);margin-bottom:8px;letter-spacing:1px;">Optimal Fix Actions</div>
      <div id="optimalActions"></div>
    </div>
    
    <!-- Action Timeline -->
    <div class="card" style="grid-column: 1 / -1;">
      <h2>⏱️ Your Action Timeline</h2>
      <table>
        <thead><tr><th>Step</th><th>Command</th><th>Target / Params</th><th>Cost</th><th>Status</th></tr></thead>
        <tbody id="timelineBody"></tbody>
      </table>
    </div>
  </div>
</div>

<script>
async function loadAnalysis() {
  try {
    const res = await fetch('/analysis-data');
    if (!res.ok) throw new Error("No analysis data available. Run an episode first.");
    const data = await res.json();
    
    document.getElementById('mainGrid').style.display = 'grid';
    document.getElementById('scenarioTitle').textContent = data.scenario.title;
    document.getElementById('scenarioDesc').textContent = data.scenario.description;
    
    // Score
    const scoreVal = data.final_score.reward;
    const sb = document.getElementById('scoreBig');
    sb.textContent = scoreVal.toFixed(2);
    sb.className = 'score-tally ' + (scoreVal >= 0.7 ? 'good' : scoreVal >= 0.4 ? 'mid' : 'low');
    
    document.getElementById('resolutionStatus').textContent = data.state.is_resolved 
      ? '✅ Incident was successfully mitigated' 
      : '❌ Operations terminated before incident was resolved';
      
    // Breakdown
    const bl = document.getElementById('breakdownList');
    const bd = data.final_score.breakdown;
    let bHtml = '';
    for(const [key, val] of Object.entries(bd)) {
      const cls = val > 0 ? 'pos' : val < 0 ? 'neg' : 'neu';
      const sign = val > 0 ? '+' : '';
      bHtml += `<li class="breakdown-item ${cls}"><span>${key.replace(/_/g, ' ')}</span><span>${sign}${val.toFixed(2)}</span></li>`;
    }
    bl.innerHTML = bHtml;
    
    // Playbook
    const optimal = data.optimal;
    document.getElementById('rootCauseDesc').innerHTML = `<strong>${optimal.root_cause_service}</strong><br><span style="font-size:12px;color:var(--muted)">${optimal.root_cause_description}</span>`;
    
    let actHtml = '';
    optimal.correct_fix_actions.forEach((act, i) => {
      actHtml += `<div class="playbook-step">
        <span class="playbook-cmd">${act.command}</span> 
        <span class="playbook-target">${act.target}</span>
      </div>`;
    });
    document.getElementById('optimalActions').innerHTML = actHtml;
    
    // Timeline
    let tHtml = '';
    data.state.actions_taken.forEach(act => {
      const succ = act.succeeded ? '<span style="color:var(--green)">Success</span>' : '<span style="color:var(--muted)">-</span>';
      tHtml += `<tr>
        <td class="col-step">${act.step}</td>
        <td class="col-act">${act.command}</td>
        <td>${act.target || '-'}</td>
        <td style="color:var(--yellow)">${act.time_cost}m</td>
        <td class="col-success">${succ}</td>
      </tr>`;
    });
    document.getElementById('timelineBody').innerHTML = tHtml;
    
  } catch (err) {
    document.getElementById('scenarioTitle').textContent = "Error Loading Analysis";
    document.getElementById('scenarioDesc').textContent = err.message;
  }
}

document.addEventListener('DOMContentLoaded', loadAnalysis);
</script>
</body>
</html>"""
