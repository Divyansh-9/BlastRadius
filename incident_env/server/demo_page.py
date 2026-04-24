"""
Interactive demo page — lets visitors play through an incident scenario
directly from their browser. Shows service health, terminal output,
reward accumulation, and cascading failures in real-time.
"""

DEMO_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Incident Simulator — Live Demo</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{--bg:#0a0e17;--card:#0f172a;--border:rgba(99,102,241,.15);--border-hi:rgba(99,102,241,.4);--text:#e2e8f0;--muted:#64748b;--green:#34d399;--yellow:#fbbf24;--red:#f87171;--blue:#818cf8;--indigo:#6366f1}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden}
.bg-grid{position:fixed;inset:0;background-image:linear-gradient(rgba(99,102,241,.04) 1px,transparent 1px),linear-gradient(90deg,rgba(99,102,241,.04) 1px,transparent 1px);background-size:50px 50px;pointer-events:none;z-index:0}

/* Layout */
.app{position:relative;z-index:1;display:grid;grid-template-rows:auto 1fr;height:100vh}
.topbar{display:flex;align-items:center;justify-content:space-between;padding:12px 20px;border-bottom:1px solid var(--border);background:rgba(10,14,23,.9);backdrop-filter:blur(12px)}
.topbar h1{font-size:16px;font-weight:700;display:flex;align-items:center;gap:8px}
.topbar h1 span{color:var(--red)}
.topbar-right{display:flex;align-items:center;gap:16px}
.stat{font-family:'JetBrains Mono',monospace;font-size:13px;display:flex;align-items:center;gap:6px}
.stat-label{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.5px}

.main{display:grid;grid-template-columns:260px 1fr 300px;gap:0;overflow:hidden}

/* Left — Service Panel */
.panel-services{border-right:1px solid var(--border);padding:16px;overflow-y:auto;background:rgba(15,23,42,.4)}
.panel-title{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:var(--indigo);margin-bottom:12px}
.svc{padding:10px 12px;border-radius:8px;border:1px solid transparent;margin-bottom:6px;cursor:pointer;transition:all .2s}
.svc:hover{border-color:var(--border-hi);background:rgba(99,102,241,.05)}
.svc.selected{border-color:var(--indigo);background:rgba(99,102,241,.08)}
.svc-header{display:flex;align-items:center;justify-content:space-between}
.svc-name{font-size:13px;font-weight:600}
.svc-badge{font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:600;padding:2px 8px;border-radius:4px;text-transform:uppercase}
.svc-badge.healthy{background:rgba(52,211,153,.12);color:var(--green)}
.svc-badge.degraded{background:rgba(251,191,36,.12);color:var(--yellow)}
.svc-badge.down{background:rgba(248,113,113,.12);color:var(--red)}
.svc-desc{font-size:11px;color:var(--muted);margin-top:4px}
.cascade-alert{font-size:11px;color:var(--red);margin-top:4px;animation:flashIn .5s}
@keyframes flashIn{from{opacity:0;transform:translateY(-4px)}to{opacity:1;transform:translateY(0)}}

/* Center — Terminal Output */
.panel-terminal{display:flex;flex-direction:column;overflow:hidden}
.terminal-header{padding:12px 16px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;background:rgba(15,23,42,.5)}
.terminal-header span{font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--muted)}
.terminal{flex:1;padding:16px;overflow-y:auto;font-family:'JetBrains Mono',monospace;font-size:12.5px;line-height:1.7;background:rgba(2,6,14,.6);white-space:pre-wrap;word-break:break-word}
.terminal .sys{color:var(--indigo)}
.terminal .ok{color:var(--green)}
.terminal .warn{color:var(--yellow)}
.terminal .err{color:var(--red)}
.terminal .reward-line{color:var(--green);font-weight:600}
.terminal .penalty-line{color:var(--red);font-weight:600}
.terminal .cascade-line{color:var(--red);animation:flashIn .5s}
.terminal .step-sep{color:rgba(99,102,241,.3);user-select:none}

/* Actions Bar */
.actions-bar{padding:12px 16px;border-top:1px solid var(--border);background:rgba(15,23,42,.6);display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.act-group{display:flex;gap:6px;align-items:center}
.act-group-label{font-size:10px;text-transform:uppercase;letter-spacing:.5px;color:var(--muted);margin-right:4px}
.btn{font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:500;padding:6px 12px;border-radius:6px;border:1px solid var(--border);background:rgba(15,23,42,.8);color:var(--text);cursor:pointer;transition:all .15s;white-space:nowrap}
.btn:hover:not(:disabled){border-color:var(--border-hi);background:rgba(99,102,241,.1);transform:translateY(-1px)}
.btn:disabled{opacity:.35;cursor:not-allowed}
.btn.primary{background:rgba(99,102,241,.15);border-color:var(--indigo);color:var(--blue)}
.btn.danger{background:rgba(239,68,68,.1);border-color:rgba(239,68,68,.3);color:var(--red)}
.btn.success{background:rgba(52,211,153,.1);border-color:rgba(52,211,153,.3);color:var(--green)}
.btn .cost{font-size:9px;opacity:.6;margin-left:4px}

/* Right — Score Panel */
.panel-score{border-left:1px solid var(--border);padding:16px;overflow-y:auto;background:rgba(15,23,42,.4)}
.score-big{font-family:'JetBrains Mono',monospace;font-size:48px;font-weight:800;text-align:center;margin:16px 0 8px;transition:color .3s}
.score-big.good{color:var(--green)}
.score-big.mid{color:var(--yellow)}
.score-big.low{color:var(--red)}
.score-label{text-align:center;font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}
.reward-history{margin-top:20px}
.rh-item{display:flex;justify-content:space-between;align-items:center;padding:6px 8px;border-radius:4px;margin-bottom:3px;font-family:'JetBrains Mono',monospace;font-size:11px;animation:fadeUp .3s}
@keyframes fadeUp{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
.rh-item.pos{background:rgba(52,211,153,.06);color:var(--green)}
.rh-item.neg{background:rgba(248,113,113,.06);color:var(--red)}
.rh-item.zero{background:rgba(100,116,139,.06);color:var(--muted)}
.rh-step{opacity:.5}
.rh-cmd{flex:1;margin:0 8px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.clock{font-family:'JetBrains Mono',monospace;font-size:28px;font-weight:700;text-align:center;margin-top:20px;color:var(--yellow)}
.clock-label{text-align:center;font-size:11px;color:var(--muted);margin-top:4px;text-transform:uppercase;letter-spacing:.5px}
.severity-badge{text-align:center;margin-top:16px}
.severity-badge span{font-family:'JetBrains Mono',monospace;font-size:14px;font-weight:700;padding:4px 16px;border-radius:6px}
.severity-badge .p1{background:rgba(239,68,68,.15);color:var(--red);border:1px solid rgba(239,68,68,.3)}
.severity-badge .p2{background:rgba(251,191,36,.15);color:var(--yellow);border:1px solid rgba(251,191,36,.3)}

/* Scenario picker overlay */
.overlay{position:fixed;inset:0;background:rgba(0,0,0,.7);backdrop-filter:blur(8px);z-index:100;display:flex;align-items:center;justify-content:center}
.overlay.hidden{display:none}
.picker{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:36px;max-width:700px;width:90%}
.picker h2{font-size:22px;font-weight:800;margin-bottom:6px;text-align:center}
.picker p{font-size:14px;color:var(--muted);text-align:center;margin-bottom:24px}
.scenario-cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.sc{padding:20px;border-radius:12px;border:1px solid var(--border);cursor:pointer;transition:all .2s;text-align:center}
.sc:hover{border-color:var(--border-hi);transform:translateY(-3px);box-shadow:0 8px 30px rgba(99,102,241,.15)}
.sc-diff{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.8px;margin-bottom:8px}
.sc-diff.easy{color:var(--green)}.sc-diff.medium{color:var(--yellow)}.sc-diff.hard{color:var(--red)}
.sc h3{font-size:14px;font-weight:700;margin-bottom:6px}
.sc p{font-size:12px;color:var(--muted);line-height:1.4}

/* Done overlay */
.done-overlay{position:fixed;inset:0;background:rgba(0,0,0,.8);backdrop-filter:blur(12px);z-index:100;display:flex;align-items:center;justify-content:center}
.done-overlay.hidden{display:none}
.done-card{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:40px;text-align:center;max-width:400px}
.done-card h2{font-size:24px;font-weight:800;margin-bottom:12px}
.done-score{font-family:'JetBrains Mono',monospace;font-size:64px;font-weight:800;margin:16px 0}

/* Diagnosis modal */
.diag-overlay{position:fixed;inset:0;background:rgba(0,0,0,.6);backdrop-filter:blur(6px);z-index:100;display:flex;align-items:center;justify-content:center}
.diag-overlay.hidden{display:none}
.diag-card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:28px;max-width:480px;width:90%}
.diag-card h3{margin-bottom:16px;font-size:18px}
.diag-card label{display:block;font-size:12px;font-weight:600;color:var(--muted);margin-bottom:4px;margin-top:12px;text-transform:uppercase;letter-spacing:.5px}
.diag-card input,.diag-card textarea{width:100%;padding:8px 12px;background:rgba(2,6,14,.6);border:1px solid var(--border);border-radius:6px;color:var(--text);font-family:'JetBrains Mono',monospace;font-size:13px;outline:none}
.diag-card textarea{height:70px;resize:vertical}
.diag-card input:focus,.diag-card textarea:focus{border-color:var(--indigo)}
.diag-actions{display:flex;gap:8px;margin-top:16px;justify-content:flex-end}

@media(max-width:900px){.main{grid-template-columns:1fr;grid-template-rows:auto 1fr auto}.panel-services,.panel-score{display:none}}
</style>
</head>
<body>
<div class="bg-grid"></div>

<!-- Scenario Picker -->
<div class="overlay" id="picker">
  <div class="picker">
    <h2>🚨 Choose Your Incident</h2>
    <p>You are the on-call SRE. A production incident just fired. Pick a scenario and diagnose the failure before it spreads.</p>
    <div class="scenario-cards">
      <div class="sc" onclick="startScenario('easy')">
        <div class="sc-diff easy">● Easy</div>
        <h3>DB Pool Exhaustion</h3>
        <p>Connection pool maxed. API returning 503s. Find the cause and fix it.</p>
      </div>
      <div class="sc" onclick="startScenario('medium')">
        <div class="sc-diff medium">● Medium</div>
        <h3>Bad Deploy Cascade</h3>
        <p>Payments are down. But is it really the payment service? Dig deeper.</p>
      </div>
      <div class="sc" onclick="startScenario('hard')">
        <div class="sc-diff hard">● Hard</div>
        <h3>Thundering Herd</h3>
        <p>CDN looks broken. Multiple services failing. Fix order matters. Don't panic.</p>
      </div>
    </div>
  </div>
</div>

<!-- Done Overlay -->
<div class="done-overlay hidden" id="doneOverlay">
  <div class="done-card">
    <h2 id="doneTitle">Incident Resolved!</h2>
    <div class="done-score" id="doneScore">0.75</div>
    <p style="color:var(--muted);margin-bottom:20px" id="doneFeedback"></p>
    <div style="display:flex;gap:12px;justify-content:center;">
      <button class="btn" onclick="showPicker()" style="font-size:14px;padding:10px 16px">New Scenario</button>
      <a href="/analysis" class="btn primary" style="font-size:14px;padding:10px 24px">View Analysis Report →</a>
    </div>
  </div>
</div>

<!-- Diagnosis Modal -->
<div class="diag-overlay hidden" id="diagOverlay">
  <div class="diag-card">
    <h3>🔍 Submit Diagnosis</h3>
    <label>Root Cause Service</label>
    <input type="text" id="diagRoot" placeholder="e.g. database, auth-service">
    <label>Causal Chain (one step per line)</label>
    <textarea id="diagChain" placeholder="database connection pool exhausted&#10;API gateway cannot acquire connections&#10;users see 503 errors"></textarea>
    <label>Confidence (0.0 – 1.0)</label>
    <input type="number" id="diagConf" value="0.8" min="0" max="1" step="0.1">
    <div class="diag-actions">
      <button class="btn" onclick="closeDiag()">Cancel</button>
      <button class="btn primary" onclick="submitDiagnosis()">Submit Diagnosis</button>
    </div>
  </div>
</div>

<!-- Main App -->
<div class="app">
  <div class="topbar">
    <h1><span>🚨</span> Incident Response Simulator</h1>
    <div class="topbar-right">
      <div class="stat"><span class="stat-label">Step</span> <span id="stepCount">0</span>/20</div>
      <div class="stat"><span class="stat-label">Score</span> <span id="topScore">0.00</span></div>
      <button class="btn" onclick="showPicker()" style="font-size:11px">↩ New Incident</button>
    </div>
  </div>

  <div class="main">
    <!-- Left: Services -->
    <div class="panel-services">
      <div class="panel-title">Services</div>
      <div id="serviceList"></div>
    </div>

    <!-- Center: Terminal -->
    <div class="panel-terminal">
      <div class="terminal-header">
        <span>incident-response-terminal</span>
        <span id="termStep">ready</span>
      </div>
      <div class="terminal" id="terminal">
<span class="sys">Welcome to the IT Incident Response Simulator.

Pick a scenario to begin. You'll need to:
  1. Investigate — check service status, logs, metrics, and dependencies
  2. Diagnose — identify the root cause and explain the causal chain
  3. Fix — apply the right remediation in the correct order

⚠️  Every action costs simulated time. Failures SPREAD while you investigate.
    Choose wisely — you have 25 steps maximum.

Hint: Start with "Check Status" to see what's broken.
</span></div>
      <div class="actions-bar">
        <div class="act-group">
          <span class="act-group-label">Investigate</span>
          <button class="btn" onclick="act('check_status')" id="btnStatus" disabled>Status <span class="cost">FREE</span></button>
          <button class="btn" onclick="actTarget('check_logs')" id="btnLogs" disabled>Logs <span class="cost">2m</span></button>
          <button class="btn" onclick="actTarget('check_metrics')" id="btnMetrics" disabled>Metrics <span class="cost">1m</span></button>
          <button class="btn" onclick="act('check_dependencies')" id="btnDeps" disabled>Deps <span class="cost">1m</span></button>
        </div>
        <div class="act-group">
          <span class="act-group-label">Act</span>
          <button class="btn primary" onclick="openDiag()" id="btnDiag" disabled>🔍 Diagnose <span class="cost">FREE</span></button>
          <button class="btn danger" onclick="actTarget('restart_service')" id="btnRestart" disabled>Restart <span class="cost">3m</span></button>
          <button class="btn danger" onclick="actTarget('rollback_deploy')" id="btnRollback" disabled>Rollback <span class="cost">5m</span></button>
          <button class="btn success" onclick="actTarget('scale_service')" id="btnScale" disabled>Scale <span class="cost">2m</span></button>
        </div>
      </div>
    </div>

    <!-- Right: Score -->
    <div class="panel-score">
      <div class="panel-title">Score</div>
      <div class="score-big low" id="scoreBig">0.00</div>
      <div class="score-label">Total Reward</div>

      <div class="severity-badge" id="sevBadge"><span class="p2">P2</span></div>

      <div class="clock" id="clock">00:00</div>
      <div class="clock-label">Time Elapsed</div>

      <div class="reward-history">
        <div class="panel-title" style="margin-top:16px">Reward Log</div>
        <div id="rewardLog"></div>
      </div>
    </div>
  </div>
</div>

<script>
const API = '';  // same origin
let selectedService = '';
let totalScore = 0;
let stepNum = 0;
let done = false;
let services = {};

function showPicker(){
  document.getElementById('picker').classList.remove('hidden');
  document.getElementById('doneOverlay').classList.add('hidden');
}

async function startScenario(taskId){
  document.getElementById('picker').classList.add('hidden');
  document.getElementById('doneOverlay').classList.add('hidden');
  totalScore=0; stepNum=0; done=false; selectedService='';
  document.getElementById('rewardLog').innerHTML='';
  document.getElementById('terminal').innerHTML='';
  toggleButtons(false);

  try{
    const res = await fetch(API+'/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task_id:taskId})});
    const data = await res.json();
    handleResponse(data, 'reset');
    toggleButtons(true);
  }catch(e){appendTerm('err','ERROR: '+e.message)}
}

function handleResponse(data, cmd){
  const obs = data.observation;
  const reward = data.reward||0;
  totalScore += reward;

  if(cmd!=='reset') stepNum++;
  updateStats();

  // Update services
  services = obs.services_status||{};
  renderServices(obs);

  // Update terminal
  if(cmd!=='reset'){
    appendTerm('step-sep','───────────────────────────────────────');
  }
  const output = obs.output||'';
  // Color code the output
  const colored = output
    .replace(/🟢/g,'<span class="ok">🟢</span>')
    .replace(/🟡/g,'<span class="warn">🟡</span>')
    .replace(/🔴/g,'<span class="err">🔴</span>')
    .replace(/(ERROR|CRITICAL|FATAL|DOWN)/g,'<span class="err">$1</span>')
    .replace(/(WARNING|DEGRADED|⚠️)/g,'<span class="warn">$1</span>')
    .replace(/(HEALTHY|✅|recovered)/g,'<span class="ok">$1</span>')
    .replace(/(CASCADE ALERT)/g,'<span class="cascade-line">$1</span>');
  appendTermRaw(colored);

  // Show hint
  if(obs.hint) appendTerm('sys','💡 '+obs.hint);

  // Reward log
  if(cmd!=='reset' && reward!==undefined) addRewardEntry(cmd, reward);

  // Severity
  const sev = obs.incident_severity||'P2';
  document.getElementById('sevBadge').innerHTML =
    `<span class="${sev.toLowerCase()}">${sev}</span>`;

  // Clock
  const mins = obs.time_elapsed_minutes||0;
  document.getElementById('clock').textContent =
    String(Math.floor(mins/60)).padStart(2,'0')+':'+String(mins%60).padStart(2,'0');

  // Done?
  if(data.done){
    done=true;
    toggleButtons(false);
    const finalScore = data.info?.final_score ?? totalScore;
    const feedback = data.info?.final_feedback || (data.info?.final_breakdown ? JSON.stringify(data.info.final_breakdown) : '');
    setTimeout(()=>{
      document.getElementById('doneTitle').textContent = obs.services_status && Object.values(obs.services_status).every(s=>s==='healthy') ? '✅ Incident Resolved!' : '⏱️ Time\\'s Up';
      const ds = document.getElementById('doneScore');
      ds.textContent = finalScore.toFixed(2);
      ds.style.color = finalScore>=0.7?'var(--green)':finalScore>=0.4?'var(--yellow)':'var(--red)';
      document.getElementById('doneFeedback').textContent = feedback||`Score: ${finalScore.toFixed(4)} in ${stepNum} steps`;
      document.getElementById('doneOverlay').classList.remove('hidden');
    },600);
  }

  // Scroll terminal
  const term = document.getElementById('terminal');
  term.scrollTop = term.scrollHeight;
}

function renderServices(obs){
  const list = document.getElementById('serviceList');
  let html='';
  const atRisk = obs.services_at_risk||[];
  for(const[name,status] of Object.entries(services)){
    const sel = name===selectedService?'selected':'';
    const risk = atRisk.includes(name)?`<div class="cascade-alert">⚠️ At risk of cascade</div>`:'';
    html+=`<div class="svc ${sel}" onclick="selectService('${name}')">
      <div class="svc-header">
        <span class="svc-name">${name}</span>
        <span class="svc-badge ${status}">${status}</span>
      </div>
      ${risk}
    </div>`;
  }
  list.innerHTML=html;
}

function selectService(name){
  selectedService=name;
  renderServices({services_status:services,services_at_risk:[]});
}

async function act(command, target, params){
  if(done) return;
  toggleButtons(false);
  const body={command, target:target||'', parameters:params||{}};
  try{
    const res=await fetch(API+'/step',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const data=await res.json();
    handleResponse(data, command+(target?' '+target:''));
  }catch(e){appendTerm('err','ERROR: '+e.message)}
  if(!done) toggleButtons(true);
}

function actTarget(command){
  if(!selectedService){
    appendTerm('warn','⚠️  Select a service from the left panel first.');
    return;
  }
  if(command==='scale_service'){
    act(command, selectedService, {instances:4, max_connections:200});
  } else {
    act(command, selectedService);
  }
}

function openDiag(){document.getElementById('diagOverlay').classList.remove('hidden')}
function closeDiag(){document.getElementById('diagOverlay').classList.add('hidden')}
function submitDiagnosis(){
  const root=document.getElementById('diagRoot').value.trim();
  const chain=document.getElementById('diagChain').value.trim().split('\\n').filter(Boolean);
  const conf=parseFloat(document.getElementById('diagConf').value)||0.8;
  if(!root){appendTerm('warn','⚠️  Enter a root cause service name.');return;}
  closeDiag();
  act('diagnose','',{root_cause:root,causal_chain:chain,confidence:conf});
}

function updateStats(){
  document.getElementById('stepCount').textContent=stepNum;
  document.getElementById('topScore').textContent=totalScore.toFixed(2);
  document.getElementById('termStep').textContent=`step ${stepNum}`;
  const sb=document.getElementById('scoreBig');
  sb.textContent=totalScore.toFixed(2);
  sb.className='score-big '+(totalScore>=0.5?'good':totalScore>=0.2?'mid':'low');
}

function addRewardEntry(cmd, reward){
  const cls=reward>0?'pos':reward<0?'neg':'zero';
  const sign=reward>0?'+':'';
  const log=document.getElementById('rewardLog');
  log.innerHTML=`<div class="rh-item ${cls}"><span class="rh-step">#${stepNum}</span><span class="rh-cmd">${cmd}</span><span>${sign}${reward.toFixed(3)}</span></div>`+log.innerHTML;
}

function appendTerm(cls, text){
  const term=document.getElementById('terminal');
  const el=document.createElement('div');
  el.className=cls;
  el.textContent=text;
  term.appendChild(el);
  term.scrollTop=term.scrollHeight;
}

function appendTermRaw(html){
  const term=document.getElementById('terminal');
  const el=document.createElement('div');
  el.innerHTML=html;
  term.appendChild(el);
  term.scrollTop=term.scrollHeight;
}

function toggleButtons(enabled){
  document.querySelectorAll('.actions-bar .btn').forEach(b=>b.disabled=!enabled);
}
</script>
</body>
</html>"""
