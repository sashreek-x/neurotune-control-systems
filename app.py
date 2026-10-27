from flask import Flask, request, jsonify, render_template_string, send_file
import numpy as np
from scipy import signal, optimize
from scipy.integrate import cumulative_trapezoid
import io, csv, json, math
from datetime import datetime

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

PAGE = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NeuroTune | Control Systems Studio</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{--bg:#0b1020;--panel:#121a2e;--panel2:#18223a;--line:#293652;--text:#eef3ff;--muted:#9caccb;--accent:#72e0c1;--blue:#8eb8ff;--warn:#ffce7a}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 20% 0%,#172447 0,#0b1020 42%);color:var(--text);font:15px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}
header{padding:26px clamp(16px,4vw,56px);border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:20px;align-items:center;flex-wrap:wrap}
.brand{display:flex;align-items:center;gap:12px}.logo{width:42px;height:42px;border-radius:13px;background:linear-gradient(135deg,var(--accent),var(--blue));display:grid;place-items:center;color:#09101e;font-weight:900;font-size:20px}
h1{font-size:23px;margin:0;letter-spacing:-.5px}header p{margin:3px 0 0;color:var(--muted);font-size:13px}.pill{border:1px solid #335b61;color:var(--accent);border-radius:999px;padding:7px 12px;font-size:12px}
main{max-width:1440px;margin:auto;padding:24px clamp(12px,3vw,40px) 48px}.intro{display:flex;justify-content:space-between;gap:24px;align-items:end;margin-bottom:20px;flex-wrap:wrap}.intro h2{margin:0;font-size:27px;letter-spacing:-.7px}.intro p{margin:6px 0 0;color:var(--muted);max-width:740px}.tabs{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}.tab{background:var(--panel);border:1px solid var(--line);color:var(--muted);padding:10px 13px;border-radius:10px;cursor:pointer}.tab.active{background:#18372f;border-color:#3b8e7b;color:var(--accent)}
.layout{display:grid;grid-template-columns:330px minmax(0,1fr);gap:18px}.panel{background:linear-gradient(180deg,rgba(20,29,50,.97),rgba(15,23,41,.97));border:1px solid var(--line);border-radius:16px;padding:18px;min-width:0}.panel h3{font-size:15px;margin:0 0 14px}.section{border-top:1px solid var(--line);padding-top:15px;margin-top:15px}.field{margin:12px 0}.field label{display:block;font-size:12px;color:var(--muted);margin-bottom:5px}.field input,.field select{width:100%;background:#0b1326;border:1px solid #33425f;border-radius:8px;padding:10px;color:var(--text);font:inherit}.twocol{display:grid;grid-template-columns:1fr 1fr;gap:10px}.btn{border:0;border-radius:9px;padding:11px 14px;font-weight:700;cursor:pointer;background:var(--accent);color:#07141b}.btn.secondary{background:#22304b;color:var(--text);border:1px solid #3a4b6d}.btnrow{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}.small{font-size:12px;color:var(--muted)}.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:14px}.metric{background:#0e172a;border:1px solid var(--line);border-radius:12px;padding:13px}.metric .k{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.6px}.metric .v{font-size:22px;font-weight:750;margin-top:5px}.metric .u{font-size:11px;color:var(--muted)}.charts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.chart{min-width:0;background:#0d1527;border:1px solid var(--line);border-radius:13px;padding:8px}.chart h4{margin:7px 9px 0;font-size:13px}.plot{height:290px;width:100%}.full{grid-column:1/-1}.status{margin:12px 0;padding:10px 12px;background:#102b27;border:1px solid #27594e;border-radius:9px;color:#b8f4e4;font-size:13px}.error{background:#351e27;border-color:#75404d;color:#ffd2da}.hidden{display:none!important}.tag{display:inline-block;border-radius:6px;padding:3px 7px;font-size:11px;background:#24304a;color:#b9c9e9;margin:2px}.report{white-space:pre-wrap;background:#0b1326;border:1px solid var(--line);border-radius:10px;padding:14px;color:#d9e4ff;font:12px/1.55 ui-monospace,monospace}
footer{color:var(--muted);font-size:12px;padding:22px 0 0;text-align:center}
@media(max-width:1000px){.layout{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,1fr)}}@media(max-width:650px){.charts{grid-template-columns:1fr}.full{grid-column:auto}.metric .v{font-size:19px}.plot{height:250px}}
</style></head>
<body>
<header><div class="brand"><div class="logo">N</div><div><h1>NeuroTune</h1><p>Control Systems Studio · Software-only engineering lab</p></div></div><div class="pill">EXPERIMENT 8 · PID DESIGN</div></header>
<main>
<div class="intro"><div><h2>Design. Simulate. Improve.</h2><p>A browser-based control engineering workspace combining PID tuning, time-domain metrics, stability analysis, root locus, Bode response, and block-diagram reduction.</p></div><div><span class="tag">Exp. 6</span><span class="tag">Exp. 7</span><span class="tag">Exp. 8</span><span class="tag">Exp. 9</span><span class="tag">Exp. 10</span></div></div>
<div class="tabs">
<button class="tab active" data-tab="design">PID Designer</button><button class="tab" data-tab="stability">Stability & Bode</button><button class="tab" data-tab="blocks">Block Reduction</button><button class="tab" data-tab="report">Report</button>
</div>
<div class="layout">
<aside class="panel">
<h3>System configuration</h3>
<div class="field"><label>Preset plant</label><select id="preset"><option value="second">Second-order plant · 1/(s² + 3s + 2)</option><option value="motor">Motor-like plant · 5/(s² + 2s)</option><option value="slow">Slow plant · 1/(s² + 0.6s + 1)</option><option value="custom">Custom transfer function</option></select></div>
<div class="field"><label>Numerator coefficients (highest power first)</label><input id="num" value="1" aria-label="Numerator coefficients"></div>
<div class="field"><label>Denominator coefficients (highest power first)</label><input id="den" value="1,3,2" aria-label="Denominator coefficients"></div>
<div class="section"><h3>PID gains</h3><div class="twocol">
<div class="field"><label>Kp</label><input id="kp" type="number" step=".1" value="3"></div><div class="field"><label>Ki</label><input id="ki" type="number" step=".1" value="1"></div><div class="field"><label>Kd</label><input id="kd" type="number" step=".01" value=".2"></div><div class="field"><label>Simulation duration (s)</label><input id="duration" type="number" min="1" max="60" step="1" value="10"></div></div>
<div class="field"><label>Target maximum overshoot (%)</label><input id="targetOS" type="number" min="0" max="80" value="5"></div>
<div class="field"><label>Target settling time (s)</label><input id="targetTs" type="number" min=".1" max="60" step=".1" value="2"></div>
<div class="btnrow"><button class="btn" id="simulate">Simulate system</button><button class="btn secondary" id="tune">Auto-tune PID</button></div>
<div class="small" style="margin-top:9px">Auto-tuning searches for stable gains that reduce tracking error and target violations. Results depend on the plant model.</div>
</div>
<div class="section"><h3>Block diagram reduction · Exp. 6</h3>
<div class="field"><label>Forward path G(s) coefficients (numerator)</label><input id="blockNum" value="1"></div>
<div class="field"><label>Forward path G(s) denominator</label><input id="blockDen" value="1,3,2"></div>
<div class="field"><label>Feedback H(s) (gain only)</label><input id="feedback" value="1"></div>
<div class="small">Unity/constant feedback: T(s) = G(s) / (1 + G(s)H(s)).</div><div class="btnrow"><button class="btn secondary" id="reduce">Reduce feedback loop</button></div><div id="reduceOut" class="small" style="margin-top:9px"></div>
</div>
</aside>
<section>
<div id="status" class="status">Ready. Select a plant, adjust PID gains, then simulate.</div>
<div id="designPanel">
<div class="metrics">
<div class="metric"><div class="k">Overshoot</div><div class="v" id="mOS">—</div><div class="u">percent</div></div>
<div class="metric"><div class="k">Settling time</div><div class="v" id="mTs">—</div><div class="u">2% band · seconds</div></div>
<div class="metric"><div class="k">Rise time</div><div class="v" id="mTr">—</div><div class="u">10–90% · seconds</div></div>
<div class="metric"><div class="k">Steady-state error</div><div class="v" id="mEss">—</div><div class="u">unit step</div></div>
</div>
<div class="charts"><div class="chart"><h4>Step response · open loop vs PID closed loop</h4><div id="stepPlot" class="plot"></div></div><div class="chart"><h4>Control-system poles</h4><div id="polePlot" class="plot"></div></div></div>
</div>
<div id="stabilityPanel" class="hidden">
<div class="metrics"><div class="metric"><div class="k">Closed-loop stability</div><div class="v" id="stable">—</div><div class="u">all poles left half-plane</div></div><div class="metric"><div class="k">Gain margin</div><div class="v" id="gm">—</div><div class="u">dB, approximate</div></div><div class="metric"><div class="k">Phase margin</div><div class="v" id="pm">—</div><div class="u">degrees, approximate</div></div><div class="metric"><div class="k">Pole count</div><div class="v" id="pcount">—</div><div class="u">closed-loop poles</div></div></div>
<div class="charts"><div class="chart"><h4>Bode magnitude</h4><div id="bodeMag" class="plot"></div></div><div class="chart"><h4>Bode phase</h4><div id="bodePhase" class="plot"></div></div><div class="chart full"><h4>Root locus · proportional gain sweep</h4><div id="rootPlot" class="plot"></div></div></div>
</div>
<div id="blocksPanel" class="hidden panel"><h3>Block diagram reduction</h3><p class="small">Reduce a forward path and constant feedback using the standard negative-feedback relation. This module is deliberately limited to constant feedback gains in this version.</p><div id="blockResult" class="report">Enter coefficients in the left panel and choose “Reduce feedback loop”.</div></div>
<div id="reportPanel" class="hidden panel"><h3>Simulation report</h3><p class="small">Generate a concise report from the latest successful simulation. Export as JSON or plain text.</p><div class="btnrow"><button class="btn" id="downloadJson">Export JSON</button><button class="btn secondary" id="downloadText">Export text report</button></div><div id="reportText" class="report" style="margin-top:12px">Run a simulation to populate the report.</div></div>
</section></div>
<footer>NeuroTune is an educational simulator, not a safety-certified engineering tool. Plotly is loaded from a public CDN; the app's calculations run locally on your machine after installation.</footer>
</main>
<script>
const $ = id => document.getElementById(id);
let latest = null;
document.querySelectorAll('.tab').forEach(b=>b.addEventListener('click',()=>{
 document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active',x===b));
 ['design','stability','blocks','report'].forEach(k=>$(k+'Panel').classList.toggle('hidden',k!==b.dataset.tab));
 if(b.dataset.tab==='stability' && latest) loadStability();
}));
$('preset').addEventListener('change',()=>{
 const p=$('preset').value;
 if(p==='second'){ $('num').value='1';$('den').value='1,3,2';}
 if(p==='motor'){ $('num').value='5';$('den').value='1,2,0';}
 if(p==='slow'){ $('num').value='1';$('den').value='1,0.6,1';}
});
function payload(){return {num:$('num').value,den:$('den').value,kp:+$('kp').value,ki:+$('ki').value,kd:+$('kd').value,duration:+$('duration').value,target_os:+$('targetOS').value,target_ts:+$('targetTs').value};}
async function api(path,data){let r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});let j=await r.json();if(!r.ok)throw new Error(j.error||'Request failed');return j;}
function status(msg,err=false){$('status').textContent=msg;$('status').classList.toggle('error',err);}
function fmt(x,n=2){return Number.isFinite(x)?x.toFixed(n):'—';}
function plot(id,data,layout={}){Plotly.react(id,data,{paper_bgcolor:'#0d1527',plot_bgcolor:'#0d1527',font:{color:'#c7d5ef',size:11},margin:{l:48,r:15,t:14,b:42},xaxis:{gridcolor:'#24314b',zerolinecolor:'#40506d'},yaxis:{gridcolor:'#24314b',zerolinecolor:'#40506d'},legend:{orientation:'h',y:1.12},...layout},{responsive:true,displaylogo:false});}
async function simulate(tuned=false){
 try{
 status(tuned?'Searching for controller gains…':'Simulating…');
 let d=await api(tuned?'/api/tune':'/api/simulate',payload());
 latest=d;
 $('mOS').textContent=fmt(d.metrics.overshoot,1)+'%';$('mTs').textContent=fmt(d.metrics.settling_time,2);$('mTr').textContent=fmt(d.metrics.rise_time,2);$('mEss').textContent=fmt(d.metrics.steady_state_error,4);
 if(tuned){$('kp').value=d.gains.kp;$('ki').value=d.gains.ki;$('kd').value=d.gains.kd;}
 plot('stepPlot',[{x:d.time,y:d.open_loop,name:'Open loop',mode:'lines',line:{color:'#8eb8ff',width:2}},{x:d.time,y:d.closed_loop,name:'PID closed loop',mode:'lines',line:{color:'#72e0c1',width:2}}],{xaxis:{title:'Time (s)',gridcolor:'#24314b'},yaxis:{title:'Output',gridcolor:'#24314b'}});
 const poles=d.poles;plot('polePlot',[{x:poles.map(p=>p[0]),y:poles.map(p=>p[1]),mode:'markers',type:'scatter',name:'Closed-loop poles',marker:{size:11,color:poles.map(p=>p[0]<0?'#72e0c1':'#ff8b9b'),symbol:'x'}}],{xaxis:{title:'Real',gridcolor:'#24314b',zeroline:true,zerolinecolor:'#8eb8ff'},yaxis:{title:'Imaginary',gridcolor:'#24314b',zeroline:true,zerolinecolor:'#8eb8ff',scaleanchor:'x',scaleratio:1}});
 $('reportText').textContent=makeReport(d);
 status(tuned?'Auto-tuning finished. Review the metrics and plots; verify performance for your chosen plant.':'Simulation complete. Switch to Stability & Bode for frequency-domain analysis.');
 }catch(e){status(e.message,true);}
}
$('simulate').onclick=()=>simulate(false);$('tune').onclick=()=>simulate(true);
async function loadStability(){try{let d=await api('/api/stability',payload());$('stable').textContent=d.stable?'STABLE':'UNSTABLE';$('stable').style.color=d.stable?'#72e0c1':'#ff8b9b';$('gm').textContent=d.gain_margin_db===null?'∞ / n.a.':fmt(d.gain_margin_db,1);$('pm').textContent=d.phase_margin_deg===null?'n.a.':fmt(d.phase_margin_deg,1);$('pcount').textContent=d.poles.length;
 plot('bodeMag',[{x:d.frequency,y:d.magnitude_db,mode:'lines',name:'Magnitude',line:{color:'#72e0c1',width:2}}],{xaxis:{type:'log',title:'Frequency (rad/s)',gridcolor:'#24314b'},yaxis:{title:'Magnitude (dB)',gridcolor:'#24314b'}});
 plot('bodePhase',[{x:d.frequency,y:d.phase_deg,mode:'lines',name:'Phase',line:{color:'#8eb8ff',width:2}}],{xaxis:{type:'log',title:'Frequency (rad/s)',gridcolor:'#24314b'},yaxis:{title:'Phase (degrees)',gridcolor:'#24314b'}});
 plot('rootPlot',d.root_locus.map((r,i)=>({x:r.map(p=>p[0]),y:r.map(p=>p[1]),mode:'lines+markers',name:'K sweep '+(i+1),line:{width:1},marker:{size:4}})),{xaxis:{title:'Real',gridcolor:'#24314b',zeroline:true,zerolinecolor:'#8eb8ff'},yaxis:{title:'Imaginary',gridcolor:'#24314b',zeroline:true,zerolinecolor:'#8eb8ff',scaleanchor:'x',scaleratio:1},showlegend:false});
 }catch(e){status(e.message,true);}}
$('reduce').onclick=async()=>{try{let d=await api('/api/reduce',{num:$('blockNum').value,den:$('blockDen').value,h:+$('feedback').value});$('reduceOut').textContent='Reduced transfer function ready.';$('blockResult').textContent='T(s) = G(s) / (1 + G(s)H(s))\\nNumerator: '+d.num.join(', ')+'\\nDenominator: '+d.den.join(', ');document.querySelector('[data-tab="blocks"]').click();}catch(e){status(e.message,true);}};
function makeReport(d){return `NEUROTUNE SIMULATION REPORT\\nGenerated: ${new Date().toLocaleString()}\\n\\nPLANT\\nG(s) numerator: ${d.plant.num.join(', ')}\\nG(s) denominator: ${d.plant.den.join(', ')}\\n\\nPID GAINS\\nKp: ${d.gains.kp}\\nKi: ${d.gains.ki}\\nKd: ${d.gains.kd}\\n\\nTIME-DOMAIN METRICS\\nOvershoot: ${fmt(d.metrics.overshoot,3)} %\\nSettling time (2%): ${fmt(d.metrics.settling_time,4)} s\\nRise time (10-90%): ${fmt(d.metrics.rise_time,4)} s\\nSteady-state error: ${fmt(d.metrics.steady_state_error,6)}\\n\\nCLOSED-LOOP POLES\\n${d.poles.map(p=>`${fmt(p[0],5)} ${p[1]>=0?'+':'-'} ${fmt(Math.abs(p[1]),5)}j`).join('\\n')}\\n\\nEducational result; validate assumptions before engineering use.`;}
$('downloadJson').onclick=()=>{if(!latest)return status('Run a simulation first.',true);download('neurotune-report.json',JSON.stringify(latest,null,2),'application/json');};
$('downloadText').onclick=()=>{if(!latest)return status('Run a simulation first.',true);download('neurotune-report.txt',makeReport(latest),'text/plain');};
function download(name,text,type){let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type}));a.download=name;a.click();URL.revokeObjectURL(a.href);}
simulate(false);
</script></body></html>
"""

def parse_coeffs(value, name):
    try:
        arr = [float(x.strip()) for x in str(value).replace(";", ",").split(",") if x.strip()]
    except Exception:
        raise ValueError(f"{name} must be comma-separated numbers.")
    if not arr or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite numeric coefficients.")
    while len(arr) > 1 and abs(arr[0]) < 1e-14:
        arr.pop(0)
    if len(arr) > 12:
        raise ValueError(f"{name} is limited to 12 coefficients.")
    return np.asarray(arr, dtype=float)

def get_plant(data):
    num = parse_coeffs(data.get("num", "1"), "Numerator")
    den = parse_coeffs(data.get("den", "1,3,2"), "Denominator")
    if len(den) < 2:
        raise ValueError("Denominator must describe a dynamic system (at least degree 1).")
    if den[0] == 0:
        raise ValueError("Leading denominator coefficient cannot be zero.")
    if len(num) > len(den):
        raise ValueError("Improper transfer functions are not supported in this version.")
    return num, den

def poly_add(a, b):
    n=max(len(a),len(b))
    aa=np.pad(a,(n-len(a),0)); bb=np.pad(b,(n-len(b),0))
    return aa+bb

def closed_loop(num, den, kp, ki, kd):
    # C(s) = Kp + Ki/s + Kd*s; unity negative feedback.
    # PID numerator over s is Kd*s^2 + Kp*s + Ki.
    cnum=np.array([kd,kp,ki],dtype=float)
    cden=np.array([1.0,0.0])
    ol_num=np.polymul(num,cnum)
    ol_den=np.polymul(den,cden)
    cl_den=poly_add(ol_den,ol_num)
    cl_num=ol_num
    # Normalize leading zeros from zero derivative/integral terms.
    while len(cl_num)>1 and abs(cl_num[0])<1e-12: cl_num=cl_num[1:]
    while len(cl_den)>1 and abs(cl_den[0])<1e-12: cl_den=cl_den[1:]
    return cl_num,cl_den

def simulate_tf(num, den, duration=10, points=1200):
    duration=float(np.clip(duration,1,60))
    t=np.linspace(0,duration,int(np.clip(points,200,3000)))
    sys=signal.TransferFunction(num,den)
    with np.errstate(all='ignore'):
        tout,y=signal.step(sys,T=t)
    y=np.asarray(y,dtype=float)
    y=np.nan_to_num(y,nan=0.0,posinf=1e8,neginf=-1e8)
    y=np.clip(y,-1e8,1e8)
    return tout,y

def metrics(t,y):
    final=float(y[-1]) if len(y) else 0
    peak=float(np.max(y)) if len(y) else 0
    os=max(0.0,(peak-final)/max(abs(final),1e-9)*100) if final>=0 else max(0.0,(final-peak)/max(abs(final),1e-9)*100)
    if abs(final)<1e-8:
        return {"overshoot":float(os),"settling_time":None,"rise_time":None,"steady_state_error":float(abs(1-final))}
    low,high=.1*final,.9*final
    if final<0: low,high=.9*final,.1*final
    lo=min(low,high); hi=max(low,high)
    ixlo=np.where(y>=lo)[0] if final>0 else np.where(y<=lo)[0]
    ixhi=np.where(y>=hi)[0] if final>0 else np.where(y<=hi)[0]
    rise=float(t[ixhi[0]]-t[ixlo[0]]) if len(ixlo) and len(ixhi) and ixhi[0]>=ixlo[0] else None
    band=max(abs(final)*.02,1e-5)
    outside=np.where(np.abs(y-final)>band)[0]
    settling=float(t[outside[-1]+1]) if len(outside) and outside[-1]+1<len(t) else (float(t[0]) if not len(outside) else None)
    return {"overshoot":float(os),"settling_time":settling,"rise_time":rise,"steady_state_error":float(abs(1-final))}

def simulate(data, tune=False):
    num,den=get_plant(data)
    duration=float(data.get("duration",10))
    if not np.isfinite(duration): duration=10
    kp=float(data.get("kp",3)); ki=float(data.get("ki",1)); kd=float(data.get("kd",.2))
    if not all(np.isfinite(v) for v in [kp,ki,kd]) or min(kp,ki,kd)<0:
        raise ValueError("PID gains must be finite and non-negative.")
    if tune:
        target_os=float(data.get("target_os",5)); target_ts=float(data.get("target_ts",2))
        if target_os<0 or target_ts<=0: raise ValueError("Targets must be non-negative and settling time positive.")
        def objective(x):
            p,i,d=np.exp(np.clip(x,-8,8))
            try:
                cn,cd=closed_loop(num,den,p,i,d)
                poles=np.roots(cd)
                if np.any(np.real(poles)>=-1e-5): return 1e5+np.sum(np.maximum(np.real(poles),0))*1e4
                t,y=simulate_tf(cn,cd,min(max(duration,5),20),600)
                m=metrics(t,y)
                if not np.all(np.isfinite(y)) or np.max(np.abs(y))>1e4: return 1e6
                ts=m["settling_time"] if m["settling_time"] is not None else duration*2
                rise=m["rise_time"] if m["rise_time"] is not None else duration*2
                ess=m["steady_state_error"]
                return (abs(m["overshoot"]-target_os)*.15 + max(0,m["overshoot"]-target_os)*.5
                        + max(0,ts-target_ts)*2 + rise*.2 + ess*15 + .001*(p+i+d))
            except Exception: return 1e6
        # Several deterministic initial points improve robustness without external services.
        starts=[np.log([max(kp,.1),max(ki,.1),max(kd,.01)]),np.log([1,1,.1]),np.log([5,2,.5])]
        best=None
        for x0 in starts:
            res=optimize.minimize(objective,x0,method="Nelder-Mead",options={"maxiter":250,"xatol":.03,"fatol":.02})
            if best is None or res.fun<best.fun: best=res
        kp,ki,kd=np.exp(np.clip(best.x,-8,8))
    cn,cd=closed_loop(num,den,kp,ki,kd)
    t,y=simulate_tf(cn,cd,duration)
    to,yo=simulate_tf(num,den,duration)
    poles=np.roots(cd)
    if not np.all(np.isfinite(y)): raise ValueError("Simulation became numerically unstable; try smaller gains or another plant.")
    return {"plant":{"num":num.tolist(),"den":den.tolist()},
            "gains":{"kp":float(kp),"ki":float(ki),"kd":float(kd)},
            "time":t.tolist(),"open_loop":yo.tolist(),"closed_loop":y.tolist(),
            "metrics":metrics(t,y),"poles":[[float(np.real(p)),float(np.imag(p))] for p in poles],
            "closed_loop_tf":{"num":cn.tolist(),"den":cd.tolist()}}

@app.route("/")
def index():
    return render_template_string(PAGE)

@app.post("/api/simulate")
def api_simulate():
    try: return jsonify(simulate(request.get_json(force=True)))
    except Exception as e: return jsonify(error=str(e)),400

@app.post("/api/tune")
def api_tune():
    try: return jsonify(simulate(request.get_json(force=True),True))
    except Exception as e: return jsonify(error=str(e)),400

@app.post("/api/stability")
def api_stability():
    try:
        data=request.get_json(force=True); result=simulate(data)
        num,den=get_plant(data); kp=result["gains"]["kp"];ki=result["gains"]["ki"];kd=result["gains"]["kd"]
        cn,cd=closed_loop(num,den,kp,ki,kd)
        stable=bool(np.all(np.real(np.roots(cd)) < -1e-8))
        # Bode of open-loop PID*plant.
        cnum=np.array([kd,kp,ki]); cden=np.array([1.,0.])
        ol_num=np.polymul(num,cnum); ol_den=np.polymul(den,cden)
        while len(ol_num)>1 and abs(ol_num[0])<1e-12: ol_num=ol_num[1:]
        while len(ol_den)>1 and abs(ol_den[0])<1e-12: ol_den=ol_den[1:]
        w=np.logspace(-2,3,500)
        _,mag,phase=signal.bode(signal.TransferFunction(ol_num,ol_den),w=w)
        # Approximate margins from phase/gain crossover interpolation.
        phase_unwrap=np.unwrap(np.deg2rad(phase))*180/np.pi
        pm=None; gm=None
        gc=np.where(np.diff(np.sign(mag))!=0)[0]
        if len(gc):
            j=gc[0]; wc=w[j]+(w[j+1]-w[j])*(0-mag[j])/(mag[j+1]-mag[j])
            ph=phase_unwrap[j]+(phase_unwrap[j+1]-phase_unwrap[j])*(wc-w[j])/(w[j+1]-w[j])
            pm=float(180+ph)
        pc=np.where(np.diff(np.sign(phase_unwrap+180))!=0)[0]
        if len(pc):
            j=pc[0]; wc=w[j]+(w[j+1]-w[j])*(-180-phase_unwrap[j])/(phase_unwrap[j+1]-phase_unwrap[j])
            mg=mag[j]+(mag[j+1]-mag[j])*(wc-w[j])/(w[j+1]-w[j]); gm=float(-mg)
        root=[]
        for kval in np.concatenate(([0.0],np.logspace(-2,3,24))):
            rnum=np.polymul(num,[kval])
            rden=poly_add(den,rnum)
            root.append([[float(p.real),float(p.imag)] for p in np.roots(rden)])
        return jsonify({"stable":stable,"poles":result["poles"],"frequency":w.tolist(),
          "magnitude_db":mag.tolist(),"phase_deg":phase_unwrap.tolist(),
          "gain_margin_db":gm,"phase_margin_deg":pm,"root_locus":root})
    except Exception as e: return jsonify(error=str(e)),400

@app.post("/api/reduce")
def api_reduce():
    try:
        d=request.get_json(force=True); num=parse_coeffs(d.get("num","1"),"Numerator"); den=parse_coeffs(d.get("den","1,3,2"),"Denominator"); h=float(d.get("h",1))
        if not np.isfinite(h): raise ValueError("Feedback gain must be finite.")
        # T=G/(1+GH), T numerator = N; denominator = D + H*N
        td=poly_add(den,h*num)
        return jsonify({"num":num.tolist(),"den":td.tolist()})
    except Exception as e: return jsonify(error=str(e)),400

@app.get("/health")
def health(): return jsonify(status="ok",app="NeuroTune")

if __name__ == "__main__":
    app.run(host="127.0.0.1",port=5000,debug=False)
