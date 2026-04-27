"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   HYDRO-MODELING PLATFORM  v2.0                                             ║
║   Tamchachate Catchment · Haut Atlas, Morocco                               ║
║   Streamlit + Plotly + Folium · Upgraded Professional Edition               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from scipy.optimize import differential_evolution

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hydro-Modeling Platform",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────
if "show_landing"    not in st.session_state: st.session_state.show_landing    = True
if "active_page"     not in st.session_state: st.session_state.active_page     = "Dashboard"
if "user_run_done"   not in st.session_state: st.session_state.user_run_done   = False
if "user_results"    not in st.session_state: st.session_state.user_results    = None
if "selected_model"  not in st.session_state: st.session_state.selected_model  = "HYMOD"

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — Syne + Space Mono, deep navy/slate dark theme
# ─────────────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
<style>
:root {
    --bg-deep:      #060d18;
    --bg-panel:     #0b1626;
    --bg-card:      #0f1e33;
    --bg-card-hi:   #132540;
    --border:       #1e3a5f;
    --border-hi:    #2e5f9e;
    --accent:       #1a8cff;
    --accent-dim:   #0f5fad;
    --accent-glow:  rgba(26,140,255,0.18);
    --teal:         #00d4b4;
    --text-primary: #e8f1fb;
    --text-secondary:#8aabbf;
    --text-muted:   #4a6a85;
    --good:   #22c55e;
    --ok:     #f59e0b;
    --warn:   #f97316;
    --bad:    #ef4444;
    --font-head: 'Syne', sans-serif;
    --font-mono: 'Space Mono', monospace;
    --font-body: 'DM Sans', sans-serif;
    --radius: 10px;
    --shadow: 0 4px 24px rgba(0,0,0,0.45);
}
html, body, [class*="css"] { font-family: var(--font-body) !important;
    background-color: var(--bg-deep) !important; color: var(--text-primary) !important; }
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background: var(--bg-panel); }
::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius:3px; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#070f1c 0%,#0b1828 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
section[data-testid="stSidebar"] > div { padding-top: 1.2rem; }

.main .block-container { padding-top:1.5rem !important; padding-bottom:3rem !important; max-width:1400px !important; }
h1,h2,h3,h4 { font-family: var(--font-head) !important; }

.hmp-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 20px 22px; box-shadow: var(--shadow);
    transition: border-color .2s, box-shadow .2s;
}
.hmp-card:hover { border-color: var(--border-hi); box-shadow: 0 6px 32px rgba(26,140,255,0.1); }

.metric-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-top: 3px solid var(--accent); border-radius: var(--radius);
    padding: 18px 16px 14px; text-align: center; box-shadow: var(--shadow);
    transition: transform .18s, border-color .18s;
}
.metric-card:hover { transform: translateY(-2px); border-color: var(--accent); }
.metric-card .mv { font-family: var(--font-head); font-size:2rem; font-weight:800;
                    color: var(--accent); line-height:1.15; }
.metric-card .ml { font-size:0.72rem; color:var(--text-secondary);
                    text-transform:uppercase; letter-spacing:0.1em; margin-top:5px; }
.metric-card .ms { font-size:0.8rem; margin-top:6px; }
.metric-card .mt { font-size:0.68rem; color:var(--text-muted); margin-top:3px; }
.metric-card .mi { font-size:1.4rem; color:var(--accent); margin-bottom:6px; opacity:0.8; }
.metric-card.teal { border-top-color: var(--teal); }
.metric-card.teal .mv, .metric-card.teal .mi { color: var(--teal); }

.sec-hdr {
    display:flex; align-items:center; gap:12px; padding:10px 18px;
    background: linear-gradient(90deg, var(--accent-glow) 0%, transparent 100%);
    border-left: 3px solid var(--accent);
    border-radius: 0 var(--radius) var(--radius) 0; margin: 22px 0 14px;
}
.sec-hdr i { color: var(--accent); font-size:1rem; }
.sec-hdr span { font-family: var(--font-head); font-size:1rem; font-weight:700;
                 color: var(--text-primary); letter-spacing:0.02em; text-transform:uppercase; }

.badge { display:inline-block; padding:3px 12px; border-radius:20px;
          font-size:0.75rem; font-weight:600; letter-spacing:0.05em; }
.badge-good { background:rgba(34,197,94,0.15);  color:var(--good); border:1px solid rgba(34,197,94,0.3); }
.badge-ok   { background:rgba(245,158,11,0.15); color:var(--ok);   border:1px solid rgba(245,158,11,0.3); }
.badge-warn { background:rgba(249,115,22,0.15); color:var(--warn); border:1px solid rgba(249,115,22,0.3); }
.badge-bad  { background:rgba(239,68,68,0.15);  color:var(--bad);  border:1px solid rgba(239,68,68,0.3); }

.info-block {
    background: rgba(26,140,255,0.08); border-left:3px solid var(--accent);
    border-radius: 0 var(--radius) var(--radius) 0; padding:12px 16px;
    font-size:0.88rem; color:var(--text-secondary); margin:10px 0;
}

.stTabs [data-baseweb="tab-list"] { gap:6px; background:transparent; border-bottom:1px solid var(--border); }
.stTabs [data-baseweb="tab"] {
    background:transparent; color:var(--text-secondary); font-family:var(--font-head);
    font-weight:600; font-size:0.82rem; letter-spacing:0.06em; text-transform:uppercase;
    border-bottom:2px solid transparent; padding:8px 18px; border-radius:0;
}
.stTabs [aria-selected="true"] {
    background:transparent !important; color:var(--accent) !important;
    border-bottom:2px solid var(--accent) !important;
}
.stButton > button {
    background: linear-gradient(135deg,var(--accent) 0%,var(--accent-dim) 100%) !important;
    color: white !important; border: none !important; border-radius: var(--radius) !important;
    font-family: var(--font-head) !important; font-weight:700 !important;
    letter-spacing:0.04em !important; padding:0.55rem 1.5rem !important;
    transition: opacity .2s, transform .15s !important;
}
.stButton > button:hover { opacity:.88 !important; transform:translateY(-1px) !important; }
.stButton > button[kind="secondary"] {
    background: var(--bg-card) !important; border: 1px solid var(--border-hi) !important;
    color: var(--accent) !important;
}
.stDownloadButton > button {
    background: linear-gradient(135deg,var(--teal) 0%,#009e88 100%) !important;
    color: #060d18 !important; border:none !important; font-weight:700 !important;
    border-radius: var(--radius) !important;
}
.stAlert { border-radius: var(--radius) !important; border:1px solid var(--border) !important; }
hr { border-color: var(--border) !important; margin:24px 0 !important; }
.stExpander { border:1px solid var(--border) !important; border-radius: var(--radius) !important; }
.stFileUploader { border:2px dashed var(--border-hi) !important;
                   border-radius: var(--radius) !important; background: var(--bg-card) !important; }
input[type="text"], input[type="number"], textarea {
    background: var(--bg-card) !important; border:1px solid var(--border) !important;
    color: var(--text-primary) !important; border-radius:6px !important;
}
</style>
""", unsafe_allow_html=True)



# ═════════════════════════════════════════════════════════════════════════════
# SECTION I — LANDING PAGE
# ═════════════════════════════════════════════════════════════════════════════
def show_landing():
    st.markdown("""
    <style>
    [data-testid="stSidebar"] { display:none !important; }
    .main .block-container { padding:0 !important; max-width:100% !important; }
    header[data-testid="stHeader"] { display:none; }
    #land {
        min-height:100vh; display:flex; align-items:center; justify-content:center;
        background:
            radial-gradient(ellipse 80% 60% at 50% 0%,  rgba(26,140,255,0.20) 0%, transparent 70%),
            radial-gradient(ellipse 40% 40% at 80% 80%, rgba(0,212,180,0.12)  0%, transparent 60%),
            linear-gradient(160deg,#030912 0%,#060d1a 40%,#081525 100%);
        position:relative; overflow:hidden;
    }
    #land::before {
        content:''; position:absolute; inset:0;
        background:
            repeating-linear-gradient(0deg,  transparent, transparent 120px,
                rgba(26,140,255,0.025) 120px, rgba(26,140,255,0.025) 121px),
            repeating-linear-gradient(90deg, transparent, transparent 120px,
                rgba(26,140,255,0.018) 120px, rgba(26,140,255,0.018) 121px);
        pointer-events:none;
    }
    #land-inner { text-align:center; z-index:1; padding:60px 40px; max-width:780px;
                   animation:fadeUp .9s ease both; }
    @keyframes fadeUp { from{opacity:0;transform:translateY(36px)} to{opacity:1;transform:translateY(0)} }
    .l-eye { font-family:'Space Mono',monospace; font-size:0.75rem; letter-spacing:0.25em;
               color:#1a8cff; text-transform:uppercase; margin-bottom:22px;
               animation:fadeUp .9s 0.1s ease both; }
    .l-title { font-family:'Syne',sans-serif; font-size:clamp(2.6rem,6vw,4.4rem);
                font-weight:800; line-height:1.03; color:#ffffff; letter-spacing:-0.03em;
                animation:fadeUp .9s 0.2s ease both; }
    .l-title .ac { color:#1a8cff; }
    .l-sub { font-family:'DM Sans',sans-serif; font-size:1.05rem; color:#8aabbf;
              margin:22px auto 0; max-width:520px; line-height:1.75;
              animation:fadeUp .9s 0.35s ease both; }
    .l-chips { display:flex; gap:10px; justify-content:center; flex-wrap:wrap;
                margin:28px 0 38px; animation:fadeUp .9s 0.5s ease both; }
    .l-chip { padding:6px 16px; border-radius:20px;
               background:rgba(26,140,255,0.12); border:1px solid rgba(26,140,255,0.28);
               color:#a8d4f0; font-size:0.78rem; font-family:'Space Mono',monospace; }
    .l-foot { margin-top:50px; font-size:0.7rem; color:#1e3a5f;
               font-family:'Space Mono',monospace; animation:fadeUp .9s 0.8s ease both; }
    .orb { position:absolute; border-radius:50%; filter:blur(80px);
            pointer-events:none; opacity:0.3; }
    .orb1 { width:500px; height:500px; background:#1a8cff; top:-200px; left:-150px; }
    .orb2 { width:400px; height:400px; background:#00d4b4; bottom:-150px; right:-100px; }
    .orb3 { width:300px; height:300px; background:#1a8cff; top:40%; right:20%; opacity:0.12; }
    </style>
    <div id="land">
        <div class="orb orb1"></div><div class="orb orb2"></div><div class="orb orb3"></div>
        <div id="land-inner">
            <div class="l-eye"><i class="fa-solid fa-water"></i> &nbsp; Hydrological Analysis Suite</div>
            <div class="l-title">HYDRO<span class="ac">-</span>MODELING<br>PLATFORM</div>
            <div class="l-sub">Interactive rainfall–runoff modeling and hydrological analysis.
                Calibrate, validate, and simulate catchment responses with precision.</div>
            <div class="l-chips">
                <span class="l-chip">HYMOD Model</span>
                <span class="l-chip">Diff. Evolution</span>
                <span class="l-chip">NSE · KGE · R²</span>
                <span class="l-chip">Interactive Maps</span>
                <span class="l-chip">Custom Data Upload</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    col_l, col_c, col_r = st.columns([2,1,2])
    with col_c:
        if st.button("Enter Platform  →", key="enter_btn", use_container_width=True):
            st.session_state.show_landing = False
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION II — HYDROLOGICAL ENGINE
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_data
def generate_data():
    A_km2=128.86; CONV=A_km2*1e6/(1000.0*86400.0)
    np.random.seed(42)
    dates=pd.date_range('2000-01-01','2019-12-31',freq='D'); n=len(dates)
    doy=np.array([d.timetuple().tm_yday for d in dates])
    P_seas=np.maximum(2.5+2.0*np.cos(2*np.pi*(doy-15)/365),0.3)
    P=np.zeros(n)
    for i in range(n):
        if np.random.rand()<np.clip(0.18*P_seas[i]/2.5,0.05,0.45):
            P[i]=np.random.exponential(P_seas[i]*1.8)
    ext=np.random.choice(n,25,replace=False); P[ext]*=np.random.uniform(3,8,25)
    P=np.clip(P,0,80)
    PET=np.maximum(3.5+2.0*np.sin(2*np.pi*(doy-100)/365)+0.2*np.random.randn(n),0.8)
    Cmax_v,Bexp_v,Alpha_v,Ks_v,Kq_v=150.0,1.5,0.55,0.006,0.45
    Soil=Cmax_v*0.3; Slow=0.0; Quick=np.zeros(3); Qmm=np.zeros(n)
    for t in range(n):
        r=min(Soil/Cmax_v,1.0); Soil=max(Soil-PET[t]*r,0.0); Sn=Soil+P[t]
        if Sn>=Cmax_v: exc=Sn-Cmax_v; Soil=Cmax_v
        else:
            r0=min(Soil/Cmax_v,1.0); r1=min(Sn/Cmax_v,1.0)
            exc=P[t]*max(r1-r0+(1-r1)**2.5-(1-r0)**2.5,0.0); Soil=Sn-exc
        UR=Alpha_v*exc; US=(1-Alpha_v)*exc
        Slow+=US; Qs=Ks_v*Slow; Slow-=Qs
        Quick[0]+=UR; q0=Kq_v*Quick[0]; Quick[0]-=q0
        Quick[1]+=q0; q1=Kq_v*Quick[1]; Quick[1]-=q1
        Quick[2]+=q1; qq=Kq_v*Quick[2]; Quick[2]-=qq; Qmm[t]=Qs+qq
    noise=1+0.08*np.random.randn(n); snow=1+0.5*np.exp(-((doy-100)**2)/(2*20**2))
    years=np.array([d.year for d in dates]); np.random.seed(10)
    yr_f={y:1+0.3*np.random.randn() for y in range(2000,2020)}
    yr_arr=np.array([yr_f[y] for y in years])
    Qobs=Qmm*np.abs(noise)*snow*np.maximum(yr_arr,0.3)*CONV
    df=pd.DataFrame({'Date':dates,'P':np.round(P,2),'PET':np.round(PET,2),'Qobs':np.round(Qobs,4)})
    df['Qobs_mm']=df['Qobs']/CONV
    return df,CONV


def _run_hymod_np(params,P,PET):
    Cmax,Bexp,Alpha,Ks,Kq=params; n=len(P); Qmm=np.zeros(n)
    Soil=Cmax*0.3; Slow=0.0; Quick=np.zeros(3)
    for t in range(n):
        r=min(Soil/Cmax,1.0); Soil=max(Soil-PET[t]*r,0.0); Sn=Soil+P[t]
        if Sn>=Cmax: exc=Sn-Cmax; Soil=Cmax
        else:
            r0=min(Soil/Cmax,1.0); r1=min(Sn/Cmax,1.0)
            exc=P[t]*max(r1-r0+(1-r1)**(Bexp+1)-(1-r0)**(Bexp+1),0.0); Soil=Sn-exc
        UR=Alpha*exc; US=(1-Alpha)*exc
        Slow+=US; Qs=Ks*Slow; Slow-=Qs
        Quick[0]+=UR; q0=Kq*Quick[0]; Quick[0]-=q0
        Quick[1]+=q0; q1=Kq*Quick[1]; Quick[1]-=q1
        Quick[2]+=q1; qq=Kq*Quick[2]; Quick[2]-=qq; Qmm[t]=Qs+qq
    return Qmm


@st.cache_data
def run_hymod(params,P,PET):
    return _run_hymod_np(params,np.asarray(P,dtype=float),np.asarray(PET,dtype=float))


@st.cache_data
def calibrate_hymod(P_run,PET_run,Qobs_cal_mm,n_warmup):
    P_a=np.asarray(P_run,dtype=float); PET_a=np.asarray(PET_run,dtype=float)
    Qo_a=np.asarray(Qobs_cal_mm,dtype=float)
    bounds=[(1.0,500.0),(0.1,5.0),(0.01,0.99),(0.0001,0.1),(0.1,0.99)]
    def obj(p):
        Qs=_run_hymod_np(p,P_a,PET_a)
        nse=1.0-np.sum((Qo_a-Qs[n_warmup:])**2)/np.sum((Qo_a-Qo_a.mean())**2)
        return -nse if not(np.isnan(nse) or np.isinf(nse)) else 9999.0
    res=differential_evolution(obj,bounds,seed=42,maxiter=800,popsize=15,
                                 tol=1e-7,mutation=(0.5,1.5),recombination=0.7,
                                 polish=True,workers=1,disp=False)
    return res.x


# Multi-model registry (extensible)
MODEL_REGISTRY = {
    "HYMOD":    {"label":"HYMOD (Active)","icon":"fa-droplet","status":"active",
                  "desc":"PDM-based bucket model · 5 parameters · Chaponnière 2008"},
    "HBV-Light":{"label":"HBV-Light (Coming Soon)","icon":"fa-mountain","status":"coming",
                  "desc":"Semi-distributed conceptual model · Bergström 1976"},
    "GR4J":     {"label":"GR4J (Coming Soon)","icon":"fa-chart-area","status":"coming",
                  "desc":"4-parameter daily lumped model · Perrin et al. 2003"},
}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION III — METRICS
# ═════════════════════════════════════════════════════════════════════════════
def calc_NSE(o,s):   return 1.0-np.sum((o-s)**2)/np.sum((o-o.mean())**2)
def calc_RMSE(o,s):  return np.sqrt(np.mean((o-s)**2))
def calc_KGE(o,s):
    r=np.corrcoef(o,s)[0,1]; a=s.std()/max(o.std(),1e-10); b=s.mean()/max(o.mean(),1e-10)
    return 1.0-np.sqrt((r-1)**2+(a-1)**2+(b-1)**2)
def calc_R2(o,s):    return np.corrcoef(o,s)[0,1]**2
def calc_PBIAS(o,s): return 100.0*np.sum(o-s)/np.sum(o)
def calc_MAE(o,s):   return np.mean(np.abs(o-s))
def calc_NSElog(o,s):
    eps=1e-6; lo=np.log(np.maximum(o,eps)); ls=np.log(np.maximum(s,eps))
    return 1.0-np.sum((lo-ls)**2)/np.sum((lo-lo.mean())**2)

def compute_metrics(obs,sim):
    return {'NSE':calc_NSE(obs,sim),'RMSE':calc_RMSE(obs,sim),'KGE':calc_KGE(obs,sim),
            'R2':calc_R2(obs,sim),'PBIAS':calc_PBIAS(obs,sim),'MAE':calc_MAE(obs,sim),
            'NSE_log':calc_NSElog(obs,sim)}

def nse_quality(nse):
    if nse>0.75: return "Very Good","good"
    if nse>0.65: return "Good","ok"
    if nse>0.50: return "Satisfactory","warn"
    return "Poor","bad"

def quality_badge(nse):
    lbl,cls=nse_quality(nse)
    return f'<span class="badge badge-{cls}">{lbl}</span>'


# ═════════════════════════════════════════════════════════════════════════════
# SECTION IV — UI HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def sec_header(icon,title):
    st.markdown(f'<div class="sec-hdr"><i class="fa-solid {icon}"></i>'
                f'<span>{title}</span></div>',unsafe_allow_html=True)

def metric_card(col,icon,value,label,subtitle="",target="",teal=False):
    cls="metric-card teal" if teal else "metric-card"
    col.markdown(f"""
    <div class="{cls}">
        <div class="mi"><i class="fa-solid {icon}"></i></div>
        <div class="mv">{value}</div>
        <div class="ml">{label}</div>
        {f'<div class="ms">{subtitle}</div>' if subtitle else ''}
        {f'<div class="mt">Target: {target}</div>' if target else ''}
    </div>""",unsafe_allow_html=True)

def plotly_base(height=420):
    return dict(
        height=height,template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(11,22,38,0.85)",
        font=dict(family="DM Sans,sans-serif",color="#8aabbf",size=11),
        margin=dict(l=12,r=12,t=44,b=12),
        legend=dict(bgcolor="rgba(0,0,0,0)",bordercolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(30,58,95,0.6)",zeroline=False),
        yaxis=dict(gridcolor="rgba(30,58,95,0.6)",zeroline=False),
    )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION V — SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
def sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding:12px 4px 20px;border-bottom:1px solid #1e3a5f;margin-bottom:16px;">
            <div style="font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:800;color:#e8f1fb;">
                <i class="fa-solid fa-water" style="color:#1a8cff;margin-right:8px;"></i>Hydro-Modeling
            </div>
            <div style="font-size:0.7rem;color:#4a6a85;font-family:'Space Mono',monospace;
                        margin-top:4px;letter-spacing:0.06em;">PLATFORM v2.0</div>
        </div>""",unsafe_allow_html=True)

        st.markdown('<div style="font-size:0.68rem;color:#4a6a85;text-transform:uppercase;'
                    'letter-spacing:0.1em;margin-bottom:6px;">Active Model</div>',unsafe_allow_html=True)
        model_choice=st.selectbox("model",list(MODEL_REGISTRY.keys()),
                                   format_func=lambda k:MODEL_REGISTRY[k]["label"],
                                   label_visibility="collapsed",key="model_sel")
        st.session_state.selected_model=model_choice
        if MODEL_REGISTRY[model_choice]["status"]=="coming":
            st.info("Coming soon. HYMOD is currently active.")

        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.68rem;color:#4a6a85;text-transform:uppercase;'
                    'letter-spacing:0.1em;margin-bottom:8px;">Navigation</div>',unsafe_allow_html=True)

        pages=[
            ("Dashboard","fa-gauge-high","Overview & KPIs"),
            ("Interactive Map","fa-map-location-dot","Basin & Stations"),
            ("Model Results","fa-chart-line","Calibration & Validation"),
            ("Simulation Tool","fa-sliders","Parameter Sensitivity"),
            ("Model Theory","fa-book-open","HYMOD Structure"),
            ("My Own Data","fa-upload","Custom Basin Analysis"),
        ]
        for pname,icon,desc in pages:
            is_active=st.session_state.active_page==pname
            style=("background:rgba(26,140,255,0.12);color:#1a8cff!important;"
                   "border-left:2px solid #1a8cff;" if is_active else "")
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;padding:9px 14px;margin:2px 0;
                        border-radius:8px;cursor:pointer;font-size:0.86rem;
                        color:{'#1a8cff' if is_active else '#8aabbf'};{style}">
                <i class="fa-solid {icon}" style="width:16px;text-align:center;font-size:0.82rem;"></i>
                {pname}
            </div>""",unsafe_allow_html=True)
            if st.button(pname,key=f"nav_{pname}",help=desc,use_container_width=True,
                          label_visibility="collapsed"):
                st.session_state.active_page=pname; st.rerun()

        st.markdown("""
        <div style="border-top:1px solid #1e3a5f;padding-top:14px;margin-top:12px;
                    font-size:0.74rem;color:#2d4a62;">
            <div style="color:#4a6a85;margin-bottom:8px;font-family:'Space Mono',monospace;
                        font-size:0.66rem;text-transform:uppercase;letter-spacing:0.08em;">
                Tamchachate · Haut Atlas
            </div>
            <div style="line-height:2.0;color:#3a5a72;">
                128.86 km² &nbsp;·&nbsp; 32.15°N / 5.50°W<br>
                ~2050 m elevation<br>
                Semi-arid nivo-pluvial<br>
                2000 – 2019 · Daily
            </div>
        </div>""",unsafe_allow_html=True)
    return st.session_state.active_page


# ═════════════════════════════════════════════════════════════════════════════
# SECTION VI — LOAD DEMO DATA
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_and_calibrate():
    df,CONV=generate_data()
    mask_warmup=df['Date']<='2001-12-31'
    mask_train=(df['Date']>'2001-12-31')&(df['Date']<='2013-12-31')
    mask_test=df['Date']>='2014-01-01'
    n_warmup=int(mask_warmup.sum())
    P_run=tuple(df.loc[mask_warmup|mask_train,'P'].values)
    PET_run=tuple(df.loc[mask_warmup|mask_train,'PET'].values)
    Qo_cal=tuple(df.loc[mask_train,'Qobs_mm'].values)
    params=calibrate_hymod(P_run,PET_run,Qo_cal,n_warmup)
    Qsim_mm=run_hymod(tuple(params),tuple(df['P'].values),tuple(df['PET'].values))
    df['Qsim']=Qsim_mm*CONV; df['Qsim_mm']=Qsim_mm
    met_train=compute_metrics(df.loc[mask_train,'Qobs'].values,df.loc[mask_train,'Qsim'].values)
    met_test =compute_metrics(df.loc[mask_test, 'Qobs'].values,df.loc[mask_test, 'Qsim'].values)
    return df,CONV,params,met_train,met_test,mask_warmup,mask_train,mask_test


# ═════════════════════════════════════════════════════════════════════════════
# SECTION VII — PAGE: DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
def page_dashboard(df,CONV,params,met_train,met_test,mask_train,mask_test):
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0b1626 0%,#0f2540 60%,#132e55 100%);
                border:1px solid #1e3a5f;border-radius:14px;padding:32px 36px 28px;
                margin-bottom:28px;position:relative;overflow:hidden;">
        <div style="position:absolute;top:-60px;right:-60px;width:280px;height:280px;
                    background:rgba(26,140,255,0.06);border-radius:50%;pointer-events:none;"></div>
        <div style="position:relative;">
            <div style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#1a8cff;
                        letter-spacing:0.2em;text-transform:uppercase;margin-bottom:10px;">
                <i class="fa-solid fa-water"></i> &nbsp; Hydro-Modeling Platform
            </div>
            <h1 style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;
                       color:#e8f1fb;margin:0 0 8px;letter-spacing:-0.02em;">
                HYMOD — Tamchachate Catchment
            </h1>
            <p style="color:#8aabbf;margin:0 0 20px;font-size:0.95rem;max-width:560px;">
                Haut Atlas Mountains, Morocco &nbsp;·&nbsp; Daily rainfall-runoff modeling &nbsp;·&nbsp;
                Differential Evolution calibration
            </p>
            <div style="display:flex;gap:10px;flex-wrap:wrap;">
                <span style="background:rgba(26,140,255,0.12);border:1px solid rgba(26,140,255,0.25);
                             border-radius:20px;padding:5px 14px;color:#a8d4f0;font-size:0.78rem;
                             font-family:'Space Mono',monospace;">128.86 km²</span>
                <span style="background:rgba(26,140,255,0.12);border:1px solid rgba(26,140,255,0.25);
                             border-radius:20px;padding:5px 14px;color:#a8d4f0;font-size:0.78rem;
                             font-family:'Space Mono',monospace;">2000–2019 · 7305 days</span>
                <span style="background:rgba(26,140,255,0.12);border:1px solid rgba(26,140,255,0.25);
                             border-radius:20px;padding:5px 14px;color:#a8d4f0;font-size:0.78rem;
                             font-family:'Space Mono',monospace;">5 Parameters</span>
                <span style="background:rgba(26,140,255,0.12);border:1px solid rgba(26,140,255,0.25);
                             border-radius:20px;padding:5px 14px;color:#a8d4f0;font-size:0.78rem;
                             font-family:'Space Mono',monospace;">DE Calibration</span>
            </div>
        </div>
    </div>""",unsafe_allow_html=True)

    sec_header("fa-circle-check","Calibration Performance — 2002 · 2013")
    c1,c2,c3,c4,c5=st.columns(5)
    metric_card(c1,"fa-chart-bar",f"{met_train['NSE']:.3f}","NSE",quality_badge(met_train['NSE']),"> 0.50")
    metric_card(c2,"fa-bullseye",f"{met_train['KGE']:.3f}","KGE","Kling-Gupta","> 0.50")
    metric_card(c3,"fa-square-root-variable",f"{met_train['R2']:.3f}","R²","Pearson²","> 0.60")
    metric_card(c4,"fa-percent",f"{met_train['PBIAS']:.1f}%","PBIAS","Volume bias","< 25%")
    metric_card(c5,"fa-ruler",f"{met_train['RMSE']:.4f}","RMSE","m³/s","→ 0",teal=True)

    sec_header("fa-flask","Validation Performance — 2014 · 2019")
    c1,c2,c3,c4,c5=st.columns(5)
    metric_card(c1,"fa-chart-bar",f"{met_test['NSE']:.3f}","NSE",quality_badge(met_test['NSE']),"> 0.50")
    metric_card(c2,"fa-bullseye",f"{met_test['KGE']:.3f}","KGE","Kling-Gupta","> 0.50")
    metric_card(c3,"fa-square-root-variable",f"{met_test['R2']:.3f}","R²","Pearson²","> 0.60")
    metric_card(c4,"fa-percent",f"{met_test['PBIAS']:.1f}%","PBIAS","Volume bias","< 25%")
    metric_card(c5,"fa-ruler",f"{met_test['RMSE']:.4f}","RMSE","m³/s","→ 0",teal=True)

    delta=met_train['NSE']-met_test['NSE']
    if delta<0.10:   gen='<span class="badge badge-good"><i class="fa-solid fa-check"></i> Excellent generalisation</span>'
    elif delta<0.20: gen='<span class="badge badge-ok"><i class="fa-solid fa-triangle-exclamation"></i> Slight over-fitting</span>'
    else:            gen='<span class="badge badge-bad"><i class="fa-solid fa-xmark"></i> Over-calibration detected</span>'
    st.markdown(f'<div class="info-block"><i class="fa-solid fa-code-compare" style="color:#1a8cff;'
                f'margin-right:8px;"></i>Generalisation check: {gen} &nbsp; ΔNSE = <b>{delta:.3f}</b>'
                f'</div>',unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    col_left,col_right=st.columns([3,1])
    with col_left:
        sec_header("fa-wave-square","Full Period Hydrograph — 2000 · 2019")
        fig=make_subplots(rows=2,cols=1,shared_xaxes=True,
                           row_heights=[0.22,0.78],vertical_spacing=0.03)
        fig.add_trace(go.Bar(x=df['Date'],y=df['P'],name='Precipitation',
                              marker_color='#3a7cbf',opacity=0.7),row=1,col=1)
        fig.add_vrect(x0=df.loc[mask_train,'Date'].min(),x1=df.loc[mask_train,'Date'].max(),
                       fillcolor='rgba(26,140,255,0.05)',line_width=0,row=2,col=1)
        fig.add_vrect(x0=df.loc[mask_test,'Date'].min(),x1=df.loc[mask_test,'Date'].max(),
                       fillcolor='rgba(0,212,180,0.04)',line_width=0,row=2,col=1)
        fig.add_trace(go.Scatter(x=df['Date'],y=df['Qobs'],name='Observed Q',
                                  line=dict(color='#4aa3e8',width=1.0),opacity=0.9),row=2,col=1)
        fig.add_trace(go.Scatter(x=df['Date'],y=df['Qsim'],name='Simulated Q',
                                  line=dict(color='#ff6b6b',width=1.0,dash='dash'),opacity=0.85),row=2,col=1)
        fig.add_annotation(x=df.loc[mask_train,'Date'].mean(),y=0,yref='paper',
                            text="CALIBRATION",showarrow=False,yanchor='bottom',
                            font=dict(color='rgba(26,140,255,0.4)',size=9,family='Space Mono'))
        fig.add_annotation(x=df.loc[mask_test,'Date'].mean(),y=0,yref='paper',
                            text="VALIDATION",showarrow=False,yanchor='bottom',
                            font=dict(color='rgba(0,212,180,0.4)',size=9,family='Space Mono'))
        fig.update_yaxes(title_text="P (mm/d)",row=1,col=1,autorange='reversed',
                          title_font=dict(size=10),gridcolor='rgba(30,58,95,0.6)')
        fig.update_yaxes(title_text="Q (m³/s)",row=2,col=1,title_font=dict(size=10),
                          gridcolor='rgba(30,58,95,0.6)')
        fig.update_layout(**plotly_base(430),legend=dict(orientation='h',y=-0.12,x=0.01))
        st.plotly_chart(fig,use_container_width=True)

    with col_right:
        sec_header("fa-sliders","Optimal Parameters")
        p_names=["Cmax","Bexp","Alpha","Ks","Kq"]
        p_units=["mm","—","—","—","—"]
        p_desc=["Soil capacity","PDM curvature","Fast fraction","Slow recession","Fast recession"]
        for nm,un,ds,val in zip(p_names,p_units,p_desc,params):
            st.markdown(f"""
            <div style="background:#0f1e33;border:1px solid #1e3a5f;border-radius:8px;
                        padding:10px 14px;margin:5px 0;display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <div style="font-size:0.7rem;color:#4a6a85;text-transform:uppercase;letter-spacing:0.07em;">{nm} ({un})</div>
                    <div style="font-size:0.74rem;color:#8aabbf;margin-top:1px;">{ds}</div>
                </div>
                <div style="font-family:'Space Mono',monospace;font-size:1.05rem;font-weight:700;color:#1a8cff;">{val:.5f}</div>
            </div>""",unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION VIII — PAGE: INTERACTIVE MAP
# ═════════════════════════════════════════════════════════════════════════════
def page_map(df):
    sec_header("fa-map-location-dot","Tamchachate Catchment — Interactive Map")
    st.markdown('<div class="info-block">Explore the basin boundary, hydro-climatic stations, '
                'and geographic context. Switch between tile layers using the layer control.</div>',
                unsafe_allow_html=True)
    m=folium.Map(location=[32.15,-5.50],zoom_start=10,tiles=None)
    for name,url,attr in [
        ('OpenStreetMap','OpenStreetMap','OpenStreetMap'),
        ('Satellite','https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}','ESRI'),
        ('Topographic','https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}','ESRI'),
    ]:
        folium.TileLayer(url if url!='OpenStreetMap' else 'OpenStreetMap',name=name,attr=attr).add_to(m)
    basin=[(32.22,-5.62),(32.25,-5.55),(32.23,-5.45),(32.20,-5.38),(32.15,-5.35),
           (32.08,-5.38),(32.05,-5.45),(32.07,-5.55),(32.10,-5.62),(32.15,-5.65),(32.22,-5.62)]
    folium.Polygon(basin,color='#1a8cff',weight=2.5,fill=True,fill_color='#4aa3e8',fill_opacity=0.18,
                    popup=folium.Popup(f"<b>Tamchachate Basin</b><br>Area: 128.86 km²<br>"
                                       f"Mean Q: {df['Qobs'].mean():.3f} m³/s",max_width=240),
                    tooltip="Tamchachate Catchment").add_to(m)
    folium.Marker([32.15,-5.50],popup="Basin Centroid",
                   icon=folium.Icon(color='darkblue',icon='home'),tooltip="Centroid").add_to(m)
    for sname,lat,lon,col,icon,info in [
        ('Hydrometric Station',32.13,-5.48,'blue','tint','Daily discharge (m³/s)'),
        ('Meteo Station Midelt',32.68,-4.73,'orange','cloud','Temperature · ETP'),
        ('Upstream Rain Gauge',32.20,-5.57,'green','info-sign','Daily precipitation (mm)'),
    ]:
        folium.Marker([lat,lon],popup=folium.Popup(f"<b>{sname}</b><br>{info}",max_width=220),
                       icon=folium.Icon(color=col,icon=icon,prefix='glyphicon'),
                       tooltip=sname).add_to(m)
    folium.LayerControl().add_to(m)
    try:
        from folium import plugins
        plugins.Fullscreen().add_to(m); plugins.MeasureControl(position='topright',primary_length_unit='kilometers').add_to(m)
    except Exception: pass

    col_map,col_info=st.columns([3,1])
    with col_map: st_folium(m,width=None,height=530)
    with col_info:
        sec_header("fa-circle-info","Station Legend")
        for lbl,color,desc,icon in [("Hydrometric","#1a8cff","Discharge gauge","fa-droplet"),
                                      ("Meteo","#e67e22","Climate station","fa-cloud-sun"),
                                      ("Rain Gauge","#22c55e","Precipitation","fa-cloud-rain")]:
            st.markdown(f"""<div style="display:flex;align-items:center;gap:10px;padding:9px 12px;
                        background:#0f1e33;border:1px solid #1e3a5f;border-radius:8px;margin:4px 0;">
                <i class="fa-solid {icon}" style="color:{color};font-size:1rem;width:18px;"></i>
                <div><div style="font-weight:600;font-size:0.84rem;color:#e8f1fb;">{lbl}</div>
                     <div style="font-size:0.7rem;color:#4a6a85;">{desc}</div></div></div>""",
                        unsafe_allow_html=True)
        sec_header("fa-chart-pie","Basin Statistics")
        for k,v in [("Area","128.86 km²"),("Total days",f"{len(df):,}"),("Wet days",f"{(df['P']>0).sum():,}"),
                     ("Mean P",f"{df['P'].mean()*365:.0f} mm/yr"),("Mean PET",f"{df['PET'].mean()*365:.0f} mm/yr"),
                     ("Mean Q",f"{df['Qobs'].mean():.3f} m³/s"),("Peak Q",f"{df['Qobs'].max():.2f} m³/s")]:
            st.markdown(f'<div style="display:flex;justify-content:space-between;padding:4px 0;'
                        f'border-bottom:1px solid #1e3a5f;font-size:0.78rem;">'
                        f'<span style="color:#8aabbf;">{k}</span>'
                        f'<span style="color:#e8f1fb;font-family:Space Mono,monospace;">{v}</span>'
                        f'</div>',unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION IX — PAGE: MODEL RESULTS
# ═════════════════════════════════════════════════════════════════════════════
def page_results(df,met_train,met_test,mask_train,mask_test):
    sec_header("fa-chart-line","Model Results — Calibration & Validation")
    period=st.radio("Period:",["Calibration  (2002–2013)","Validation  (2014–2019)"],horizontal=True)
    is_cal=period.startswith("Cal")
    mask_p=mask_train if is_cal else mask_test
    met_p=met_train if is_cal else met_test
    Qobs_p=df.loc[mask_p,'Qobs'].values; Qsim_p=df.loc[mask_p,'Qsim'].values
    dates_p=df.loc[mask_p,'Date'].values; P_p=df.loc[mask_p,'P'].values
    clr="#ff6b6b" if is_cal else "#ff9c5b"

    cols7=st.columns(7)
    for col,(ic,lbl,val,tgt,tl) in zip(cols7,[
        ("fa-chart-bar","NSE",met_p['NSE'],"> 0.50",False),
        ("fa-droplet","NSE log",met_p['NSE_log'],"> 0.40",False),
        ("fa-bullseye","KGE",met_p['KGE'],"> 0.50",False),
        ("fa-square-root-variable","R²",met_p['R2'],"> 0.60",False),
        ("fa-ruler","RMSE",met_p['RMSE'],"→ 0",True),
        ("fa-minus","MAE",met_p['MAE'],"→ 0",True),
        ("fa-percent","PBIAS",met_p['PBIAS'],"< 25%",True),
    ]): metric_card(col,ic,f"{val:.3f}",lbl,target=tgt,teal=tl)

    st.markdown("<br>",unsafe_allow_html=True)
    tab1,tab2,tab3,tab4,tab5=st.tabs(["  Hydrograph  ","  Scatter  ","  Flow Duration  ","  Seasonal  ","  Radar  "])

    with tab1:
        fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.22,0.78],vertical_spacing=0.04)
        fig.add_trace(go.Bar(x=dates_p,y=P_p,name='P (mm/d)',marker_color='#3a7cbf',opacity=0.7),row=1,col=1)
        fig.add_trace(go.Scatter(x=dates_p,y=Qobs_p,name='Observed',line=dict(color='#4aa3e8',width=1.2)),row=2,col=1)
        fig.add_trace(go.Scatter(x=dates_p,y=Qsim_p,name='Simulated',line=dict(color=clr,width=1.2,dash='dash')),row=2,col=1)
        fig.update_yaxes(title_text="P (mm/d)",row=1,col=1,autorange='reversed',gridcolor='rgba(30,58,95,0.6)')
        fig.update_yaxes(title_text="Q (m³/s)",row=2,col=1,gridcolor='rgba(30,58,95,0.6)')
        fig.update_layout(**plotly_base(470),
                           title=dict(text=f"NSE={met_p['NSE']:.3f}  ·  KGE={met_p['KGE']:.3f}  ·  R²={met_p['R2']:.3f}",
                                      font=dict(size=11,color='#8aabbf',family='Space Mono')),
                           legend=dict(orientation='h',y=-0.13))
        st.plotly_chart(fig,use_container_width=True)

    with tab2:
        cl,cr=st.columns(2); mq=max(Qobs_p.max(),Qsim_p.max())*1.05
        with cl:
            fig2=go.Figure()
            fig2.add_trace(go.Scatter(x=Qobs_p,y=Qsim_p,mode='markers',marker=dict(color=clr,opacity=0.22,size=4)))
            fig2.add_trace(go.Scatter(x=[0,mq],y=[0,mq],name='1:1',line=dict(color='#aaa',dash='dash',width=1.5)))
            c=np.polyfit(Qobs_p,Qsim_p,1); xr=np.linspace(0,mq,100)
            fig2.add_trace(go.Scatter(x=xr,y=np.polyval(c,xr),name=f'Regr. slope={c[0]:.3f}',line=dict(color='#00d4b4',width=1.5)))
            fig2.update_layout(**plotly_base(360),
                                title=dict(text=f"Scatter — R²={met_p['R2']:.4f}",font=dict(size=11,color='#8aabbf',family='Space Mono')),
                                xaxis_title="Qobs (m³/s)",yaxis_title="Qsim (m³/s)")
            st.plotly_chart(fig2,use_container_width=True)
        with cr:
            res=Qobs_p-Qsim_p; fig3=go.Figure()
            fig3.add_trace(go.Histogram(x=res,nbinsx=50,marker_color='#1a8cff',opacity=0.7))
            fig3.add_vline(x=0,line_dash='dash',line_color='#ff6b6b',annotation_text='zero')
            fig3.add_vline(x=res.mean(),line_dash='dot',line_color='#f59e0b',annotation_text=f'mean={res.mean():.4f}')
            fig3.update_layout(**plotly_base(360),
                                title=dict(text="Residual Distribution",font=dict(size=11,color='#8aabbf',family='Space Mono')),
                                xaxis_title="Residual (m³/s)",yaxis_title="Frequency")
            st.plotly_chart(fig3,use_container_width=True)

    with tab3:
        Qo_s=np.sort(Qobs_p)[::-1]; Qs_s=np.sort(Qsim_p)[::-1]; pct=np.arange(1,len(Qo_s)+1)/len(Qo_s)*100
        fig4=go.Figure()
        fig4.add_trace(go.Scatter(x=pct,y=Qo_s,name='Observed',line=dict(color='#4aa3e8',width=2)))
        fig4.add_trace(go.Scatter(x=pct,y=Qs_s,name='Simulated',line=dict(color=clr,width=2,dash='dash')))
        for qp in [50,90]:
            fig4.add_vline(x=qp,line_dash='dot',line_color='rgba(90,90,90,0.6)',
                            annotation_text=f'Q{qp}',annotation_position='top right',annotation_font_size=9)
        fig4.update_layout(**plotly_base(420),
                            title=dict(text="Flow Duration Curve (log scale)",font=dict(size=11,color='#8aabbf',family='Space Mono')),
                            xaxis_title="Exceedance probability (%)",yaxis_title="Q (m³/s)",yaxis_type='log')
        st.plotly_chart(fig4,use_container_width=True)

    with tab4:
        dp=df[mask_p].copy(); dp['Month']=pd.to_datetime(dp['Date']).dt.month
        mnths=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        fig5=make_subplots(specs=[[{"secondary_y":True}]])
        fig5.add_trace(go.Bar(x=mnths,y=dp.groupby('Month')['P'].mean().values,name='Mean P',
                               marker_color='#3a7cbf',opacity=0.5),secondary_y=True)
        fig5.add_trace(go.Scatter(x=mnths,y=dp.groupby('Month')['Qobs'].mean().values,name='Obs Q',
                                   line=dict(color='#4aa3e8',width=2.5),mode='lines+markers'),secondary_y=False)
        fig5.add_trace(go.Scatter(x=mnths,y=dp.groupby('Month')['Qsim'].mean().values,name='Sim Q',
                                   line=dict(color=clr,width=2.5,dash='dash'),mode='lines+markers'),secondary_y=False)
        fig5.update_yaxes(title_text="Mean Q (m³/s)",secondary_y=False,gridcolor='rgba(30,58,95,0.6)')
        fig5.update_yaxes(title_text="Mean P (mm/d)",secondary_y=True,autorange='reversed')
        fig5.update_layout(**plotly_base(420),
                            title=dict(text="Mean Seasonal Cycle",font=dict(size=11,color='#8aabbf',family='Space Mono')))
        st.plotly_chart(fig5,use_container_width=True)

    with tab5:
        def rv(m): return [max(m['NSE'],0),max(m['NSE_log'],0),max(m['KGE'],0),max(m['R2'],0),max(1-abs(m['PBIAS'])/100,0)]
        cats=['NSE','NSE log','KGE','R²','1-|PBIAS|']
        vt=rv(met_train); vv=rv(met_test)
        fig6=go.Figure()
        fig6.add_trace(go.Scatterpolar(r=vt+[vt[0]],theta=cats+[cats[0]],fill='toself',name='Calibration',
                                        line_color='#1a8cff',fillcolor='rgba(26,140,255,0.18)'))
        fig6.add_trace(go.Scatterpolar(r=vv+[vv[0]],theta=cats+[cats[0]],fill='toself',name='Validation',
                                        line_color='#00d4b4',fillcolor='rgba(0,212,180,0.12)'))
        fig6.add_trace(go.Scatterpolar(r=[1]*len(cats)+[1],theta=cats+[cats[0]],
                                        name='Ideal',line=dict(color='#2d4a62',dash='dot',width=1)))
        fig6.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,1],gridcolor='#1e3a5f',linecolor='#1e3a5f'),
                                       angularaxis=dict(linecolor='#1e3a5f',gridcolor='#1e3a5f')),
                            **{k:v for k,v in plotly_base(440).items() if k not in('xaxis','yaxis')})
        col_r,col_t=st.columns([2,1])
        with col_r: st.plotly_chart(fig6,use_container_width=True)
        with col_t:
            sec_header("fa-table","Full Metrics")
            rows=[]
            for k,d,tg in [("NSE","Nash-Sutcliffe","> 0.50"),("NSE_log","NSE log-Q","> 0.40"),
                            ("RMSE","RMSE (m³/s)","→ 0"),("MAE","MAE (m³/s)","→ 0"),
                            ("KGE","Kling-Gupta","> 0.50"),("R2","R² Pearson","> 0.60"),("PBIAS","Bias (%)","< 25%")]:
                rows.append({"Metric":k,"Calib.":f"{met_train[k]:.4f}","Valid.":f"{met_test[k]:.4f}","Target":tg})
            st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION X — PAGE: SIMULATION TOOL
# ═════════════════════════════════════════════════════════════════════════════
def page_simulation(df,CONV,params,met_train,met_test,mask_train,mask_test):
    sec_header("fa-sliders","Interactive Simulation Tool")
    st.markdown('<div class="info-block">Modify HYMOD parameters and rainfall forcing in real-time. '
                'Compare against the calibrated reference to understand parameter sensitivity.</div>',
                unsafe_allow_html=True)
    col_ctrl,col_plot=st.columns([1,3])
    with col_ctrl:
        st.markdown("**Forcing Multiplier**")
        rain_mult=st.slider("Rainfall ×",0.5,2.0,1.0,0.05,key="rain_m")
        st.markdown('<div style="font-size:0.7rem;color:#4a6a85;">Scales all P inputs</div>',unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("**HYMOD Parameters**")
        Cmax =st.slider("Cmax (mm)",10.0,500.0,float(round(params[0],1)),5.0)
        Bexp =st.slider("Bexp (–)",0.1,5.0,float(round(params[1],3)),0.05)
        Alpha=st.slider("Alpha (–)",0.01,0.99,float(round(params[2],3)),0.01)
        Ks   =st.slider("Ks (–)",0.001,0.10,float(round(params[3],4)),0.001)
        Kq   =st.slider("Kq (–)",0.10,0.99,float(round(params[4],3)),0.01)
        st.markdown("---")
        sim_period=st.radio("Period",["Calibration","Validation","Full record"])
        if st.button("Reset to calibrated values",use_container_width=True): st.rerun()

    mask_p=mask_train if sim_period=="Calibration" else(mask_test if sim_period=="Validation" else df['Date'].notna())
    P_sc=df['P'].values*rain_mult
    u_par=(Cmax,Bexp,Alpha,Ks,Kq)
    Qsim_u_mm=_run_hymod_np(u_par,P_sc,df['PET'].values)
    Qsim_u_m3s=Qsim_u_mm*CONV
    Qobs_s=df.loc[mask_p,'Qobs'].values; Qsim_s=Qsim_u_m3s[mask_p]
    Qref_s=df.loc[mask_p,'Qsim'].values; dates_s=df.loc[mask_p,'Date'].values
    met_u=compute_metrics(Qobs_s,Qsim_s); met_r=compute_metrics(Qobs_s,Qref_s)

    with col_plot:
        c1,c2,c3,c4=st.columns(4)
        for col,key,ic in[(c1,"NSE","fa-chart-bar"),(c2,"KGE","fa-bullseye"),
                           (c3,"R2","fa-square-root-variable"),(c4,"PBIAS","fa-percent")]:
            uv=met_u[key]; rv2=met_r[key]; d=uv-rv2
            dc="#22c55e" if d>=0 else "#ef4444"
            arrow="▲" if d>0.0005 else("▼" if d<-0.0005 else "—")
            col.markdown(f"""<div class="metric-card"><div class="mi"><i class="fa-solid {ic}"></i></div>
                <div class="mv">{uv:.3f}</div><div class="ml">{key}</div>
                <div class="ms" style="color:{dc};font-family:Space Mono,monospace;font-size:0.78rem;">
                {arrow} {abs(d):.3f}</div></div>""",unsafe_allow_html=True)

        fig_s=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.22,0.78],vertical_spacing=0.04)
        fig_s.add_trace(go.Bar(x=dates_s,y=df.loc[mask_p,'P'].values*rain_mult,
                                name=f'P ×{rain_mult:.2f}',marker_color='#3a7cbf',opacity=0.7),row=1,col=1)
        fig_s.add_trace(go.Scatter(x=dates_s,y=Qobs_s,name='Observed',line=dict(color='#4aa3e8',width=1.2)),row=2,col=1)
        fig_s.add_trace(go.Scatter(x=dates_s,y=Qref_s,name='Calibrated ref.',line=dict(color='#6b7280',width=1.0,dash='dot')),row=2,col=1)
        fig_s.add_trace(go.Scatter(x=dates_s,y=Qsim_s,name='User simulation',line=dict(color='#ff9c5b',width=1.5,dash='dash')),row=2,col=1)
        fig_s.update_yaxes(title_text=f"P ×{rain_mult:.2f}",row=1,col=1,autorange='reversed',gridcolor='rgba(30,58,95,0.6)')
        fig_s.update_yaxes(title_text="Q (m³/s)",row=2,col=1,gridcolor='rgba(30,58,95,0.6)')
        fig_s.update_layout(**plotly_base(450),
                             title=dict(text=f"Rainfall ×{rain_mult:.2f}  ·  Period: {sim_period}",
                                        font=dict(size=11,color='#8aabbf',family='Space Mono')),
                             legend=dict(orientation='h',y=-0.13))
        st.plotly_chart(fig_s,use_container_width=True)
        pct_q=(Qsim_s.mean()-Qref_s.mean())/max(Qref_s.mean(),1e-9)*100
        pct_p=(rain_mult-1)*100
        st.markdown(f'<div class="info-block"><i class="fa-solid fa-circle-info" '
                    f'style="color:#1a8cff;margin-right:8px;"></i>'
                    f'Rainfall {pct_p:+.0f}% → Mean simulated Q {pct_q:+.1f}% '
                    f'({Qref_s.mean():.3f} → {Qsim_s.mean():.3f} m³/s). '
                    f'Non-linear response reflects soil moisture saturation dynamics.</div>',
                    unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION XI — PAGE: MODEL THEORY
# ═════════════════════════════════════════════════════════════════════════════
def page_theory():
    sec_header("fa-book-open","HYMOD — Model Structure & Theory")
    st.markdown('<div class="info-block">HYMOD is a conceptual lumped rainfall-runoff model based on the '
                'Probability Distributed Model (PDM). It simulates soil moisture dynamics, fast overland '
                'flow, and slow baseflow using 5 parameters.</div>',unsafe_allow_html=True)

    sec_header("fa-sitemap","Conceptual Architecture")
    fig_d=go.Figure()
    fig_d.add_shape(type='rect',x0=0,y0=0,x1=10,y1=9,fillcolor='rgba(11,22,38,0.97)',line_color='#1e3a5f')
    for x0,y0,x1,y1,fill,border,lbl,sub in [
        (3.5,7.2,6.5,8.6,'rgba(26,140,255,0.2)','#1a8cff','ATMOSPHERE','P  ·  PET inputs'),
        (3.5,5.0,6.5,6.5,'rgba(34,197,94,0.2)','#22c55e','SOIL STORE','Cmax · Bexp'),
        (1.0,2.5,3.5,4.0,'rgba(139,92,246,0.2)','#8b5cf6','SLOW RES.','Ks coefficient'),
        (4.0,2.5,6.0,4.0,'rgba(239,68,68,0.2)','#ef4444','FAST RES. 1','Kq coefficient'),
        (6.5,2.5,8.5,4.0,'rgba(239,68,68,0.2)','#ef4444','FAST RES. 2','Kq coefficient'),
        (3.5,0.5,6.5,1.8,'rgba(0,212,180,0.2)','#00d4b4','TOTAL Q','Qs + Qq  [m³/s]'),
    ]:
        fig_d.add_shape(type='rect',x0=x0,y0=y0,x1=x1,y1=y1,fillcolor=fill,line_color=border,line_width=1.5)
        cx,cy=(x0+x1)/2,(y0+y1)/2
        fig_d.add_annotation(x=cx,y=cy+0.2,text=f"<b>{lbl}</b>",showarrow=False,
                              font=dict(color='#e8f1fb',size=10,family='Syne'))
        fig_d.add_annotation(x=cx,y=cy-0.25,text=sub,showarrow=False,
                              font=dict(color='#8aabbf',size=8,family='Space Mono'))
    for ax,ay,bx,by in [(5.0,7.2,5.0,6.5),(4.5,5.0,2.25,4.0),(5.5,5.0,5.0,4.0),
                          (2.25,2.5,4.5,1.8),(7.5,2.5,6.3,1.8),(6.0,3.25,6.5,3.25)]:
        fig_d.add_annotation(ax=ax,ay=ay,x=bx,y=by,showarrow=True,arrowhead=2,
                              arrowsize=1.2,arrowwidth=1.5,arrowcolor='#1a8cff',text='')
    fig_d.add_annotation(x=5.0,y=4.75,text="α split",showarrow=False,
                          font=dict(color='#f59e0b',size=9,family='Space Mono'))
    fig_d.update_xaxes(visible=False,range=[-0.2,10.2])
    fig_d.update_yaxes(visible=False,range=[0,9.5])
    fig_d.update_layout(height=390,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(11,22,38,0.97)',
                         margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig_d,use_container_width=True)

    st.markdown("<br>",unsafe_allow_html=True)
    sec_header("fa-layer-group","Component Details")
    c1,c2,c3=st.columns(3)
    for col,color,icon,title,body,params_txt in [
        (c1,'#22c55e','fa-seedling','Soil Moisture Store',
         'The PDM component represents spatial variability of soil storage. Precipitation fills the store first; only when capacity is reached does excess runoff occur.',
         'Cmax — Maximum storage (mm)<br>Bexp — Curvature of distribution'),
        (c2,'#ef4444','fa-bolt','Fast Flow Component',
         'Fraction α of excess rainfall routes through 3 cascaded linear reservoirs, producing a gamma-shaped unit hydrograph for surface/subsurface storm runoff.',
         'Alpha — Fraction to fast store<br>Kq — Fast recession constant'),
        (c3,'#8b5cf6','fa-hourglass-half','Slow Flow Component',
         'Fraction (1−α) feeds a single slow reservoir representing groundwater baseflow, sustaining low flows during dry seasons with long memory.',
         'Alpha — Controls (1−α) to slow<br>Ks — Slow recession constant'),
    ]:
        col.markdown(f"""<div class="hmp-card" style="border-top:3px solid {color};">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
                <i class="fa-solid {icon}" style="color:{color};font-size:1.1rem;"></i>
                <span style="font-family:'Syne',sans-serif;font-weight:700;font-size:0.95rem;color:#e8f1fb;">{title}</span>
            </div>
            <p style="font-size:0.83rem;color:#8aabbf;line-height:1.7;margin:0 0 12px;">{body}</p>
            <div style="background:#060d18;border-radius:6px;padding:8px 12px;
                        font-family:'Space Mono',monospace;font-size:0.7rem;color:{color};line-height:1.8;">{params_txt}</div>
        </div>""",unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    sec_header("fa-superscript","Model Equations")
    st.markdown("""
    | Component | Equation | Description |
    |---|---|---|
    | **Soil moisture** | $AET = PET \\cdot \\min\\left(\\dfrac{S}{C_{max}}, 1\\right)$ | Actual evapotranspiration |
    | **Excess runoff** | $exc = P \\cdot f(S, C_{max}, B_{exp})$ | PDM probability distribution |
    | **Fast routing** | $Q_q = K_q \\cdot Q_3$; cascade $Q_1 \\to Q_2 \\to Q_3$ | 3 cascaded linear reservoirs |
    | **Slow routing** | $Q_s = K_s \\cdot S_{slow}$ | Single baseflow reservoir |
    | **Total output** | $Q_{sim} = Q_s + Q_q$ | [mm/d] converted to [m³/s] |
    """)

    sec_header("fa-book","References")
    for ref in [
        "Moore, R.J. (1985). The probability-distributed principle. *Hydrological Sciences Journal*, 30(2), 273–297.",
        "Wagener et al. (2001). A framework for hydrological models. *HESS*, 5(1), 13–26.",
        "Chaponnière et al. (2008). Hydrological processes in the Moroccan High Atlas. *Hydrological Processes*, 22(12).",
        "Boudhar et al. (2009). Snowmelt runoff model evaluation, Moroccan High Atlas. *HSJ*, 54(6).",
    ]:
        st.markdown(f"- {ref}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION XII — PAGE: MY OWN DATA
# ═════════════════════════════════════════════════════════════════════════════
def page_my_data():
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0b1626 0%,#0f2540 60%,#132e55 100%);
                border:1px solid #1e3a5f;border-radius:14px;padding:28px 34px 22px;margin-bottom:24px;">
        <div style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#1a8cff;
                    letter-spacing:0.2em;text-transform:uppercase;margin-bottom:8px;">
            <i class="fa-solid fa-upload"></i> &nbsp; Custom Basin Analysis
        </div>
        <h2 style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;
                   color:#e8f1fb;margin:0 0 6px;">Run HYMOD on Your Own Data</h2>
        <p style="color:#8aabbf;margin:0;font-size:0.9rem;max-width:620px;">
            Upload a CSV or Excel file, map your columns, define calibration and validation periods,
            then the platform calibrates and validates HYMOD automatically.
        </p>
    </div>""",unsafe_allow_html=True)

    # Steps progress
    steps=[("fa-upload","Upload file","CSV / Excel"),("fa-table-columns","Map columns","4 variables"),
           ("fa-calendar-days","Set periods","Cal + Val dates"),("fa-gear","Run HYMOD","Auto calibration"),
           ("fa-chart-line","View results","Full diagnostics")]
    s_html='<div style="display:flex;gap:0;margin:20px 0 28px;position:relative;">'
    s_html+='<div style="position:absolute;top:14px;left:5%;width:90%;height:1px;background:#1e3a5f;z-index:0;"></div>'
    for i,(ic,t,d) in enumerate(steps):
        s_html+=f"""<div style="flex:1;text-align:center;position:relative;z-index:1;">
            <div style="width:28px;height:28px;border-radius:50%;background:{'#1a8cff' if i<4 else '#132e55'};
                        border:2px solid {'#1a8cff' if i<4 else '#1e3a5f'};display:inline-flex;
                        align-items:center;justify-content:center;margin-bottom:6px;">
                <i class="fa-solid {ic}" style="font-size:0.65rem;color:{'white' if i<4 else '#4a6a85'};"></i>
            </div>
            <div style="font-size:0.7rem;font-weight:600;color:#e8f1fb;">{t}</div>
            <div style="font-size:0.62rem;color:#4a6a85;margin-top:1px;">{d}</div></div>"""
    st.markdown(s_html+'</div>',unsafe_allow_html=True)

    # Step 1
    sec_header("fa-upload","Step 1 — Upload Data File")
    col_up,col_fmt=st.columns([2,3])
    with col_up:
        uploaded=st.file_uploader("Upload CSV or Excel",type=["csv","xlsx","xls"],
                                   help="Daily time-series with Date, P, PET, Q columns")
    with col_fmt:
        st.markdown("""<div class="hmp-card" style="font-size:0.82rem;">
            <div style="font-family:'Syne',sans-serif;font-weight:700;color:#1a8cff;margin-bottom:10px;">
                <i class="fa-solid fa-circle-info"></i> &nbsp; Expected Format
            </div>
            <table style="width:100%;border-collapse:collapse;font-size:0.76rem;">
                <tr style="color:#1a8cff;font-family:'Space Mono',monospace;">
                    <th style="padding:4px 8px;text-align:left;">Column</th><th style="padding:4px 8px;text-align:left;">Unit</th><th style="padding:4px 8px;text-align:left;">Example</th>
                </tr>
                <tr style="border-top:1px solid #1e3a5f;"><td style="padding:4px 8px;color:#e8f1fb;">Date</td><td style="color:#8aabbf;">—</td><td style="color:#4a6a85;font-family:Space Mono,monospace;">2005-03-21</td></tr>
                <tr style="border-top:1px solid #1e3a5f;"><td style="padding:4px 8px;color:#e8f1fb;">Precipitation</td><td style="color:#8aabbf;">mm/day</td><td style="color:#4a6a85;font-family:Space Mono,monospace;">4.7</td></tr>
                <tr style="border-top:1px solid #1e3a5f;"><td style="padding:4px 8px;color:#e8f1fb;">PET</td><td style="color:#8aabbf;">mm/day</td><td style="color:#4a6a85;font-family:Space Mono,monospace;">3.2</td></tr>
                <tr style="border-top:1px solid #1e3a5f;"><td style="padding:4px 8px;color:#e8f1fb;">Observed Q</td><td style="color:#8aabbf;">m³/s</td><td style="color:#4a6a85;font-family:Space Mono,monospace;">0.854</td></tr>
            </table>
            <div style="margin-top:10px;color:#4a6a85;font-size:0.7rem;">
                Separator: , or ; &nbsp;·&nbsp; Decimal: . or , &nbsp;·&nbsp; Min: 3 years
            </div></div>""",unsafe_allow_html=True)

    with st.expander("Download a sample CSV template"):
        sample=pd.DataFrame({"Date":pd.date_range("2005-01-01",periods=10,freq="D").strftime("%Y-%m-%d"),
                              "P_mm":[0.0,5.2,12.1,0.0,0.0,8.4,2.1,0.0,0.0,3.7],
                              "PET_mm":[2.1,1.9,2.0,2.3,2.5,1.8,2.0,2.4,2.6,2.2],
                              "Qobs_m3s":[0.32,0.35,0.41,0.38,0.34,0.37,0.40,0.36,0.33,0.34]})
        st.dataframe(sample,use_container_width=True,hide_index=True)
        st.download_button("Download template.csv",sample.to_csv(index=False).encode(),"hymod_template.csv","text/csv")

    if uploaded is None:
        st.markdown('<div class="info-block"><i class="fa-solid fa-arrow-up-from-bracket" '
                    'style="color:#1a8cff;margin-right:8px;"></i>Upload a file above to continue.</div>',
                    unsafe_allow_html=True)
        return

    # Step 2
    sec_header("fa-table-columns","Step 2 — Column Mapping")
    try:
        if uploaded.name.endswith((".xlsx",".xls")):
            raw_df=pd.read_excel(uploaded)
        else:
            content=uploaded.read(); uploaded.seek(0)
            try:
                raw_df=pd.read_csv(uploaded,sep=",",decimal=".",encoding="utf-8")
                if raw_df.shape[1]<3:
                    uploaded.seek(0); raw_df=pd.read_csv(uploaded,sep=";",decimal=",",encoding="utf-8")
            except Exception:
                import io; raw_df=pd.read_csv(io.BytesIO(content),sep=";",decimal=",",encoding="latin-1")
        raw_df.columns=[c.strip() for c in raw_df.columns]
    except Exception as e:
        st.error(f"Could not read file: {e}"); return

    with st.expander("Raw file preview (first 8 rows)"):
        st.dataframe(raw_df.head(8),use_container_width=True)

    def best(kws):
        for kw in kws:
            for c in raw_df.columns:
                if kw.lower() in c.lower(): return c
        return raw_df.columns[0]

    ca,cb,cc,cd=st.columns(4)
    col_date=ca.selectbox("Date",raw_df.columns,index=list(raw_df.columns).index(best(["date","Date","DATE","time"])))
    col_P   =cb.selectbox("Precipitation (mm/d)",raw_df.columns,index=list(raw_df.columns).index(best(["P","prec","rain","precip"])))
    col_PET =cc.selectbox("PET (mm/d)",raw_df.columns,index=list(raw_df.columns).index(best(["PET","pet","ETP","ET"])))
    col_Q   =cd.selectbox("Observed Q (m³/s)",raw_df.columns,index=list(raw_df.columns).index(best(["Q","Qobs","discharge","flow","debit"])))

    # Step 3
    sec_header("fa-calendar-days","Step 3 — Basin Info & Study Periods")
    c1,c2,c3=st.columns(3)
    with c1:
        basin_name=st.text_input("Basin name","My Basin")
        area_km2=st.number_input("Area (km²)",0.1,100000.0,100.0,0.1,format="%.2f")
    with c2:
        st.markdown("**Calibration period**")
        cal_start=st.text_input("Start (YYYY-MM-DD)","",key="cal_s")
        cal_end  =st.text_input("End   (YYYY-MM-DD)","",key="cal_e")
        st.caption("Leave blank → auto (first 70%)")
    with c3:
        st.markdown("**Validation period**")
        val_start=st.text_input("Start (YYYY-MM-DD)","",key="val_s")
        val_end  =st.text_input("End   (YYYY-MM-DD)","",key="val_e")
        st.caption("Leave blank → remainder of record")
    warmup_yrs=st.slider("Warm-up years (excluded from metrics)",0,3,1)

    # Step 4 — Run
    sec_header("fa-gear","Step 4 — Run HYMOD")
    run_btn=st.button("Run HYMOD Calibration  →",type="primary",use_container_width=True)
    if not run_btn and not st.session_state.user_run_done:
        st.markdown('<div class="info-block"><i class="fa-solid fa-circle-play" '
                    'style="color:#1a8cff;margin-right:8px;"></i>Configure settings above then click Run.</div>',
                    unsafe_allow_html=True)
        return
    if run_btn: st.session_state.user_run_done=False

    # Processing
    progress=st.progress(0,"Parsing data…")
    try:
        udf=raw_df[[col_date,col_P,col_PET,col_Q]].copy()
        udf.columns=["Date","P","PET","Qobs"]
        udf["Date"]=pd.to_datetime(udf["Date"],dayfirst=True,errors="coerce")
        for c in["P","PET","Qobs"]: udf[c]=pd.to_numeric(udf[c],errors="coerce")
        udf=udf.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        udf[["P","PET","Qobs"]]=udf[["P","PET","Qobs"]].interpolate().fillna(0)
        udf["P"]=udf["P"].clip(lower=0); udf["PET"]=udf["PET"].clip(lower=0); udf["Qobs"]=udf["Qobs"].clip(lower=0)
    except Exception as e:
        st.error(f"Data processing error: {e}"); return
    if len(udf)<365: st.error("Need at least 1 year of daily data."); return

    progress.progress(15,"Computing conversion factor…")
    CONV_u=area_km2*1e6/(1000.0*86400.0); udf["Qobs_mm"]=udf["Qobs"]/CONV_u
    t_min=udf["Date"].min(); t_max=udf["Date"].max()
    try:
        cal_s=pd.to_datetime(cal_start) if cal_start else t_min
        cal_e=pd.to_datetime(cal_end)   if cal_end   else t_min+pd.Timedelta(days=int(0.70*(t_max-t_min).days))
        val_s=pd.to_datetime(val_start) if val_start else cal_e+pd.Timedelta(days=1)
        val_e=pd.to_datetime(val_end)   if val_end   else t_max
    except Exception: st.error("Invalid date format. Use YYYY-MM-DD."); return

    warmup_end=cal_s+pd.DateOffset(years=warmup_yrs)
    mask_wu=(udf["Date"]<=warmup_end); mask_cal=(udf["Date"]>warmup_end)&(udf["Date"]<=cal_e); mask_val=(udf["Date"]>=val_s)&(udf["Date"]<=val_e)
    if mask_cal.sum()<180: st.error("Calibration period too short (min 6 months)."); return

    progress.progress(35,"Preparing calibration arrays…")
    P_arr=np.asarray(udf.loc[mask_wu|mask_cal,"P"].values,dtype=float)
    PET_arr=np.asarray(udf.loc[mask_wu|mask_cal,"PET"].values,dtype=float)
    Qo_arr=np.asarray(udf.loc[mask_cal,"Qobs_mm"].values,dtype=float)
    n_wu=int(mask_wu.sum())

    progress.progress(42,"Running Differential Evolution calibration… (~30s)")
    bounds_u=[(1.0,500.0),(0.1,5.0),(0.01,0.99),(0.0001,0.1),(0.1,0.99)]
    def obj_u(p):
        Qs=_run_hymod_np(p,P_arr,PET_arr)
        nse=1.0-np.sum((Qo_arr-Qs[n_wu:])**2)/np.sum((Qo_arr-Qo_arr.mean())**2)
        return -nse if not(np.isnan(nse) or np.isinf(nse)) else 9999.0
    res_u=differential_evolution(obj_u,bounds_u,seed=42,maxiter=800,popsize=15,tol=1e-7,
                                   mutation=(0.5,1.5),recombination=0.7,polish=True,workers=1,disp=False)
    params_u=res_u.x

    progress.progress(87,"Full-period simulation…")
    Qsim_mm=_run_hymod_np(params_u,udf["P"].values,udf["PET"].values)
    udf["Qsim"]=Qsim_mm*CONV_u; udf["Qsim_mm"]=Qsim_mm
    met_cal_u=compute_metrics(udf.loc[mask_cal,"Qobs"].values,udf.loc[mask_cal,"Qsim"].values)
    met_val_u=compute_metrics(udf.loc[mask_val,"Qobs"].values,udf.loc[mask_val,"Qsim"].values) if mask_val.sum()>0 else None
    progress.progress(100,"Done!")

    st.session_state.user_results=dict(udf=udf,CONV_u=CONV_u,params_u=params_u,
                                        met_cal_u=met_cal_u,met_val_u=met_val_u,
                                        mask_cal=mask_cal,mask_val=mask_val,basin_name=basin_name)
    st.session_state.user_run_done=True
    _render_user_results(st.session_state.user_results)


def _render_user_results(res):
    udf=res['udf']; params_u=res['params_u']; CONV_u=res['CONV_u']
    met_cal_u=res['met_cal_u']; met_val_u=res['met_val_u']
    mask_cal=res['mask_cal']; mask_val=res['mask_val']; basin_name=res['basin_name']

    st.success(f"HYMOD calibrated on **{basin_name}** — {len(udf):,} days")
    sec_header("fa-chart-pie","Data Overview")
    c1,c2,c3,c4,c5=st.columns(5)
    rc=udf['Qobs_mm'].mean()/max(udf['P'].mean(),1e-6)*100
    for col,ic,lbl,val,tl in [(c1,"fa-calendar","Total days",f"{len(udf):,}",False),
                                (c2,"fa-cloud-rain","Mean P",f"{udf['P'].mean()*365:.0f} mm/yr",False),
                                (c3,"fa-sun","Mean PET",f"{udf['PET'].mean()*365:.0f} mm/yr",False),
                                (c4,"fa-water","Mean Qobs",f"{udf['Qobs'].mean():.3f} m³/s",True),
                                (c5,"fa-percent","Runoff coeff.",f"{rc:.1f}%",True)]:
        metric_card(col,ic,val,lbl,teal=tl)

    sec_header("fa-sliders","Calibrated Parameters")
    pcols=st.columns(5)
    for col,nm,un,ds,rng,val in zip(pcols,["Cmax","Bexp","Alpha","Ks","Kq"],["mm","–","–","–","–"],
                                     ["Max soil storage","PDM curvature","Fast/slow split","Slow recession","Fast recession"],
                                     ["1–500","0.1–5.0","0.01–0.99","0.0001–0.10","0.10–0.99"],params_u):
        col.markdown(f"""<div style="background:#0f1e33;border:1px solid #1e3a5f;border-top:2px solid #1a8cff;
                    border-radius:8px;padding:14px 10px;text-align:center;margin:2px;">
            <div style="font-family:'Space Mono',monospace;font-size:1.1rem;font-weight:700;color:#1a8cff;">{val:.5f}</div>
            <div style="font-size:0.7rem;color:#8aabbf;text-transform:uppercase;letter-spacing:0.07em;margin-top:5px;">{nm} ({un})</div>
            <div style="font-size:0.68rem;color:#4a6a85;margin-top:2px;">{ds}</div>
            <div style="font-size:0.65rem;color:#2d4a62;margin-top:2px;">[{rng}]</div>
        </div>""",unsafe_allow_html=True)

    sec_header("fa-circle-check","Performance Metrics")
    cc,cv=st.columns(2)
    with cc:
        st.markdown(f"**Calibration** &nbsp; {quality_badge(met_cal_u['NSE'])}",unsafe_allow_html=True)
        cc1,cc2,cc3,cc4=st.columns(4)
        for col,k,ic in[(cc1,"NSE","fa-chart-bar"),(cc2,"KGE","fa-bullseye"),(cc3,"R2","fa-square-root-variable"),(cc4,"PBIAS","fa-percent")]:
            metric_card(col,ic,f"{met_cal_u[k]:.3f}",k)
    with cv:
        if met_val_u:
            st.markdown(f"**Validation** &nbsp; {quality_badge(met_val_u['NSE'])}",unsafe_allow_html=True)
            vc1,vc2,vc3,vc4=st.columns(4)
            for col,k,ic in[(vc1,"NSE","fa-chart-bar"),(vc2,"KGE","fa-bullseye"),(vc3,"R2","fa-square-root-variable"),(vc4,"PBIAS","fa-percent")]:
                metric_card(col,ic,f"{met_val_u[k]:.3f}",k,teal=True)
        else: st.info("No validation period available.")

    sec_header("fa-chart-line","Visualisations")
    vt1,vt2,vt3=st.tabs(["  Hydrograph  ","  Flow Duration  ","  Seasonal Cycle  "])

    with vt1:
        psel=st.radio("Period:",["Calibration","Validation","Full record"],horizontal=True,key="ur_hyd2")
        mp2=mask_cal if psel=="Calibration" else(mask_val if psel=="Validation" else udf["Date"].notna())
        dt2=udf.loc[mp2,"Date"].values; Qo2=udf.loc[mp2,"Qobs"].values; Qs2=udf.loc[mp2,"Qsim"].values; P2=udf.loc[mp2,"P"].values
        m2=compute_metrics(Qo2,Qs2)
        fig_h2=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.22,0.78],vertical_spacing=0.04)
        fig_h2.add_trace(go.Bar(x=dt2,y=P2,name='P',marker_color='#3a7cbf',opacity=0.7),row=1,col=1)
        fig_h2.add_trace(go.Scatter(x=dt2,y=Qo2,name='Observed',line=dict(color='#4aa3e8',width=1.2)),row=2,col=1)
        fig_h2.add_trace(go.Scatter(x=dt2,y=Qs2,name='Simulated',line=dict(color='#ff6b6b',width=1.2,dash='dash')),row=2,col=1)
        fig_h2.update_yaxes(title_text="P (mm/d)",row=1,col=1,autorange='reversed',gridcolor='rgba(30,58,95,0.6)')
        fig_h2.update_yaxes(title_text="Q (m³/s)",row=2,col=1,gridcolor='rgba(30,58,95,0.6)')
        fig_h2.update_layout(**plotly_base(460),
                              title=dict(text=f"{basin_name}  ·  NSE={m2['NSE']:.3f}  ·  KGE={m2['KGE']:.3f}  ·  R²={m2['R2']:.3f}",
                                         font=dict(size=11,color='#8aabbf',family='Space Mono')),
                              legend=dict(orientation='h',y=-0.13))
        st.plotly_chart(fig_h2,use_container_width=True)

    with vt2:
        pairs=[(udf.loc[mask_cal,"Qobs"].values,udf.loc[mask_cal,"Qsim"].values,"Calibration")]
        if met_val_u: pairs.append((udf.loc[mask_val,"Qobs"].values,udf.loc[mask_val,"Qsim"].values,"Validation"))
        cols_fdc=st.columns(len(pairs))
        for idx,(Qo_f,Qs_f,ttl) in enumerate(pairs):
            Qo_s2=np.sort(Qo_f)[::-1]; Qs_s2=np.sort(Qs_f)[::-1]; pct2=np.arange(1,len(Qo_s2)+1)/len(Qo_s2)*100
            fig_f=go.Figure()
            fig_f.add_trace(go.Scatter(x=pct2,y=Qo_s2,name='Observed',line=dict(color='#4aa3e8',width=2)))
            fig_f.add_trace(go.Scatter(x=pct2,y=Qs_s2,name='Simulated',line=dict(color='#ff6b6b',width=2,dash='dash')))
            for qp in[50,90]: fig_f.add_vline(x=qp,line_dash='dot',line_color='rgba(90,90,90,0.5)',annotation_text=f'Q{qp}',annotation_font_size=9)
            mf=compute_metrics(Qo_f,Qs_f)
            fig_f.update_layout(**plotly_base(370),
                                 title=dict(text=f"FDC — {ttl} | NSE={mf['NSE']:.3f}",font=dict(size=11,color='#8aabbf',family='Space Mono')),
                                 xaxis_title="Exceedance %",yaxis_type='log',yaxis_title="Q (m³/s)")
            with cols_fdc[idx]: st.plotly_chart(fig_f,use_container_width=True)

    with vt3:
        ds2=udf[mask_cal].copy(); ds2['Month']=pd.to_datetime(ds2['Date']).dt.month
        mnths=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        fig_s2=make_subplots(specs=[[{"secondary_y":True}]])
        fig_s2.add_trace(go.Bar(x=mnths,y=ds2.groupby('Month')['P'].mean().values,name='P',marker_color='#3a7cbf',opacity=0.5),secondary_y=True)
        fig_s2.add_trace(go.Scatter(x=mnths,y=ds2.groupby('Month')['Qobs'].mean().values,name='Obs Q',line=dict(color='#4aa3e8',width=2.5),mode='lines+markers'),secondary_y=False)
        fig_s2.add_trace(go.Scatter(x=mnths,y=ds2.groupby('Month')['Qsim'].mean().values,name='Sim Q',line=dict(color='#ff6b6b',width=2.5,dash='dash'),mode='lines+markers'),secondary_y=False)
        fig_s2.update_yaxes(title_text="Mean Q (m³/s)",secondary_y=False,gridcolor='rgba(30,58,95,0.6)')
        fig_s2.update_yaxes(title_text="Mean P (mm/d)",secondary_y=True,autorange='reversed')
        fig_s2.update_layout(**plotly_base(400),title=dict(text=f"Seasonal Cycle — {basin_name}",font=dict(size=11,color='#8aabbf',family='Space Mono')))
        st.plotly_chart(fig_s2,use_container_width=True)

    sec_header("fa-download","Export Results")
    ex_df=udf[["Date","P","PET","Qobs","Qsim"]].copy()
    ex_df["Period"]="outside"; ex_df.loc[mask_cal,"Period"]="calibration"; ex_df.loc[mask_val,"Period"]="validation"
    ex_df["Residual"]=ex_df["Qobs"]-ex_df["Qsim"]
    cd1,cd2=st.columns(2)
    with cd1:
        st.download_button("Download simulated discharge (CSV)",
                            ex_df.to_csv(index=False,sep=";",decimal=",").encode(),
                            f"HYMOD_{basin_name.replace(' ','_')}_results.csv","text/csv",use_container_width=True)
    with cd2:
        rpt=pd.DataFrame({"Parameter":["Cmax","Bexp","Alpha","Ks","Kq"],"Value":[round(v,6) for v in params_u],"Unit":["mm","–","–","–","–"]})
        st.download_button("Download parameters & metrics (CSV)",rpt.to_csv(index=False).encode(),
                            f"HYMOD_{basin_name.replace(' ','_')}_params.csv","text/csv",use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION XIII — MAIN ROUTER
# ═════════════════════════════════════════════════════════════════════════════
def main():
    inject_css()
    if st.session_state.show_landing:
        show_landing(); return

    page=sidebar()

    with st.spinner("Initialising platform…"):
        df,CONV,params,met_train,met_test,mask_warmup,mask_train,mask_test=load_and_calibrate()

    if   page=="Dashboard":       page_dashboard(df,CONV,params,met_train,met_test,mask_train,mask_test)
    elif page=="Interactive Map":  page_map(df)
    elif page=="Model Results":    page_results(df,met_train,met_test,mask_train,mask_test)
    elif page=="Simulation Tool":  page_simulation(df,CONV,params,met_train,met_test,mask_train,mask_test)
    elif page=="Model Theory":     page_theory()
    elif page=="My Own Data":      page_my_data()

if __name__=="__main__":
    main()
