import os
import time
import cv2
import gdown
import torch
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

try:
    import av
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    WEBRTC_AVAILABLE = True
except Exception:
    WEBRTC_AVAILABLE = False

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEHAZE_MODEL_PATH = "remove_hazy_model_256x256.pth"
DEHAZE_GDRIVE_ID  = "1ji3x-KO19X2yGpT7oaUIpJ5DiCgQg8xS"
YOLO_MODEL_NAME   = "yolov8n.pt"

st.set_page_config(
    page_title="NEXTGEN VISION AI",
    page_icon="🟢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:ital,wght@0,400;0,700;1,400&display=swap');

/* ── Reset & Base ─────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif !important;
    background: #06090f !important;
    color: #eef4ff !important;
}

.stApp {
    background: #06090f !important;
}

/* hide streamlit chrome */
header[data-testid="stHeader"]   { display: none !important; }
.stDeployButton                  { display: none !important; }
#MainMenu                        { display: none !important; }
footer                           { display: none !important; }

/* ── Sidebar ──────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0b1018 !important;
    border-right: 1px solid rgba(0,255,180,0.12) !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

/* ── Sidebar logo block ───────────────────── */
.nv-logo {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 24px 20px 18px;
    border-bottom: 1px solid rgba(0,255,180,0.10);
    margin-bottom: 6px;
}
.nv-logo-n {
    width: 38px; height: 38px;
    background: #00ffb4;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Space Mono', monospace;
    font-weight: 700; font-size: 18px;
    color: #06090f;
    flex-shrink: 0;
    border-radius: 4px;
}
.nv-logo-text  { font-size: 13px; font-weight: 800; letter-spacing: 0.1em; color: #eef4ff; line-height: 1.2; }
.nv-logo-sub   { font-size: 9px; font-family: 'Space Mono', monospace; color: #4a6070; letter-spacing: 0.1em; margin-top: 2px; }

/* ── Status dots ──────────────────────────── */
.nv-status-bar { padding: 12px 20px 8px; border-bottom: 1px solid rgba(0,255,180,0.08); margin-bottom: 4px; }
.nv-status-item {
    display: flex; align-items: center; gap: 8px;
    font-size: 11px; font-family: 'Space Mono', monospace;
    color: #4a6070; padding: 3px 0;
}
.nv-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.nv-dot-ok   { background: #00ffb4; box-shadow: 0 0 6px #00ffb4; }
.nv-dot-warn { background: #ff6b35; box-shadow: 0 0 6px #ff6b35; }
.nv-dot-off  { background: #2a3a4a; }

/* ── Sidebar section labels ───────────────── */
.nv-section-label {
    font-size: 9px; font-family: 'Space Mono', monospace;
    letter-spacing: 0.2em; text-transform: uppercase;
    color: #2e4455; padding: 14px 20px 6px;
}

/* ── Slider label rows ────────────────────── */
.ctrl-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 2px 20px 0;
    font-size: 11px; font-family: 'Space Mono', monospace; color: #4a6070;
}

/* ── Streamlit slider override ────────────── */
[data-testid="stSlider"] {
    padding: 0 20px !important;
}
[data-testid="stSlider"] .stSlider { padding: 0 !important; }
[data-testid="stSlider"] > div > div > div > div {
    background: #00ffb4 !important;
}

/* ── Streamlit toggle override ────────────── */
[data-testid="stToggle"] {
    padding: 2px 20px !important;
}
[data-testid="stToggle"] label {
    font-size: 12px !important;
    font-family: 'Space Mono', monospace !important;
    color: #4a6070 !important;
}
[data-testid="stToggle"] [data-testid="stWidgetLabel"] p {
    font-size: 11px !important;
    font-family: 'Space Mono', monospace !important;
    color: #4a6070 !important;
}

/* ── Selectbox override ───────────────────── */
[data-testid="stSelectbox"] {
    padding: 0 20px !important;
}
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: #111820 !important;
    border: 1px solid rgba(0,255,180,0.18) !important;
    border-radius: 6px !important;
    color: #eef4ff !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
}

/* ── Radio override ───────────────────────── */
[data-testid="stRadio"] {
    padding: 0 20px !important;
}
[data-testid="stRadio"] label {
    font-size: 12px !important;
    font-family: 'Space Mono', monospace !important;
    color: #4a6070 !important;
}
[data-testid="stRadio"] [data-testid="stWidgetLabel"] p {
    font-size: 9px !important; letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    font-family: 'Space Mono', monospace !important;
    color: #2e4455 !important;
}

/* ── Main block container ─────────────────── */
.block-container {
    max-width: 100% !important;
    padding: 0 !important;
}

/* ── Top bar ──────────────────────────────── */
.nv-topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 18px 36px;
    border-bottom: 1px solid rgba(0,255,180,0.10);
    background: rgba(11,16,24,0.6);
    backdrop-filter: blur(8px);
    position: sticky; top: 0; z-index: 100;
}
.nv-topbar-path {
    font-size: 11px; font-family: 'Space Mono', monospace;
    letter-spacing: 0.18em; color: #2e4455; text-transform: uppercase;
}
.nv-topbar-path span { color: #00ffb4; }
.nv-chip-row { display: flex; gap: 8px; align-items: center; }
.nv-chip {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 12px;
    background: rgba(0,255,180,0.05);
    border: 1px solid rgba(0,255,180,0.15);
    border-radius: 4px;
    font-size: 9px; font-family: 'Space Mono', monospace;
    color: #4a6070; letter-spacing: 0.12em;
}
.nv-chip-dot { width: 5px; height: 5px; border-radius: 50%; background: #00ffb4; }

/* ── Hero ─────────────────────────────────── */
.nv-hero {
    padding: 40px 36px 28px;
    position: relative; overflow: hidden;
}
.nv-hero::before {
    content: '';
    position: absolute; top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(0,255,180,0.06), transparent 70%);
    pointer-events: none;
}
.nv-hero-headline {
    font-size: 52px; font-weight: 800;
    line-height: 0.95; letter-spacing: -0.04em;
    color: #eef4ff; margin-bottom: 14px;
}
.nv-hero-headline em {
    font-style: normal; color: #00ffb4;
    display: block;
}
.nv-hero-desc {
    font-size: 13px; font-family: 'Space Mono', monospace;
    color: #4a6070; line-height: 1.8;
    max-width: 520px; margin-bottom: 20px;
}
.nv-badge-row { display: flex; gap: 8px; flex-wrap: wrap; }
.nv-badge {
    font-size: 9px; font-family: 'Space Mono', monospace;
    letter-spacing: 0.12em; padding: 5px 10px;
    border: 1px solid rgba(0,255,180,0.3);
    border-radius: 3px; color: #00ffb4;
    text-transform: uppercase;
}

/* ── Metric cards ─────────────────────────── */
.nv-metrics { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; padding: 0 36px 28px; }
.nv-metric {
    background: #0b1018;
    border: 1px solid rgba(0,255,180,0.12);
    border-radius: 8px;
    padding: 18px 20px;
    position: relative; overflow: hidden;
}
.nv-metric::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #00ffb4, #00c8ff);
}
.nv-metric-val {
    font-size: 22px; font-weight: 800;
    font-family: 'Space Mono', monospace;
    letter-spacing: -0.02em; color: #eef4ff;
    margin-bottom: 6px; margin-top: 4px;
}
.nv-metric-lbl {
    font-size: 9px; letter-spacing: 0.18em;
    text-transform: uppercase; color: #2e4455;
    font-family: 'Space Mono', monospace;
}
.nv-metric-sub { font-size: 10px; color: #00ffb4; font-family: 'Space Mono', monospace; margin-top: 4px; }

/* ── Content area ─────────────────────────── */
.nv-content { padding: 0 36px 40px; }

/* ── Section header ───────────────────────── */
.nv-section-head {
    display: flex; align-items: center; gap: 14px;
    margin-bottom: 16px;
}
.nv-section-title { font-size: 20px; font-weight: 800; letter-spacing: -0.02em; color: #eef4ff; }
.nv-section-tag {
    font-size: 9px; font-family: 'Space Mono', monospace;
    letter-spacing: 0.18em; padding: 4px 9px;
    border: 1px solid rgba(0,255,180,0.3);
    color: #00ffb4; border-radius: 3px; text-transform: uppercase;
}

/* ── Info / warning banners ───────────────── */
.nv-info {
    background: rgba(0,200,255,0.06);
    border: 1px solid rgba(0,200,255,0.2);
    border-radius: 6px; padding: 12px 16px;
    font-size: 12px; font-family: 'Space Mono', monospace;
    color: rgba(0,200,255,0.85); margin-bottom: 18px;
    line-height: 1.7;
}
.nv-warn {
    background: rgba(255,107,53,0.07);
    border: 1px solid rgba(255,107,53,0.22);
    border-radius: 6px; padding: 12px 16px;
    font-size: 12px; font-family: 'Space Mono', monospace;
    color: rgba(255,160,80,0.9); margin-bottom: 18px;
    line-height: 1.7;
}

/* ── File uploader ────────────────────────── */
[data-testid="stFileUploader"] {
    background: rgba(0,255,180,0.02) !important;
    border: 1px dashed rgba(0,255,180,0.28) !important;
    border-radius: 10px !important;
    padding: 1.5rem !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,255,180,0.55) !important;
}
[data-testid="stFileUploader"] label {
    font-family: 'Space Mono', monospace !important;
    color: #4a6070 !important; font-size: 12px !important;
}
[data-testid="stFileUploader"] button {
    background: rgba(0,255,180,0.10) !important;
    border: 1px solid rgba(0,255,180,0.30) !important;
    color: #00ffb4 !important;
    border-radius: 5px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
}
[data-testid="stFileUploader"] button:hover {
    background: rgba(0,255,180,0.20) !important;
}

/* ── Images ───────────────────────────────── */
[data-testid="stImage"] img {
    border-radius: 8px !important;
    border: 1px solid rgba(0,255,180,0.15) !important;
}

/* ── Image caption override ───────────────── */
.nv-img-card {
    background: #0b1018;
    border: 1px solid rgba(0,255,180,0.12);
    border-radius: 10px;
    overflow: hidden;
}
.nv-img-card-head {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 14px;
    border-bottom: 1px solid rgba(0,255,180,0.10);
}
.nv-img-card-title {
    font-size: 9px; font-family: 'Space Mono', monospace;
    letter-spacing: 0.18em; text-transform: uppercase; color: #4a6070;
}
.nv-img-status {
    font-size: 9px; font-family: 'Space Mono', monospace;
    color: #00ffb4; background: rgba(0,255,180,0.10);
    padding: 2px 8px; border-radius: 3px; letter-spacing: 0.1em;
}
.nv-img-status-warn {
    color: #ff6b35; background: rgba(255,107,53,0.10);
}

/* ── Vis metrics ──────────────────────────── */
.nv-vis-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin-top: 20px; }
.nv-vis-bar {
    background: #0b1018;
    border: 1px solid rgba(0,255,180,0.12);
    border-radius: 8px; padding: 14px 16px;
}
.nv-vis-lbl {
    font-size: 9px; font-family: 'Space Mono', monospace;
    letter-spacing: 0.16em; text-transform: uppercase;
    color: #2e4455; margin-bottom: 10px;
}
.nv-vis-track {
    height: 3px; background: rgba(255,255,255,0.06);
    border-radius: 2px; margin-bottom: 8px; overflow: hidden;
}
.nv-vis-fill { height: 100%; border-radius: 2px; }
.nv-vis-val {
    font-size: 16px; font-weight: 700;
    font-family: 'Space Mono', monospace;
}

/* ── Detection table ──────────────────────── */
.nv-det-table {
    background: #0b1018;
    border: 1px solid rgba(0,255,180,0.12);
    border-radius: 10px;
    overflow: hidden; margin-top: 20px;
}
.nv-det-table-head {
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 18px;
    border-bottom: 1px solid rgba(0,255,180,0.10);
}
.nv-det-title {
    font-size: 10px; font-family: 'Space Mono', monospace;
    letter-spacing: 0.18em; text-transform: uppercase; color: #4a6070;
}
.nv-det-count {
    font-size: 10px; font-family: 'Space Mono', monospace;
    color: #00ffb4;
}

/* ── Dataframe override ───────────────────── */
[data-testid="stDataFrame"] {
    background: transparent !important;
}
[data-testid="stDataFrame"] table {
    background: transparent !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
    color: #eef4ff !important;
}
[data-testid="stDataFrame"] thead tr th {
    background: rgba(0,255,180,0.06) !important;
    color: #00ffb4 !important;
    font-size: 9px !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    border-color: rgba(0,255,180,0.12) !important;
}
[data-testid="stDataFrame"] tbody tr:hover td {
    background: rgba(0,255,180,0.04) !important;
}

/* ── Buttons ──────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, rgba(0,255,180,0.15), rgba(0,200,255,0.10)) !important;
    border: 1px solid rgba(0,255,180,0.35) !important;
    color: #00ffb4 !important;
    border-radius: 7px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 12px !important;
    letter-spacing: 0.12em !important;
    padding: 10px 24px !important;
    transition: all 0.15s !important;
    box-shadow: 0 0 20px rgba(0,255,180,0.08) !important;
}
.stButton > button:hover {
    background: rgba(0,255,180,0.22) !important;
    box-shadow: 0 0 28px rgba(0,255,180,0.18) !important;
    border-color: rgba(0,255,180,0.60) !important;
}

/* ── Progress bar ─────────────────────────── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #00ffb4, #00c8ff) !important;
    border-radius: 2px !important;
}
[data-testid="stProgressBar"] > div {
    background: rgba(0,255,180,0.10) !important;
    border-radius: 2px !important;
}

/* ── st.metric override ───────────────────── */
[data-testid="stMetric"] {
    background: #0b1018 !important;
    border: 1px solid rgba(0,255,180,0.12) !important;
    border-radius: 8px !important;
    padding: 16px 18px !important;
    position: relative !important;
    overflow: hidden !important;
}
[data-testid="stMetric"]::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #00ffb4, #00c8ff);
}
[data-testid="stMetricLabel"] {
    font-size: 9px !important;
    font-family: 'Space Mono', monospace !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    color: #2e4455 !important;
}
[data-testid="stMetricValue"] {
    font-size: 20px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    color: #eef4ff !important;
}

/* ── st.video ─────────────────────────────── */
[data-testid="stVideo"] video {
    border-radius: 10px !important;
    border: 1px solid rgba(0,255,180,0.15) !important;
}

/* ── Alerts ───────────────────────────────── */
[data-testid="stAlert"] {
    background: #0b1018 !important;
    border: 1px solid rgba(0,255,180,0.18) !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
}

/* ── Expander ─────────────────────────────── */
[data-testid="stExpander"] {
    background: #0b1018 !important;
    border: 1px solid rgba(0,255,180,0.12) !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary {
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    color: #4a6070 !important;
    letter-spacing: 0.1em !important;
}

/* ── Download button ──────────────────────── */
[data-testid="stDownloadButton"] > button {
    background: rgba(0,255,180,0.10) !important;
    border: 1px solid rgba(0,255,180,0.30) !important;
    color: #00ffb4 !important;
    border-radius: 6px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
}

/* ── Divider ──────────────────────────────── */
hr { border-color: rgba(0,255,180,0.10) !important; margin: 8px 0 !important; }

/* ── Success/error overrides ──────────────── */
.stSuccess { background: rgba(0,255,180,0.06) !important; border: 1px solid rgba(0,255,180,0.2) !important; border-radius: 6px !important; font-family: 'Space Mono', monospace !important; font-size: 11px !important; }
.stError   { background: rgba(255,60,60,0.06) !important; border: 1px solid rgba(255,60,60,0.2) !important; border-radius: 6px !important; font-family: 'Space Mono', monospace !important; font-size: 11px !important; }

/* ── Spinner ──────────────────────────────── */
[data-testid="stSpinner"] { color: #00ffb4 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPER COMPONENTS
# ─────────────────────────────────────────────
def topbar(page: str):
    parts = page.split("/")
    path_html = " <span style='color:#2e4455'>/</span> ".join(
        [f"<span style='color:#00ffb4'>{p}</span>" if i == len(parts)-1
         else f"<span style='color:#2e4455'>{p}</span>"
         for i, p in enumerate(parts)]
    )
    st.markdown(f"""
    <div class="nv-topbar">
      <div class="nv-topbar-path">/ {path_html}</div>
      <div class="nv-chip-row">
        <div class="nv-chip"><div class="nv-chip-dot"></div>SYSTEM ONLINE</div>
        <div class="nv-chip">DCP + RESNET</div>
        <div class="nv-chip">YOLOV8N</div>
      </div>
    </div>""", unsafe_allow_html=True)

def hero():
    st.markdown("""
    <div class="nv-hero">
      <div class="nv-hero-headline">NEXTGEN<em>VISION AI</em></div>
      <div class="nv-hero-desc">
        Real-time AR vision enhancement for adverse weather driving conditions.<br>
        Visibility restoration · hazard detection · performance metrics.
      </div>
      <div class="nv-badge-row">
        <div class="nv-badge">DCP + ResNet Dehazing</div>
        <div class="nv-badge">YOLOv8 Detection</div>
        <div class="nv-badge">Image · Video · Live</div>
      </div>
    </div>""", unsafe_allow_html=True)

def metric_cards(device, enable_detection, inference_size):
    yolo_val = "ON" if enable_detection else "OFF"
    yolo_sub = "conf ≥ 0.35" if enable_detection else "disabled"
    st.markdown(f"""
    <div class="nv-metrics">
      <div class="nv-metric">
        <div class="nv-metric-lbl">Dehazing Model</div>
        <div class="nv-metric-val">READY</div>
        <div class="nv-metric-sub">DCP + ResNet</div>
      </div>
      <div class="nv-metric">
        <div class="nv-metric-lbl">Compute Device</div>
        <div class="nv-metric-val">{str(device).upper()}</div>
        <div class="nv-metric-sub">torch — active</div>
      </div>
      <div class="nv-metric">
        <div class="nv-metric-lbl">YOLO Detection</div>
        <div class="nv-metric-val">{yolo_val}</div>
        <div class="nv-metric-sub">{yolo_sub}</div>
      </div>
      <div class="nv-metric">
        <div class="nv-metric-lbl">Inference Size</div>
        <div class="nv-metric-val">{inference_size}px</div>
        <div class="nv-metric-sub">balanced mode</div>
      </div>
    </div>""", unsafe_allow_html=True)

def section_head(title: str, tag: str):
    st.markdown(f"""
    <div class="nv-section-head">
      <div class="nv-section-title">{title}</div>
      <div class="nv-section-tag">{tag}</div>
    </div>""", unsafe_allow_html=True)

def info_box(msg: str):
    st.markdown(f'<div class="nv-info">ℹ &nbsp; {msg}</div>', unsafe_allow_html=True)

def warn_box(msg: str):
    st.markdown(f'<div class="nv-warn">⚡ &nbsp; {msg}</div>', unsafe_allow_html=True)

def img_card(title: str, status: str, img_rgb, warn=False):
    status_cls = "nv-img-status-warn" if warn else ""
    st.markdown(f"""
    <div class="nv-img-card">
      <div class="nv-img-card-head">
        <div class="nv-img-card-title">{title}</div>
        <div class="nv-img-status {status_cls}">{status}</div>
      </div>
    </div>""", unsafe_allow_html=True)
    st.image(img_rgb, use_container_width=True)

def vis_bars(orig_score, enh_score, proc_time, det_count):
    o_pct = min(orig_score, 100)
    e_pct = min(enh_score, 100)
    t_pct = min(int(proc_time / 5 * 100), 100)
    d_pct = min(det_count * 10, 100)
    st.markdown(f"""
    <div class="nv-vis-grid">
      <div class="nv-vis-bar">
        <div class="nv-vis-lbl">Orig. Visibility</div>
        <div class="nv-vis-track"><div class="nv-vis-fill" style="width:{o_pct}%;background:#ff6b35"></div></div>
        <div class="nv-vis-val" style="color:#ff6b35">{o_pct}%</div>
      </div>
      <div class="nv-vis-bar">
        <div class="nv-vis-lbl">Enhanced Vis.</div>
        <div class="nv-vis-track"><div class="nv-vis-fill" style="width:{e_pct}%;background:#00ffb4"></div></div>
        <div class="nv-vis-val" style="color:#00ffb4">{e_pct}%</div>
      </div>
      <div class="nv-vis-bar">
        <div class="nv-vis-lbl">Process Time</div>
        <div class="nv-vis-track"><div class="nv-vis-fill" style="width:{t_pct}%;background:#00c8ff"></div></div>
        <div class="nv-vis-val" style="color:#00c8ff">{proc_time:.2f}s</div>
      </div>
      <div class="nv-vis-bar">
        <div class="nv-vis-lbl">Objects Found</div>
        <div class="nv-vis-track"><div class="nv-vis-fill" style="width:{d_pct}%;background:#00ffb4"></div></div>
        <div class="nv-vis-val" style="color:#00ffb4">{det_count}</div>
      </div>
    </div>""", unsafe_allow_html=True)

def det_table_header(count: int):
    st.markdown(f"""
    <div class="nv-det-table">
      <div class="nv-det-table-head">
        <div class="nv-det-title">Detection Results</div>
        <div class="nv-det-count">{count} object(s) detected</div>
      </div>
    </div>""", unsafe_allow_html=True)

def sidebar_logo(dehaze_ready: bool, yolo_ready: bool):
    dehaze_dot = "nv-dot-ok" if dehaze_ready else "nv-dot-warn"
    yolo_dot   = "nv-dot-ok" if yolo_ready   else "nv-dot-off"
    dehaze_txt = "Dehazing model — ready" if dehaze_ready else "Dehazing model — missing"
    yolo_txt   = "YOLOv8 — available"    if yolo_ready   else "YOLOv8 — not installed"
    st.markdown(f"""
    <div class="nv-logo">
      <div class="nv-logo-n">N</div>
      <div>
        <div class="nv-logo-text">NEXTGEN VISION</div>
        <div class="nv-logo-sub">AR ENHANCEMENT SYSTEM v2</div>
      </div>
    </div>
    <div class="nv-status-bar">
      <div class="nv-status-item"><div class="nv-dot {dehaze_dot}"></div>{dehaze_txt}</div>
      <div class="nv-status-item"><div class="nv-dot {yolo_dot}"></div>{yolo_txt}</div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────
def download_dehaze_model_if_needed():
    if os.path.exists(DEHAZE_MODEL_PATH):
        return True, None
    try:
        gdown.download(f"https://drive.google.com/uc?id={DEHAZE_GDRIVE_ID}", DEHAZE_MODEL_PATH, quiet=False)
        return (os.path.exists(DEHAZE_MODEL_PATH), None if os.path.exists(DEHAZE_MODEL_PATH) else "Downloaded but file not found")
    except Exception as e:
        return False, str(e)

class GuidedFilter(nn.Module):
    def __init__(self, r=40, eps=1e-3):
        super().__init__(); self.r=r; self.eps=eps; self.boxfilter=nn.AvgPool2d(2*r+1,1,r)
    def forward(self,I,p):
        N=self.boxfilter(torch.ones(p.size(),device=p.device,dtype=p.dtype)); mean_I=self.boxfilter(I)/N; mean_p=self.boxfilter(p)/N; mean_Ip=self.boxfilter(I*p)/N; cov_Ip=mean_Ip-mean_I*mean_p; mean_II=self.boxfilter(I*I)/N; var_I=mean_II-mean_I*mean_I; a=cov_Ip/(var_I+self.eps); b=mean_p-a*mean_I; return (self.boxfilter(a)/N)*I+self.boxfilter(b)/N

class DCPDehazeGenerator(nn.Module):
    def __init__(self, win_size=15, r=40, eps=1e-3):
        super().__init__(); self.guided_filter=GuidedFilter(r,eps); self.neighborhood_size=win_size; self.omega=.95
    def get_dark_channel(self,img,w):
        img,_=torch.min(img,dim=1); img=torch.unsqueeze(img,dim=1); p=int(np.floor(w/2)); pads=[p,p,p,p] if w%2 else [p,p-1,p,p-1]; return -F.max_pool2d(-F.pad(img,pads,mode='replicate'),kernel_size=w,stride=1)
    def atmospheric_light(self,img,dark_img):
        num,chl,h,w=img.shape; top=max(int(.001*h*w),1); A=torch.zeros(num,chl,1,1,device=img.device,dtype=img.dtype)
        for n in range(num):
            _,idx=dark_img[n,0].reshape(h*w).sort(descending=True)
            for c in range(chl): A[n,c,0,0]=torch.mean(img[n,c].reshape(h*w)[idx[:top]])
        return A
    def forward(self,x):
        guidance=(.2989*x[:,0]+.5870*x[:,1]+.1140*x[:,2]) if x.shape[1]>1 else x[:,0]; guidance=torch.unsqueeze((guidance+1)/2,1); img=(x+1)/2; _,_,h,w=img.shape; dark=self.get_dark_channel(img,self.neighborhood_size); A=self.atmospheric_light(img,dark); map_A=A.repeat(1,1,h,w).clamp(min=1e-6); trans=1-self.omega*self.get_dark_channel(img/map_A,self.neighborhood_size); trans=trans.clamp(.05,1); T=self.guided_filter(guidance,trans).clamp(.05,1); return ((img-map_A)/T.repeat(1,3,1,1)+map_A).clamp(0,1)

class ResnetBlock(nn.Module):
    def __init__(self,dim,padding_type,norm_layer,use_dropout,use_bias):
        super().__init__(); block=[]
        for i in range(2):
            block += [nn.ReflectionPad2d(1)] if padding_type=='reflect' else []; p=0 if padding_type=='reflect' else 1
            block += [nn.Conv2d(dim,dim,3,1,p,bias=use_bias), norm_layer(dim)]
            if i==0: block += [nn.ReLU(True)] + ([nn.Dropout(.5)] if use_dropout else [])
        self.conv_block=nn.Sequential(*block)
    def forward(self,x): return x+self.conv_block(x)

class ResnetGenerator(nn.Module):
    def __init__(self,input_nc,output_nc,ngf=64,norm_layer=nn.BatchNorm2d,use_dropout=False,n_blocks=9,padding_type='reflect'):
        super().__init__(); use_bias=norm_layer==nn.InstanceNorm2d; model=[nn.ReflectionPad2d(3),nn.Conv2d(input_nc,ngf,7,padding=0,bias=use_bias),norm_layer(ngf),nn.ReLU(True)]
        for i in range(2):
            m=2**i; model += [nn.Conv2d(ngf*m,ngf*m*2,3,2,1,bias=use_bias),norm_layer(ngf*m*2),nn.ReLU(True)]
        for _ in range(n_blocks): model += [ResnetBlock(ngf*4,padding_type,norm_layer,use_dropout,use_bias)]
        for i in range(2):
            m=2**(2-i); model += [nn.ConvTranspose2d(ngf*m,int(ngf*m/2),3,2,1,output_padding=1,bias=use_bias),norm_layer(int(ngf*m/2)),nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3),nn.Conv2d(ngf,output_nc,7,padding=0,bias=use_bias),nn.Tanh()]; self.model=nn.Sequential(*model)
    def forward(self,x): return torch.clamp(self.model(x),-1,1)


# ─────────────────────────────────────────────
# MODEL LOADERS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_dehaze_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dcp = DCPDehazeGenerator().to(device).eval()
    resnet = ResnetGenerator(3, 3, norm_layer=nn.InstanceNorm2d).to(device)
    ckpt = torch.load(DEHAZE_MODEL_PATH, map_location=device)
    sd = ckpt[next((k for k in ['params','state_dict','model','net_g','generator'] if isinstance(ckpt,dict) and k in ckpt), None)] if isinstance(ckpt,dict) and any(k in ckpt for k in ['params','state_dict','model','net_g','generator']) else ckpt
    sd = {k.replace('module.',''): v for k,v in sd.items()}
    missing, unexpected = resnet.load_state_dict(sd, strict=False)
    resnet.eval()
    return dcp, resnet, device, missing, unexpected

@st.cache_resource(show_spinner=False)
def load_yolo_model():
    return YOLO(YOLO_MODEL_NAME) if YOLO_AVAILABLE else None


# ─────────────────────────────────────────────
# PROCESSING HELPERS
# ─────────────────────────────────────────────
def bgr_to_tensor(img, size):
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    return torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0) * 2 - 1

def tensor_to_bgr(tensor, hw):
    out = tensor.squeeze(0).detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
    bgr = cv2.cvtColor((out * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
    h, w = hw
    return cv2.resize(bgr, (w, h), interpolation=cv2.INTER_CUBIC)

def dehaze_image(img_bgr, strength=1.0, dcp_only=False, inference_size=192):
    h, w = img_bgr.shape[:2]
    dcp, resnet, device, _, _ = load_dehaze_models()
    x = bgr_to_tensor(img_bgr, inference_size).to(device)
    with torch.no_grad():
        dcp_out = dcp(x)
        refined  = dcp_out if dcp_only else (resnet(dcp_out) + 1) / 2
        result   = tensor_to_bgr(refined, (h, w))
    return cv2.addWeighted(img_bgr, 1 - strength, result, strength, 0) if strength < 1 else result

DRIVING_CLASSES = {'person','bicycle','car','motorcycle','bus','truck','traffic light','stop sign'}

def detect_objects_yolo(img_bgr, conf_threshold=0.35, only_driving_classes=True, draw_ar_style=True):
    model = load_yolo_model()
    if model is None:
        return img_bgr, []
    res = model(img_bgr, conf=conf_threshold, verbose=False)[0]
    annotated = img_bgr.copy()
    det = []
    if res.boxes is None:
        return annotated, det
    for box in res.boxes:
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        name = model.names[cls]
        if only_driving_classes and name not in DRIVING_CLASSES:
            continue
        x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().astype(int)
        det.append({'class': name, 'confidence': round(conf,2), 'box': [int(x1),int(y1),int(x2),int(y2)]})
        color = (0,255,180) if name not in ['person','motorcycle','bicycle'] else (53,107,255)
        if name in ['traffic light','stop sign']:
            color = (255,107,0)
        cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
        label = f'{name.upper()} {conf:.2f}'
        ly = max(y1 - 10, 25)
        cv2.rectangle(annotated, (x1, ly-24), (x1 + max(130, len(label)*12), ly+5), color, -1)
        cv2.putText(annotated, label, (x1+6, ly-5), cv2.FONT_HERSHEY_SIMPLEX, .55, (2,6,23), 2, cv2.LINE_AA)
        cv2.circle(annotated, (int((x1+x2)/2), int((y1+y2)/2)), 4, color, -1)
    return annotated, det

def draw_system_overlay(img_bgr, mode='IMAGE', fps=None, inference_time=None, detection_count=0):
    out = img_bgr.copy()
    cv2.putText(out, f'NEXTGEN VISION AI | {mode}', (15,30), cv2.FONT_HERSHEY_SIMPLEX, .72, (0,255,180), 2, cv2.LINE_AA)
    line = f'Objects: {detection_count}'
    line += f' | FPS: {fps:.1f}' if fps is not None else ''
    line += f' | Time: {inference_time:.2f}s' if inference_time is not None else ''
    cv2.putText(out, line, (15,60), cv2.FONT_HERSHEY_SIMPLEX, .55, (0,200,255), 2, cv2.LINE_AA)
    return out

def visibility_score(img_bgr):
    gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    contrast = float(gray.std())
    sharp    = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return min(100, max(0, int(contrast * 1.4 + (sharp ** .5) * 2))), round(contrast,2), round(sharp,2)

def process_pipeline(frame, strength, dcp_only, inference_size, enable_detection, conf, only_classes, ar_style, mode):
    t0 = time.time()
    dehazed = dehaze_image(frame, strength, dcp_only, inference_size)
    dt = time.time() - t0
    t1 = time.time()
    final, dets = detect_objects_yolo(dehazed, conf, only_classes, ar_style) if enable_detection else (dehazed, [])
    yt = time.time() - t1
    return dehazed, draw_system_overlay(final, mode=mode, inference_time=dt+yt, detection_count=len(dets)), dets, dt, yt


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    sidebar_logo(
        dehaze_ready=os.path.exists(DEHAZE_MODEL_PATH),
        yolo_ready=YOLO_AVAILABLE,
    )

    st.markdown('<div class="nv-section-label">Input Mode</div>', unsafe_allow_html=True)
    app_mode = st.radio(
        "Mode",
        ['Image Upload', 'Video Upload', 'Live Camera'],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown('<div class="nv-section-label">Enhancement</div>', unsafe_allow_html=True)
    strength       = st.slider('Enhancement strength',  0.0, 1.0, 1.0, 0.05, label_visibility="visible")
    dcp_only       = st.toggle('DCP only mode',        value=False)
    inference_size = st.selectbox('Inference size (px)', [128, 192, 256], index=1)

    st.markdown('<div class="nv-section-label">Object Detection</div>', unsafe_allow_html=True)
    enable_detection      = st.toggle('Enable YOLO detection',   value=True)
    conf_threshold        = st.slider('Confidence threshold',  0.10, 0.90, 0.35, 0.05)
    only_driving_classes  = st.toggle('Driving classes only',    value=True)
    draw_ar_style         = st.toggle('AR-style overlay',        value=True)

    st.markdown('<div class="nv-section-label">Video Settings</div>', unsafe_allow_html=True)
    video_max_frames  = st.slider('Max frames to process', 10, 180, 60, 10)
    video_frame_skip  = st.slider('Process every Nth frame', 1, 10, 3, 1)

    st.markdown("""
    <div style="padding:16px 20px 20px;font-size:10px;font-family:'Space Mono',monospace;
    color:#2e4455;line-height:1.8;border-top:1px solid rgba(0,255,180,0.08);margin-top:12px">
    Best flow: Image for quality demo · Video for full pipeline · Live as prototype.
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL DOWNLOAD & LOAD
# ─────────────────────────────────────────────
if not os.path.exists(DEHAZE_MODEL_PATH):
    with st.spinner('Downloading dehazing model from Google Drive ...'):
        ok, err = download_dehaze_model_if_needed()
    if ok:
        st.rerun()
    else:
        st.error(f'Could not download model: {err}')
        st.stop()

try:
    with st.spinner('Loading AI models ...'):
        dcp_model, resnet_model, device, missing_keys, unexpected_keys = load_dehaze_models()
        if enable_detection and YOLO_AVAILABLE:
            yolo_model = load_yolo_model()
except Exception as e:
    st.error(f'Model loading failed: {e}')
    st.stop()


# ─────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────
if app_mode == 'Image Upload':
    topbar("NEXTGEN VISION AI / Image Enhancement")
    hero()
    metric_cards(device, enable_detection, inference_size)

    st.markdown('<div style="padding:0 36px 0">', unsafe_allow_html=True)
    section_head("Image Enhancement", "MODE: IMAGE")
    info_box("Upload a hazy or foggy road image — compare original, dehazed, and final AR detection output side by side.")

    uploaded = st.file_uploader(
        'Upload a hazy / foggy road image',
        type=['jpg','jpeg','png'],
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded:
        rgb = np.array(Image.open(uploaded).convert('RGB'))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        with st.spinner('Running pipeline ...'):
            deh, final, dets, dt, yt = process_pipeline(
                bgr, strength, dcp_only, inference_size,
                enable_detection, conf_threshold,
                only_driving_classes, draw_ar_style, 'IMAGE'
            )

        st.markdown('<div style="padding:0 36px">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="nv-img-card">
              <div class="nv-img-card-head">
                <div class="nv-img-card-title">Original Input</div>
                <div class="nv-img-status nv-img-status-warn">RAW</div>
              </div>
            </div>""", unsafe_allow_html=True)
            st.image(rgb, use_container_width=True)
        with col2:
            st.markdown("""
            <div class="nv-img-card">
              <div class="nv-img-card-head">
                <div class="nv-img-card-title">Dehazed Output</div>
                <div class="nv-img-status">ENHANCED</div>
              </div>
            </div>""", unsafe_allow_html=True)
            st.image(cv2.cvtColor(deh, cv2.COLOR_BGR2RGB), use_container_width=True)
        with col3:
            st.markdown("""
            <div class="nv-img-card">
              <div class="nv-img-card-head">
                <div class="nv-img-card-title">AR Detection Output</div>
                <div class="nv-img-status">LIVE AR</div>
              </div>
            </div>""", unsafe_allow_html=True)
            st.image(cv2.cvtColor(final, cv2.COLOR_BGR2RGB), use_container_width=True)

        oscore, _, _ = visibility_score(bgr)
        escore, _, _ = visibility_score(deh)
        vis_bars(oscore, escore, dt + yt, len(dets))

        if dets:
            det_table_header(len(dets))
            st.dataframe(pd.DataFrame(dets), use_container_width=True, hide_index=True)
        else:
            st.info('No driving-related objects detected in this frame.')

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="padding:0 36px">
          <div style="text-align:center;padding:60px 32px;border:1px dashed rgba(0,255,180,0.2);
          border-radius:10px;background:rgba(0,255,180,0.01);margin-top:8px">
            <div style="font-size:36px;margin-bottom:16px;opacity:0.3">⬆</div>
            <div style="font-size:14px;font-weight:700;color:#eef4ff;margin-bottom:8px">No image loaded</div>
            <div style="font-size:11px;font-family:'Space Mono',monospace;color:#2e4455">
              Use the file uploader above to get started
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    with st.expander('System Diagnostics'):
        st.markdown(f"""
        <div style="font-family:'Space Mono',monospace;font-size:11px;color:#4a6070;line-height:2">
        Dehazing missing keys: {len(missing_keys)}<br>
        Dehazing unexpected keys: {len(unexpected_keys)}<br>
        YOLO available: {YOLO_AVAILABLE}<br>
        WebRTC available: {WEBRTC_AVAILABLE}<br>
        Torch device: {device}<br>
        CUDA available: {torch.cuda.is_available()}
        </div>""", unsafe_allow_html=True)


elif app_mode == 'Video Upload':
    topbar("NEXTGEN VISION AI / Video Upload")
    hero()
    metric_cards(device, enable_detection, inference_size)

    st.markdown('<div style="padding:0 36px">', unsafe_allow_html=True)
    section_head("Video Processing", "MODE: VIDEO")
    info_box("Best mode for presentations — shows dehazing + object detection together without webcam lag. Upload any road video and hit Process.")

    uploaded_video = st.file_uploader(
        'Upload a hazy / foggy road video',
        type=['mp4','avi','mov','mkv'],
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_video:
        inp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        inp.write(uploaded_video.read())
        inp.close()

        st.markdown('<div style="padding:0 36px">', unsafe_allow_html=True)
        st.video(inp.name)

        if st.button('⚡ PROCESS VIDEO', use_container_width=False):
            cap = cv2.VideoCapture(inp.name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = fps if fps and fps > 0 else 10
            w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            out = cv2.VideoWriter(
                out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                max(1, fps / video_frame_skip), (w, h)
            )

            prog    = st.progress(0)
            status  = st.empty()
            preview = st.empty()
            idx = processed = total_det = 0
            start = time.time()

            while cap.isOpened() and processed < video_max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                idx += 1
                if idx % video_frame_skip != 0:
                    continue
                deh, final, dets, dt, yt = process_pipeline(
                    frame, strength, dcp_only, inference_size,
                    enable_detection, conf_threshold,
                    only_driving_classes, draw_ar_style, 'VIDEO'
                )
                total_det += len(dets)
                out.write(final)
                processed += 1
                if processed % 3 == 0:
                    preview.image(
                        cv2.cvtColor(final, cv2.COLOR_BGR2RGB),
                        caption=f'Frame {processed} / {video_max_frames}',
                        use_container_width=True,
                    )
                prog.progress(min(processed / video_max_frames, 1.0))
                status.markdown(
                    f'<div class="nv-info">Processing frame {processed} / {video_max_frames} &nbsp;·&nbsp; Latest detections: {len(dets)}</div>',
                    unsafe_allow_html=True,
                )

            cap.release()
            out.release()
            total = time.time() - start

            st.success('Video processing complete.')
            st.video(out_path)

            c1,c2,c3,c4 = st.columns(4)
            c1.metric('Frames Processed', processed)
            c2.metric('Total Time',        f'{total:.1f}s')
            c3.metric('Avg / Frame',       f'{total/max(processed,1):.2f}s')
            c4.metric('Total Detections',  total_det)

            with open(out_path, 'rb') as f:
                st.download_button(
                    '↓ DOWNLOAD PROCESSED VIDEO',
                    data=f,
                    file_name='nextgen_vision_processed.mp4',
                    mime='video/mp4',
                    use_container_width=False,
                )
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="padding:0 36px">
          <div style="text-align:center;padding:60px 32px;border:1px dashed rgba(0,255,180,0.2);
          border-radius:10px;background:rgba(0,255,180,0.01)">
            <div style="font-size:36px;margin-bottom:16px;opacity:0.3">🎬</div>
            <div style="font-size:14px;font-weight:700;color:#eef4ff;margin-bottom:8px">No video loaded</div>
            <div style="font-size:11px;font-family:'Space Mono',monospace;color:#2e4455">
              Upload a road video above to begin
            </div>
          </div>
        </div>""", unsafe_allow_html=True)


else:  # Live Camera
    topbar("NEXTGEN VISION AI / Live Camera")
    hero()
    metric_cards(device, enable_detection, inference_size)

    st.markdown('<div style="padding:0 36px">', unsafe_allow_html=True)
    section_head("Live Camera", "MODE: LIVE")
    warn_box("Camera preview is a prototype demonstration — use Video Upload for a stable full-pipeline presentation.")

    if not WEBRTC_AVAILABLE:
        st.error('streamlit-webrtc is not installed. Add it to requirements.txt.')
        st.stop()

    rtc_config = RTCConfiguration({'iceServers': [{'urls': ['stun:stun.l.google.com:19302']}]})

    class LiveProcessor(VideoProcessorBase):
        def __init__(self):
            self.last_time   = time.time()
            self.fps         = 0.
            self.frame_count = 0
            self.cached_frame = None
            self.cached_dets  = []

        def recv(self, frame):
            img = frame.to_ndarray(format='bgr24')
            start = time.time()
            try:
                deh = dehaze_image(img, strength, dcp_only, 128)
                self.frame_count += 1
                if enable_detection and self.frame_count % 5 == 0:
                    final, dets = detect_objects_yolo(deh, conf_threshold, only_driving_classes, draw_ar_style)
                    self.cached_frame = final.copy()
                    self.cached_dets  = dets
                elif self.cached_frame is not None:
                    final = self.cached_frame.copy()
                    dets  = self.cached_dets
                else:
                    final = deh
                    dets  = []
                now = time.time()
                dt  = now - self.last_time
                self.last_time = now
                self.fps = 1 / dt if dt > 0 else self.fps
                final = draw_system_overlay(final, 'LIVE', self.fps, time.time()-start, len(dets))
            except Exception:
                final = img
            return av.VideoFrame.from_ndarray(final, format='bgr24')

    webrtc_streamer(
        key='nextgen-live-camera',
        video_processor_factory=LiveProcessor,
        rtc_configuration=rtc_config,
        media_stream_constraints={
            'video': {'width': {'ideal': 320}, 'height': {'ideal': 240}, 'frameRate': {'ideal': 8, 'max': 10}},
            'audio': False,
        },
        async_processing=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)
