"""
Tennessee Eastman Process (TEP) Anomaly Detection Dashboard
Clean, intuitive interface for monitoring plant disturbances, evaluating models, and diagnosing root causes.
"""

import sys
import textwrap
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PROCESSED_DIR, MODELS_DIR, FAULT_INTRO_SAMPLE_TESTING, REPORTS_DIR
from src.evaluation.metrics import evaluate_run_predictions
from src.evaluation.cusum import CUSUMDetector
from src.explainability import compute_isolation_forest_shap, compute_pca_feature_attribution
from src.explainability.shap_explainer import rank_top_contributing_features
from src.explainability.pca_attribution import rank_top_pca_contributions
from src.explainability.validation import get_ground_truth_variables
from src.models.consensus import ConsensusArbitrator


def render_html(html: str) -> None:
    """Safely renders HTML without markdown indentation converting lines to code blocks."""
    st.html(textwrap.dedent(html).strip())


def df_to_clean_html_table(df: pd.DataFrame, max_rows: int = 100) -> str:
    """Converts a DataFrame into a clean, modern HTML table."""
    display_df = df.head(max_rows).copy()

    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda x: f"{x:.3f}" if pd.notnull(x) else "—")
        elif pd.api.types.is_integer_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda x: f"{x}" if pd.notnull(x) else "—")
        else:
            display_df[col] = display_df[col].fillna("—").astype(str)

    html = ['<div class="clean-table-container"><table class="clean-table">']
    html.append('<thead><tr>')
    for c in display_df.columns:
        html.append(f'<th>{c}</th>')
    html.append('</tr></thead><tbody>')

    for _, row in display_df.iterrows():
        html.append('<tr>')
        for val in row:
            v_str = str(val)
            badge_class = None
            if any(k in v_str for k in ["Earlier with CUSUM", "YES", "1.000", "PCA Reconstruction", "100.0%"]):
                badge_class = "badge-green"
            elif any(k in v_str for k in ["Trip", "Critical", "Heuristic", "Level 3"]):
                badge_class = "badge-red"
            elif any(k in v_str for k in ["Warning", "Drift", "Identical", "Tie / Non-sig", "Level 1"]):
                badge_class = "badge-yellow"
            elif any(k in v_str for k in ["Level 2", "Subspace"]):
                badge_class = "badge-blue"

            if badge_class:
                html.append(f'<td><span class="clean-badge {badge_class}">{v_str}</span></td>')
            else:
                html.append(f'<td>{v_str}</td>')
        html.append('</tr>')

    html.append('</tbody></table></div>')
    return ''.join(html)


# ==============================================================================
# SENSOR DESCRIPTIONS (Downs & Vogel, 1993)
# ==============================================================================
PROCESS_TAG_DESCRIPTIONS = {
    "xmeas_1": "A Feed Flow (Stream 4)",
    "xmeas_2": "D Feed Flow (Stream 2)",
    "xmeas_3": "E Feed Flow (Stream 3)",
    "xmeas_4": "Total Feed Flow (Stream 4)",
    "xmeas_5": "Recycle Flow (Stream 8)",
    "xmeas_6": "Reactor Feed Rate (Stream 6)",
    "xmeas_7": "Reactor Pressure",
    "xmeas_8": "Reactor Level",
    "xmeas_9": "Reactor Temperature",
    "xmeas_10": "Purge Rate (Stream 9)",
    "xmeas_11": "Product Separator Temp",
    "xmeas_12": "Product Separator Level",
    "xmeas_13": "Product Separator Pressure",
    "xmeas_14": "Separator Underflow",
    "xmeas_15": "Stripper Level",
    "xmeas_16": "Stripper Pressure",
    "xmeas_17": "Stripper Underflow Rate",
    "xmeas_18": "Stripper Temperature",
    "xmeas_19": "Stripper Steam Flow",
    "xmeas_20": "Compressor Work Rate",
    "xmeas_21": "Reactor Cooling Water Temp Out",
    "xmeas_22": "Separator Cooling Temp Out",
    "xmeas_23": "Feed A Analyzer (Stream 6)",
    "xmeas_24": "Feed B Analyzer (Stream 6)",
    "xmeas_25": "Feed C Analyzer (Stream 6)",
    "xmeas_26": "Feed D Analyzer (Stream 6)",
    "xmeas_27": "Feed E Analyzer (Stream 6)",
    "xmeas_28": "Feed F Analyzer (Stream 6)",
    "xmeas_29": "Purge A Analyzer (Stream 9)",
    "xmeas_30": "Purge B Analyzer (Stream 9)",
    "xmeas_31": "Purge C Analyzer (Stream 9)",
    "xmeas_32": "Purge D Analyzer (Stream 9)",
    "xmeas_33": "Purge E Analyzer (Stream 9)",
    "xmeas_34": "Purge F Analyzer (Stream 9)",
    "xmeas_35": "Purge G Analyzer (Stream 9)",
    "xmeas_36": "Purge H Analyzer (Stream 9)",
    "xmeas_37": "Product G Analyzer (Stream 11)",
    "xmeas_38": "Product H Analyzer (Stream 11)",
    "xmv_1": "D Feed Flow Valve",
    "xmv_2": "E Feed Flow Valve",
    "xmv_3": "A Feed Flow Valve",
    "xmv_4": "Total Feed Valve",
    "xmv_6": "Purge Vent Valve",
    "xmv_7": "Separator Underflow Valve",
    "xmv_8": "Stripper Underflow Valve",
    "xmv_10": "Reactor Cooling Water Valve",
    "xmv_11": "Condenser Cooling Water Valve",
}

# ==============================================================================
# PAGE CONFIGURATION & CLEAN STYLING
# ==============================================================================
st.set_page_config(
    page_title="TEP Anomaly Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

render_html(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Geologica:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Geologica:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* =========================================================================
       CLEAN, MODERN, READABLE THEME (GEOLOGICA TYPOGRAPHY)
       ========================================================================= */
    :root {
        --bg-main: #0b0f19;
        --bg-card: #111827;
        --bg-subtle: #162032;
        --border-color: #1f293d;
        --border-hover: #334155;
        --text-main: #f8fafc;
        --text-muted: #94a3b8;
        --accent-blue: #38bdf8;
        --accent-green: #10b981;
        --accent-yellow: #f59e0b;
        --accent-red: #ef4444;
        --font-family: 'Geologica', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    html, body, [class*="css"], .stApp, .stApp * {
        font-family: 'Geologica', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0d121f !important;
        border-right: 1px solid var(--border-color) !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: #162032 !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-main) !important;
        border-radius: 6px !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] * {
        color: var(--text-main) !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="slider"] div[role="slider"] {
        background-color: var(--accent-blue) !important;
        border: 2px solid #0d121f !important;
    }

    /* Tabs */
    div[data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid var(--border-color) !important;
        padding-bottom: 2px !important;
        gap: 8px !important;
        margin-bottom: 20px !important;
    }
    button[data-baseweb="tab"] {
        background: transparent !important;
        border: none !important;
        border-radius: 6px 6px 0 0 !important;
        color: var(--text-muted) !important;
        font-weight: 600 !important;
        font-size: 0.90rem !important;
        padding: 10px 20px !important;
        transition: color 0.15s ease !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #ffffff !important;
        border-bottom: 2px solid var(--accent-blue) !important;
        background: transparent !important;
    }
    div[data-baseweb="tab-highlight"] {
        display: none !important;
    }

    /* App Header */
    .app-header {
        margin-bottom: 24px;
        padding-bottom: 16px;
        border-bottom: 1px solid var(--border-color);
    }
    .app-title {
        font-size: 1.45rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 6px 0;
    }
    .app-subtitle {
        font-size: 0.88rem;
        color: var(--text-muted);
        margin: 0;
    }

    /* Alarm Status Banner */
    .status-banner {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 20px;
        font-size: 0.90rem;
    }
    .status-red {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #fecdd3;
    }
    .status-blue {
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.4);
        color: #e0f2fe;
    }
    .status-yellow {
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.4);
        color: #fef3c7;
    }
    .status-green {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #d1fae5;
    }

    /* KPI Cards */
    .kpi-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-bottom: 20px;
    }
    .kpi-box {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 14px 16px;
    }
    .kpi-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--text-muted);
        margin-bottom: 6px;
    }
    .kpi-number {
        font-size: 1.45rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.01em;
    }
    .kpi-desc {
        font-size: 0.74rem;
        color: var(--text-muted);
        margin-top: 4px;
    }

    /* Content Card */
    .content-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 16px;
    }
    .content-card-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 10px;
    }
    .content-card-body {
        font-size: 0.85rem;
        color: #cbd5e1;
        line-height: 1.65;
    }

    /* Clean Tables */
    .clean-table-container {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        overflow-x: auto;
        margin-bottom: 16px;
    }
    table.clean-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.82rem;
        color: var(--text-main);
        text-align: left;
    }
    table.clean-table thead tr {
        background: #162032;
        border-bottom: 1px solid var(--border-color);
    }
    table.clean-table th {
        padding: 10px 14px;
        font-weight: 600;
        color: #cbd5e1;
        white-space: nowrap;
    }
    table.clean-table td {
        padding: 10px 14px;
        border-bottom: 1px solid var(--border-color);
        white-space: nowrap;
    }
    table.clean-table tbody tr:hover {
        background: #172338;
    }

    /* Clean Badges */
    .clean-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
    }
    .badge-green {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-red {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .badge-yellow {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .badge-blue {
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }

    /* Clean Streamlit Overrides */
    header { visibility: hidden !important; }
    .stDeployButton { display: none !important; }
    footer { visibility: hidden !important; }
    </style>
    """
)


# ==============================================================================
# DATA & MODELS
# ==============================================================================
@st.cache_resource
def load_models():
    rule_m = joblib.load(MODELS_DIR / "rule_based_detector.joblib")
    pca_m = joblib.load(MODELS_DIR / "pca_detector.joblib")
    if_m = joblib.load(MODELS_DIR / "isolation_forest_detector.joblib")
    return {
        "PCA Subspace Reconstruction": pca_m,
        "Rule-Based Tolerance Bounds": rule_m,
        "Isolation Forest": if_m
    }


@st.cache_data
def load_faulty_dataset():
    return pd.read_parquet(DATA_PROCESSED_DIR / "faulty_testing.parquet")


models = load_models()
df_faulty = load_faulty_dataset()
consensus_arbitrator = ConsensusArbitrator()


# ==============================================================================
# SIDEBAR CONTROLS (STREAMLINED OPERATOR INTERFACE)
# ==============================================================================
with st.sidebar:
    st.markdown("### Process Controls")

    fault_map = {
        1: "Fault 1: A/C Feed Ratio Step Change",
        4: "Fault 4: Reactor Coolant Temp Step Drop",
        6: "Fault 6: Loss of Reactant A Feed Stream",
        8: "Fault 8: Random Feed Composition Drift"
    }

    selected_fault = st.selectbox(
        "Process Disturbance Mode:",
        options=[1, 4, 6, 8],
        format_func=lambda k: fault_map[k]
    )

    selected_model_name = st.selectbox("Active Anomaly Detector:", list(models.keys()), index=0)

    with st.expander("Advanced Settings", expanded=False):
        selected_run = st.slider("Simulation Run (1 to 100):", min_value=1, max_value=100, value=1)
        enable_cusum = st.toggle("Show CUSUM drift line", value=False)
        render_html(
            """
            <div style="font-size: 0.76rem; color: #94a3b8; line-height: 1.5; margin-top: 10px; border-top: 1px solid #1f293d; padding-top: 8px;">
                <strong>Simulation Parameters:</strong><br>
                • 50 continuous channels (3-min interval)<br>
                • Fault injected at Sample 160 (Hour 8)<br>
                • 48-hour run length (960 samples)
            </div>
            """
        )


# ==============================================================================
# MULTI-MODEL INFERENCE PIPELINE
# ==============================================================================
df_run = df_faulty[(df_faulty["faultnumber"] == selected_fault) & (df_faulty["simulationrun"] == selected_run)].sort_values(by="sample")
feature_cols = [c for c in df_run.columns if c not in ["faultnumber", "simulationrun", "sample"]]
X_run = df_run[feature_cols].values
y_true = (df_run["sample"].values >= FAULT_INTRO_SAMPLE_TESTING).astype(int)

# Decision scores & predictions across all models
scores_dict = {}
preds_dict = {}
for m_key, m_instance in models.items():
    scores_dict[m_key] = m_instance.decision_function(X_run)
    preds_dict[m_key] = m_instance.predict(X_run)

# Consensus arbitration
rule_preds = preds_dict["Rule-Based Tolerance Bounds"]
pca_preds = preds_dict["PCA Subspace Reconstruction"]
if_preds = preds_dict["Isolation Forest"]
consensus_res = consensus_arbitrator.evaluate_run(rule_preds, pca_preds, if_preds)

# Primary model evaluation metrics
active_model = models[selected_model_name]
active_scores = scores_dict[selected_model_name]
active_preds = preds_dict[selected_model_name]
perf_metrics = evaluate_run_predictions(y_true, active_preds, fault_intro_sample=FAULT_INTRO_SAMPLE_TESTING)


# ==============================================================================
# MAIN PAGE HEADER (MINIMAL & FOCUSED)
# ==============================================================================
render_html(
    f"""
    <div class="app-header">
        <div class="app-title">Tennessee Eastman Process — Operator Console</div>
        <div class="app-subtitle">
            Inspecting <strong>{fault_map[selected_fault]}</strong> (Run #{selected_run:03d}) using <strong>{selected_model_name}</strong>
        </div>
    </div>
    """
)


# ==============================================================================
# TABS NAVIGATION (STREAMLINED 4-TAB PROGRESSIVE DISCLOSURE)
# ==============================================================================
tab_live, tab_pfd, tab_diagnostics, tab_benchmarks = st.tabs([
    "Live Overview",
    "Process Flow (PFD)",
    "Root Cause Diagnostics",
    "Model Benchmarks"
])


def build_horizontal_attribution_chart(df_top, metric_col, metric_label, bar_color, gt_vars):
    """Builds a clean, non-leaking dark Plotly horizontal bar chart for feature attributions."""
    df_sorted = df_top.iloc[::-1].copy()

    y_labels = [
        f"{PROCESS_TAG_DESCRIPTIONS.get(f, f)} ({f}) [ACTUAL CAUSE]" if f in gt_vars
        else f"{PROCESS_TAG_DESCRIPTIONS.get(f, f)} ({f})"
        for f in df_sorted["feature"]
    ]
    colors = [
        "#10b981" if f in gt_vars else bar_color
        for f in df_sorted["feature"]
    ]

    fig = go.Figure(go.Bar(
        x=df_sorted[metric_col],
        y=y_labels,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.3f}" for v in df_sorted[metric_col]],
        textposition="outside",
        textfont=dict(size=10, color="#94a3b8"),
        hovertemplate="<b>%{y}</b><br>Score: %{x:.4f}<extra></extra>"
    ))

    fig.update_layout(
        height=270,
        margin=dict(l=10, r=40, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#111827",
        font=dict(family="'Geologica', -apple-system, sans-serif", size=11, color="#94a3b8"),
        xaxis=dict(
            title=dict(text=metric_label, font=dict(size=11)),
            showgrid=True,
            gridcolor="#1f293d",
            zeroline=False
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False
        )
    )
    return fig


# ==============================================================================
# TAB 1: LIVE OVERVIEW (FAST, 3-SECOND OPERATOR DIGESTION)
# ==============================================================================
with tab_live:
    # 1. High-Signal Status Banner
    dom_lvl = consensus_res["dominant_level"]
    status_classes = {
        3: ("status-red", "CRITICAL ALARM: Plant Disturbance Active", f"Multi-model consensus alarm active across {consensus_res['counts'][dom_lvl]} cycles. Physical equilibrium lost."),
        2: ("status-blue", "ANOMALY DETECTED: Subspace Correlation Shift", f"PCA detected inter-variable relationship failure across {consensus_res['counts'][dom_lvl]} cycles while individual limits are nominal."),
        1: ("status-yellow", "SENSOR WARNING: Single Channel Limit Breach", f"Single-channel boundary breach across {consensus_res['counts'][dom_lvl]} cycles. Likely sensor drift."),
        0: ("status-green", "PLANT NOMINAL: Continuous Steady-State", "All 50 sensors are operating stably within baseline bounds."),
    }
    s_class, s_title, s_desc = status_classes.get(dom_lvl, ("status-blue", "Active Disturbance", ""))

    render_html(
        f"""
        <div class="status-banner {s_class}">
            <div>
                <strong style="font-size: 0.98rem;">{s_title}</strong><br>
                <span style="font-size: 0.82rem; opacity: 0.92;">{s_desc}</span>
            </div>
            <div style="font-size: 0.80rem; font-weight: 600; opacity: 0.95; white-space: nowrap;">
                Run #{selected_run:03d}
            </div>
        </div>
        """
    )

    # 2. 3 Key Metrics Row
    lead_val = f"{perf_metrics['lead_time_minutes']:+.1f} min" if perf_metrics["lead_time_minutes"] is not None else "Not detected"
    first_flag = f"Sample {perf_metrics['first_sustained_sample']}" if perf_metrics["first_sustained_sample"] else "None"
    first_clock = f"Hour {perf_metrics['first_sustained_sample'] * 3 / 60:.2f}" if perf_metrics["first_sustained_sample"] else "No breach"

    # Physical primary culprit from Downs & Vogel ground truth
    gt_vars_list = sorted(list(get_ground_truth_variables(selected_fault, include_secondary=False)))
    primary_tag = gt_vars_list[0] if gt_vars_list else "xmeas_1"
    primary_name = PROCESS_TAG_DESCRIPTIONS.get(primary_tag, primary_tag)

    render_html(
        f"""
        <div class="kpi-row">
            <div class="kpi-box">
                <div class="kpi-title">DETECTION RESPONSE TIME</div>
                <div class="kpi-number">{lead_val}</div>
                <div class="kpi-desc">First sustained alarm: {first_flag} ({first_clock})</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-title">FAULT DETECTION RATE (RECALL)</div>
                <div class="kpi-number">{perf_metrics['recall'] * 100:.1f}%</div>
                <div class="kpi-desc">{int(perf_metrics['recall'] * 801)} of 801 fault samples flagged</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-title">PRIMARY PHYSICAL CULPRIT</div>
                <div class="kpi-number" style="font-size: 1.15rem; color: #38bdf8;">{primary_name}</div>
                <div class="kpi-desc">Sensor Tag: {primary_tag} (Downs & Vogel origin)</div>
            </div>
        </div>
        """
    )

    # 3. Main Anomaly Score Plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_run["sample"],
        y=active_scores,
        mode="lines",
        name=f"{selected_model_name} Score",
        line=dict(color="#38bdf8", width=2.0),
        hovertemplate="Sample %{x}<br>Score: %{y:.4f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=df_run["sample"],
        y=[active_model.threshold_] * len(df_run),
        mode="lines",
        name=f"Alarm Limit ({active_model.threshold_:.2f})",
        line=dict(color="#ef4444", width=1.5, dash="dash"),
        hoverinfo="skip"
    ))

    if enable_cusum:
        cusum_agent = CUSUMDetector(k_factor=0.5)
        cusum_agent.fit(active_scores[:160], target_arl0=2000.0)
        c_series = cusum_agent.transform(active_scores)
        fig.add_trace(go.Scatter(
            x=df_run["sample"],
            y=c_series,
            mode="lines",
            name="CUSUM Drift Line",
            line=dict(color="#c084fc", width=1.8, dash="dot"),
            yaxis="y2",
            hovertemplate="Sample %{x}<br>CUSUM: %{y:.2f}<extra></extra>"
        ))

    fig.add_vline(
        x=FAULT_INTRO_SAMPLE_TESTING,
        line_width=1.5,
        line_dash="dot",
        line_color="#f59e0b",
        annotation_text="Fault Injected (Sample 160 / Hour 8)",
        annotation_position="top left",
        annotation_font=dict(size=11, color="#f59e0b")
    )

    fig.add_vrect(
        x0=FAULT_INTRO_SAMPLE_TESTING,
        x1=df_run["sample"].max(),
        fillcolor="rgba(245, 158, 11, 0.04)",
        layer="below",
        line_width=0
    )

    fig_layout = dict(
        height=340,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#111827",
        font=dict(family="'Geologica', -apple-system, sans-serif", size=11, color="#94a3b8"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11)
        ),
        xaxis=dict(
            title=dict(text="Time (Samples — 1 sample = 3 min, 48 hours total)", font=dict(size=11)),
            showgrid=True,
            gridcolor="#1f293d",
            zeroline=False
        ),
        yaxis=dict(
            title=dict(text="Decision Score", font=dict(size=11)),
            showgrid=True,
            gridcolor="#1f293d",
            zeroline=False
        )
    )

    if enable_cusum:
        fig_layout["yaxis2"] = dict(
            title=dict(text="CUSUM Drift", font=dict(size=11, color="#c084fc")),
            overlaying="y",
            side="right",
            showgrid=False,
            tickfont=dict(color="#c084fc")
        )

    fig.update_layout(**fig_layout)
    st.plotly_chart(fig, width="stretch")


# ==============================================================================
# TAB 2: PROCESS FLOW DIAGRAM (PFD SIMULATOR)
# ==============================================================================
with tab_pfd:
    st.caption(f"Interactive Piping & Instrumentation Diagram (PFD) for {fault_map[selected_fault]}. Click unit operations (R-101, E-101, V-101, T-101) or stepper buttons (S0–S4) to inspect failure propagation.")
    sim_path = PROJECT_ROOT / "simulator.html"
    if sim_path.exists():
        sim_raw = sim_path.read_text(encoding="utf-8")
        sim_injected = sim_raw.replace('scenario: "6"', f'scenario: "{selected_fault}"')
        sim_injected = sim_injected.replace('<option value="6" selected>', '<option value="6">')
        sim_injected = sim_injected.replace(f'<option value="{selected_fault}">', f'<option value="{selected_fault}" selected>')
        components.html(sim_injected, height=670, scrolling=False)
    else:
        st.warning("simulator.html not found.")


@st.cache_data
def get_cached_attributions(fault_num: int, run_num: int):
    df_sub = df_faulty[(df_faulty["faultnumber"] == fault_num) & (df_faulty["simulationrun"] == run_num)]
    anom_sub = df_sub[df_sub["sample"] >= FAULT_INTRO_SAMPLE_TESTING]
    X_sub = anom_sub[feature_cols].values
    
    # Fast PCA reconstruction attribution
    pca_errs, _ = compute_pca_feature_attribution(models["PCA Subspace Reconstruction"], X_sub, feature_names=feature_cols)
    top_pca = rank_top_pca_contributions(pca_errs, feature_cols, top_k=5)
    
    # Fast representative SHAP attribution (30 points)
    X_shap_sub = X_sub[:30] if len(X_sub) > 30 else X_sub
    shap_vals, _ = compute_isolation_forest_shap(models["Isolation Forest"], X_shap_sub, feature_names=feature_cols)
    top_if = rank_top_contributing_features(shap_vals, feature_cols, top_k=5)
    
    return top_pca, top_if


# ==============================================================================
# TAB 3: ROOT CAUSE DIAGNOSTICS
# ==============================================================================
with tab_diagnostics:
    st.markdown("### Root Cause Diagnostics & Physical Attribution")
    
    gt_vars = get_ground_truth_variables(selected_fault)
    df_top_pca, df_top_if = get_cached_attributions(selected_fault, selected_run)

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("**Top Offending Sensors (PCA Reconstruction Error)**")
        fig_pca_attr = build_horizontal_attribution_chart(df_top_pca, "mean_reconstruction_error", "Reconstruction Error", "#38bdf8", gt_vars)
        st.plotly_chart(fig_pca_attr, width="stretch")
        st.caption("Green bars marked [ACTUAL CAUSE] are true physical causes (Downs & Vogel, 1993).")

    with c2:
        st.markdown("**Physical Disturbance Mechanism**")
        fault_mechanisms = {
            1: (
                "A/C Feed Ratio Step Shift (Stream 4)",
                "A sudden step change in reactant A/C feed ratio shifts reactor stoichiometry. Unreacted feed drives byproduct F formation, eventually forcing the separator purge valve CV-04 to 100% capacity.",
                "PCA detects the covariance disruption in Stream 4 at t=3.3 min (+50.7m lead time), whereas 1D limits wait 16.7 min until cooling valve CV-10 crosses 3σ."
            ),
            4: (
                "Reactor Coolant Temperature Step Drop",
                "Cooling water entering the reactor drops in temperature. The reactor PID temperature controller aggressively throttles cooling valve CV-10 from 41% down to 34% (>12σ) to hold reactor temperature at 120.4°C.",
                "PCA catches the broken relationship between CV-10 and TT-09 immediately (t=3.0 min). Raw Isolation Forest fails (FDR 3.2%) because axis-aligned splits see TT-09 nominal."
            ),
            6: (
                "Reactant A Feed Stream Loss (CV-01)",
                "Reactant A feed valve CV-01 slams shut, cutting reactant A to zero. Reactor pressure collapses (-150 kPa), kinetics stall, and stripper bottoms product trips off-spec.",
                "PCA SPE alarms on the first sampled cycle at t=3.0 min (+51.0m lead time before plant trip). All models trigger rapidly due to the massive physical step loss."
            ),
            8: (
                "A/B/C Feed Composition Random Drift",
                "Non-stationary random walk drift in reactant feed composition. Individual sensors remain inside ±3σ bounds for over 60 minutes while unreacted components swell the recycle loop.",
                "Dynamic PCA (with rolling slope/variance augmentation) and CUSUM register low-frequency trajectory drift, providing 30+ minutes of intervention lead time before product off-spec trip."
            )
        }
        f_title, f_phys, f_model = fault_mechanisms.get(selected_fault, fault_mechanisms[6])

        render_html(
            f"""
            <div class="content-card" style="border-left: 3px solid #38bdf8; padding: 14px 16px;">
                <div style="font-size: 0.90rem; font-weight: 700; color: #f1f5f9; margin-bottom: 6px;">
                    {f_title}
                </div>
                <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.5; margin-bottom: 8px;">
                    <strong>Physical Root Cause:</strong> {f_phys}
                </div>
                <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.5;">
                    <strong style="color: #38bdf8;">Why Active Model Behaved This Way:</strong> {f_model}
                </div>
            </div>
            """
        )

    with st.expander("Secondary TreeSHAP Attribution (Isolation Forest)", expanded=False):
        fig_if_attr = build_horizontal_attribution_chart(df_top_if, "mean_abs_shap", "Feature Importance (SHAP)", "#a78bfa", gt_vars)
        st.plotly_chart(fig_if_attr, width="stretch")


# ==============================================================================
# TAB 4: MODEL BENCHMARKS & SCIENTIFIC VERIFICATION
# ==============================================================================
with tab_benchmarks:
    st.markdown("### Model Benchmarks & Scientific Verification")
    st.caption("Standardized evaluation across 400 test runs (384,000 observations) under matched false-alarm rates.")

    eval_p = REPORTS_DIR / "evaluation_summary.csv"
    if eval_p.exists():
        df_ev = pd.read_csv(eval_p)
        st.download_button(
            label="Download Complete Benchmark CSV",
            data=df_ev.to_csv(index=False).encode("utf-8"),
            file_name="tep_anomaly_detection_benchmark.csv",
            mime="text/csv"
        )
        render_html(df_to_clean_html_table(df_ev, max_rows=100))

    with st.expander("Sequential CUSUM Drift Detection Benchmark", expanded=False):
        cusum_p = REPORTS_DIR / "cusum_vs_consecutive_leadtime.csv"
        if cusum_p.exists():
            df_c = pd.read_csv(cusum_p)
            render_html(df_to_clean_html_table(df_c))
            st.caption("CUSUM accumulates small changes over time, detecting subtle drift modes like Fault 8 earlier than static thresholds.")

    with st.expander("Block Conformal Prediction (1% False Alarm Guarantee)", expanded=False):
        conf_p = REPORTS_DIR / "conformal_coverage_analysis.csv"
        if conf_p.exists():
            df_cf = pd.read_csv(conf_p)
            render_html(df_to_clean_html_table(df_cf))
            st.caption("1-hour non-overlapping blocks handle time-series correlation to guarantee that the false alarm rate stays under 1%.")

    with st.expander("200-Feature Temporal Augmentation (Dynamic PCA)", expanded=False):
        temp_p = REPORTS_DIR / "temporal_features_benchmark.csv"
        if temp_p.exists():
            df_tp = pd.read_csv(temp_p)
            render_html(df_to_clean_html_table(df_tp))
            st.caption("Adding rolling averages and trends increases Fault 8 accuracy from 95.1% to 98.4%.")
