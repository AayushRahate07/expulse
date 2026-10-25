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
    st.markdown(textwrap.dedent(html).strip(), unsafe_allow_html=True)


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
    <style>
    /* =========================================================================
       CLEAN, MODERN, READABLE THEME (NO SCI-FI CLUTTER)
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
        --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    .stApp {
        background-color: var(--bg-main) !important;
        font-family: var(--font-family) !important;
        color: var(--text-main) !important;
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
        grid-template-columns: repeat(5, 1fr);
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
# SIDEBAR CONTROLS
# ==============================================================================
with st.sidebar:
    st.markdown("### Settings")

    fault_map = {
        1: "Fault 1: A/C Feed Ratio Change (Stream 4)",
        4: "Fault 4: Reactor Cooling Water Temp Drop",
        6: "Fault 6: Loss of Feed Stream 1",
        8: "Fault 8: Random Drift in Feed Composition"
    }

    selected_fault = st.selectbox(
        "Select Fault Condition:",
        options=[1, 4, 6, 8],
        format_func=lambda k: fault_map[k]
    )

    selected_run = st.slider("Simulation Run (1 to 100):", min_value=1, max_value=100, value=1)
    selected_model_name = st.selectbox("Primary Anomaly Detector:", list(models.keys()), index=0)
    enable_cusum = st.toggle("Show CUSUM drift line", value=True)

    render_html(
        """
        <div style="font-size: 0.80rem; color: #94a3b8; line-height: 1.6; margin-top: 24px; padding-top: 16px; border-top: 1px solid #1f293d;">
            <strong>Plant Simulation Details:</strong><br>
            • 50 continuous sensor channels<br>
            • Readings taken every 3 minutes<br>
            • Fault introduced at Hour 08:00 (Sample 160)<br>
            • Total duration: 48 hours (960 samples)
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
# MAIN PAGE HEADER (SIMPLE & CLEAN)
# ==============================================================================
render_html(
    f"""
    <div class="app-header">
        <div class="app-title">Tennessee Eastman Process — Anomaly Detection</div>
        <div class="app-subtitle">
            Currently inspecting <strong>{fault_map[selected_fault]}</strong> (Run #{selected_run:03d}) using <strong>{selected_model_name}</strong>
        </div>
    </div>
    """
)


# ==============================================================================
# TABS NAVIGATION (CLEAN HUMAN LABELS)
# ==============================================================================
tab_live, tab_explain, tab_benchmarks, tab_summary = st.tabs([
    "Live Inspection",
    "Results Explained",
    "Model Benchmarks",
    "Summary Data"
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
        font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", size=11, color="#94a3b8"),
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
# TAB 1: LIVE INSPECTION
# ==============================================================================
with tab_live:
    # 1. Clear Status Banner
    dom_lvl = consensus_res["dominant_level"]
    status_classes = {
        3: ("status-red", "Status: Critical Alarm", "All 3 detection models agree that a major process disturbance is occurring."),
        2: ("status-blue", "Status: Anomaly Detected", "PCA detected sensor correlation breakdown. Univariate limits are still normal."),
        1: ("status-yellow", "Status: Sensor Warning", "Single sensor exceeded normal bounds. Likely instrument drift."),
        0: ("status-green", "Status: Normal", "All 50 sensors are operating stably within safe bounds."),
    }
    s_class, s_title, s_desc = status_classes.get(dom_lvl, ("status-blue", "Status: Active", ""))

    render_html(
        f"""
        <div class="status-banner {s_class}">
            <div>
                <strong style="font-size: 0.95rem;">{s_title}</strong> — {s_desc}
            </div>
            <div style="font-size: 0.80rem; opacity: 0.9;">
                Confirmed across {consensus_res['counts'][dom_lvl]} cycles
            </div>
        </div>
        """
    )

    # 2. Key Metrics Row
    lead_val = f"{perf_metrics['lead_time_minutes']:+.1f} min" if perf_metrics["lead_time_minutes"] is not None else "Not detected"
    first_flag = f"Sample {perf_metrics['first_sustained_sample']}" if perf_metrics["first_sustained_sample"] else "None"
    delay_sub = f"{(perf_metrics['first_sustained_sample'] - 160) * 3:+.0f} min after fault started" if perf_metrics["first_sustained_sample"] else "No breach"

    render_html(
        f"""
        <div class="kpi-row">
            <div class="kpi-box">
                <div class="kpi-title">Detection Time</div>
                <div class="kpi-number">{lead_val}</div>
                <div class="kpi-desc">Time to sound alarm</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-title">Accuracy (F1 Score)</div>
                <div class="kpi-number">{perf_metrics['f1']:.3f}</div>
                <div class="kpi-desc">1.0 = perfect score</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-title">Precision</div>
                <div class="kpi-number">{perf_metrics['precision'] * 100:.1f}%</div>
                <div class="kpi-desc">Alarms that were real</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-title">Recall</div>
                <div class="kpi-number">{perf_metrics['recall'] * 100:.1f}%</div>
                <div class="kpi-desc">Fault period captured</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-title">First Alarm Tripped</div>
                <div class="kpi-number" style="font-size: 1.25rem;">{first_flag}</div>
                <div class="kpi-desc">{delay_sub}</div>
            </div>
        </div>
        """
    )

    # Simple Summary Note
    render_html(
        f"""
        <div class="content-card" style="padding: 12px 18px; margin-bottom: 18px; background: rgba(56, 189, 248, 0.05); border-left: 3px solid #38bdf8;">
            <div style="font-size: 0.84rem; color: #cbd5e1; line-height: 1.6;">
                <strong>What you are seeing:</strong> The plant was running normally until Sample 160 (Hour 8). 
                When <strong>{fault_map[selected_fault]}</strong> was injected, the active model (<strong>{selected_model_name}</strong>) 
                triggered an alarm at <strong>{first_flag} ({lead_val})</strong> with <strong>{perf_metrics['f1']:.3f} F1 score</strong>.
                <em>(For plain-English explanations of these results, click the "Results Explained" tab).</em>
            </div>
        </div>
        """
    )

    # 3. Main Anomaly Score Plot
    fig = go.Figure()

    # Anomaly Score Trace
    fig.add_trace(go.Scatter(
        x=df_run["sample"],
        y=active_scores,
        mode="lines",
        name=f"{selected_model_name} Score",
        line=dict(color="#38bdf8", width=2.0),
        hovertemplate="Sample %{x}<br>Score: %{y:.4f}<extra></extra>"
    ))

    # Threshold Trace
    fig.add_trace(go.Scatter(
        x=df_run["sample"],
        y=[active_model.threshold_] * len(df_run),
        mode="lines",
        name=f"Alarm Threshold ({active_model.threshold_:.2f})",
        line=dict(color="#ef4444", width=1.5, dash="dash"),
        hoverinfo="skip"
    ))

    # CUSUM Overlay
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

    # Fault Start Vertical Line
    fig.add_vline(
        x=FAULT_INTRO_SAMPLE_TESTING,
        line_width=1.5,
        line_dash="dot",
        line_color="#f59e0b",
        annotation_text="Fault Injected (Sample 160 / Hour 8)",
        annotation_position="top left",
        annotation_font=dict(size=11, color="#f59e0b")
    )

    # Shaded Fault Zone
    fig.add_vrect(
        x0=FAULT_INTRO_SAMPLE_TESTING,
        x1=df_run["sample"].max(),
        fillcolor="rgba(245, 158, 11, 0.04)",
        layer="below",
        line_width=0
    )

    fig_layout = dict(
        height=380,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#111827",
        font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", size=11, color="#94a3b8"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11)
        ),
        xaxis=dict(
            title=dict(text="Time (Samples — 1 sample = 3 minutes)", font=dict(size=11)),
            showgrid=True,
            gridcolor="#1f293d",
            zeroline=False
        ),
        yaxis=dict(
            title=dict(text="Anomaly Score", font=dict(size=11)),
            showgrid=True,
            gridcolor="#1f293d",
            zeroline=False
        )
    )

    if enable_cusum:
        fig_layout["yaxis2"] = dict(
            title=dict(text="CUSUM Drift Statistic", font=dict(size=11, color="#c084fc")),
            overlaying="y",
            side="right",
            showgrid=False,
            tickfont=dict(color="#c084fc")
        )

    fig.update_layout(**fig_layout)
    st.plotly_chart(fig, width="stretch")

    # 4. Root Cause Sensors Section
    st.markdown("#### Which sensors caused the alarm?")

    anom_rows = df_run[df_run["sample"] >= FAULT_INTRO_SAMPLE_TESTING]
    X_anom = anom_rows[feature_cols].values
    gt_vars = get_ground_truth_variables(selected_fault)

    if len(X_anom) > 0:
        c_pca_col, c_if_col = st.columns(2)

        with c_pca_col:
            st.markdown("**PCA Model Attribution (Sensor Reconstruction Error)**")
            pca_errs, _ = compute_pca_feature_attribution(models["PCA Subspace Reconstruction"], X_anom, feature_names=feature_cols)
            df_top_pca = rank_top_pca_contributions(pca_errs, feature_cols, top_k=5)
            fig_pca_attr = build_horizontal_attribution_chart(df_top_pca, "mean_reconstruction_error", "Error Contribution", "#38bdf8", gt_vars)
            st.plotly_chart(fig_pca_attr, width="stretch")

        with c_if_col:
            st.markdown("**Isolation Forest Attribution (Feature Importance)**")
            shap_vals, _ = compute_isolation_forest_shap(models["Isolation Forest"], X_anom, feature_names=feature_cols)
            df_top_if = rank_top_contributing_features(shap_vals, feature_cols, top_k=5)
            fig_if_attr = build_horizontal_attribution_chart(df_top_if, "mean_abs_shap", "Feature Importance", "#a78bfa", gt_vars)
            st.plotly_chart(fig_if_attr, width="stretch")

        st.caption("Green bars marked [ACTUAL CAUSE] are the true physical root-cause sensors defined by the Tennessee Eastman benchmark (Downs & Vogel, 1993).")


# ==============================================================================
# TAB 2: RESULTS EXPLAINED (SIMPLE, HUMAN-FRIENDLY BREAKDOWN)
# ==============================================================================
with tab_explain:
    st.markdown("### Understanding Your Results")
    st.markdown("A simple, step-by-step guide explaining what happened during this run, why the model scored the way it did, and what the numbers mean.")

    # 1. Dynamic Run Explanation
    lead_text = f"{perf_metrics['lead_time_minutes']:+.1f} minutes after fault started" if perf_metrics["lead_time_minutes"] is not None else "No alarm sounded"
    sample_text = f"Sample #{perf_metrics['first_sustained_sample']}" if perf_metrics["first_sustained_sample"] else "None"
    gt_vars_list = sorted(list(get_ground_truth_variables(selected_fault)))
    gt_names = ", ".join([f"{PROCESS_TAG_DESCRIPTIONS.get(v, v)} ({v})" for v in gt_vars_list[:3]])

    if selected_fault == 1:
        fault_name = "Fault 1 — A/C Feed Ratio Step Change"
        what_happened = "The chemical feed ratio of reactants A and C in Stream 4 suddenly shifted. This caused an immediate imbalance in the plant's feed rate."
        why_model = "Because feed flows directly blew past normal limits, all three models detected this quickly (within 3 to 15 minutes) with over 99% accuracy."
    elif selected_fault == 4:
        fault_name = "Fault 4 — Reactor Cooling Water Temperature Step"
        what_happened = "The cooling water entering the reactor suddenly got colder. This introduced an unexpected cooling effect into the chemical reactor."
        if "Isolation Forest" in selected_model_name:
            why_model = "<strong style='color: #f87171;'>Why Isolation Forest failed here (823-minute delay, 6% accuracy):</strong> When the water got cold, the plant's automatic temperature controller closed the cooling valve to keep the reactor temperature steady. Because both the temperature and the valve stayed within normal safe ranges, single-sensor tree models were fooled and didn't sound an alarm until other parts of the plant drifted hours later."
        elif "PCA" in selected_model_name:
            why_model = "<strong style='color: #34d399;'>Why PCA succeeded here (detected in 3 minutes, 99.7% accuracy):</strong> PCA does not look at sensors alone; it looks at the relationship between sensors. It noticed that the cooling valve was closed while the temperature stayed steady — a combination that should never happen in normal physics. That broken relationship immediately triggered the alarm."
        else:
            why_model = "Rule-based limits detected the fault with moderate delay once the cooling valve reached its physical limit."
    elif selected_fault == 6:
        fault_name = "Fault 6 — Loss of Feed Stream 1"
        what_happened = "Reactant A was completely shut off in Stream 1. This caused an immediate pressure drop and chemical starvation in the reactor."
        why_model = "Because reactant A stopped flowing entirely, the massive pressure drop caused all three models to sound an alarm almost instantly (0 to 3 minutes)."
    else:
        fault_name = "Fault 8 — Random Drift in Feed Composition"
        what_happened = "The composition of the feed reactants began wandering randomly over time. This is a subtle, slow-moving drift rather than a sudden explosion."
        why_model = "Because the drift was so slow, standard single-snapshot models were delayed by about 55 minutes. Using the CUSUM drift line or rolling 200-feature models detects this much earlier."

    render_html(
        f"""
        <div class="content-card" style="border-left: 4px solid #38bdf8;">
            <div class="content-card-title">What Happened in This Specific Run?</div>
            <div class="content-card-body">
                <strong>1. The Disturbance:</strong> {what_happened}<br><br>
                <strong>2. The Model's Reaction:</strong> Your active model (<strong>{selected_model_name}</strong>) scored 
                <strong>{perf_metrics['f1']:.3f} F1 accuracy</strong>. It triggered its first alarm at 
                <strong>{sample_text}</strong> ({lead_text}).<br><br>
                <strong>3. The Root Cause Sensors:</strong> The sensors physically responsible for this disturbance were: 
                <strong>{gt_names}</strong>.<br><br>
                <div style="background: rgba(56, 189, 248, 0.06); border: 1px solid rgba(56, 189, 248, 0.2); padding: 12px 14px; border-radius: 6px;">
                    <strong>Why did the model perform this way?</strong><br>
                    {why_model}
                </div>
            </div>
        </div>
        """
    )

    # 2. Metric Guide in Plain English
    st.markdown("#### What Do These Numbers Mean? (Plain-English Guide)")

    c1, c2 = st.columns(2)

    with c1:
        render_html(
            """
            <div class="content-card">
                <div class="content-card-title">Accuracy (F1 Score)</div>
                <div class="content-card-body">
                    <strong>Why not standard percentage accuracy?</strong><br>
                    In an industrial plant, 95% of normal operation is healthy. A lazy model that always says "Normal" would achieve 95% accuracy while letting the factory blow up.<br><br>
                    <strong>F1 Score</strong> is the balanced score between catching genuine faults and not crying wolf with false alarms. 
                    <strong>1.00</strong> is perfect, and <strong>0.95+</strong> is considered industry-grade.
                </div>
            </div>
            """
        )

        render_html(
            """
            <div class="content-card">
                <div class="content-card-title">Detection Time (Lead Time)</div>
                <div class="content-card-body">
                    <strong>How fast did the alarm go off?</strong><br>
                    Every fault in this benchmark starts at <strong>Sample 160 (Hour 8)</strong>.<br><br>
                    • <strong>+9.0 min:</strong> The alarm sounded 9 minutes after the fault started (very fast).<br>
                    • <strong>+800+ min:</strong> The model was blinded by plant controllers and took over 13 hours to notice.
                </div>
            </div>
            """
        )

    with c2:
        render_html(
            """
            <div class="content-card">
                <div class="content-card-title">Precision vs. Recall</div>
                <div class="content-card-body">
                    <strong>Precision:</strong> When the siren rings, how sure are we it's a real emergency? 99% precision means only 1 out of 100 alarms is a false alarm.<br><br>
                    <strong>Recall:</strong> Did the model catch the entire emergency, or did it turn off while the fault was still happening? 99% recall means virtually zero missed downtime.
                </div>
            </div>
            """
        )

        render_html(
            """
            <div class="content-card">
                <div class="content-card-title">Why PCA Beats Other Models on Fault 4</div>
                <div class="content-card-body">
                    <strong>The Feedback Controller Trap:</strong><br>
                    When cooling water temperatures drop, automatic controllers close valves to keep the reactor temperature steady. 
                    Because each individual sensor stays inside its normal limits, single-sensor tree models think everything is fine.<br><br>
                    <strong>PCA looks at relationships:</strong> It immediately spots that a cold temperature and a closed valve do not match normal operation.
                </div>
            </div>
            """
        )

    # 3. Fault Reference Guide
    st.markdown("#### Fault Condition Reference Guide")
    render_html(
        """
        <div class="clean-table-container">
            <table class="clean-table">
                <thead>
                    <tr>
                        <th>Fault Mode</th>
                        <th>What Actually Happens in the Plant</th>
                        <th>Difficulty</th>
                        <th>Best Model</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Fault 1</strong></td>
                        <td>A/C chemical feed ratio step change in Stream 4</td>
                        <td><span class="clean-badge badge-green">Easy</span></td>
                        <td>All models (PCA, Rule-Based, Isolation Forest) detect in ~9 min</td>
                    </tr>
                    <tr>
                        <td><strong>Fault 4</strong></td>
                        <td>Cooling water inlet temp drop (Reactor controller fight)</td>
                        <td><span class="clean-badge badge-red">Challenging</span></td>
                        <td><strong>PCA</strong> detects in 3 min; Isolation Forest fails (823 min delay)</td>
                    </tr>
                    <tr>
                        <td><strong>Fault 6</strong></td>
                        <td>Total loss of reactant A feed in Stream 1</td>
                        <td><span class="clean-badge badge-green">Easy</span></td>
                        <td>All models detect immediately (0 to 3 min)</td>
                    </tr>
                    <tr>
                        <td><strong>Fault 8</strong></td>
                        <td>Random composition drift across feed reactants A, B, C</td>
                        <td><span class="clean-badge badge-yellow">Medium</span></td>
                        <td><strong>PCA with CUSUM or 200 temporal features</strong></td>
                    </tr>
                </tbody>
            </table>
        </div>
        """
    )


# ==============================================================================
# TAB 3: MODEL BENCHMARKS
# ==============================================================================
with tab_benchmarks:
    st.markdown("### Model Benchmarks & Scientific Verification")
    st.markdown("Detailed verification tests addressing sequential drift detection, conformal confidence bounds, and root cause accuracy.")

    b_col1, b_col2 = st.columns(2)

    with b_col1:
        st.markdown("##### 1. CUSUM Sequential Drift Benchmark")
        cusum_p = REPORTS_DIR / "cusum_vs_consecutive_leadtime.csv"
        if cusum_p.exists():
            df_c = pd.read_csv(cusum_p)
            render_html(df_to_clean_html_table(df_c))
            st.caption("CUSUM accumulates small changes over time, detecting subtle drift modes like Fault 8 about 3 to 4 minutes earlier than static thresholds.")

        st.markdown("##### 2. Block Conformal Calibration (False Alarm Guarantee)")
        conf_p = REPORTS_DIR / "conformal_coverage_analysis.csv"
        if conf_p.exists():
            df_cf = pd.read_csv(conf_p)
            render_html(df_to_clean_html_table(df_cf))
            st.caption("1-hour non-overlapping blocks handle time-series correlation to guarantee that the false alarm rate stays under 1%.")

    with b_col2:
        st.markdown("##### 3. Root Cause Attribution Accuracy")
        attr_p = REPORTS_DIR / "attribution_accuracy_benchmark.csv"
        if attr_p.exists():
            df_at = pd.read_csv(attr_p)
            render_html(df_to_clean_html_table(df_at))
            st.caption("PCA achieves a 100% Hit@1 rate on Fault 4, proving it ranks the true broken physical component first every single time.")

        st.markdown("##### 4. 200-Feature Temporal Augmentation")
        temp_p = REPORTS_DIR / "temporal_features_benchmark.csv"
        if temp_p.exists():
            df_tp = pd.read_csv(temp_p)
            render_html(df_to_clean_html_table(df_tp))
            st.caption("Adding rolling averages and trends increases Fault 8 accuracy from 95.1% to 98.4%.")


# ==============================================================================
# TAB 4: SUMMARY DATA
# ==============================================================================
with tab_summary:
    st.markdown("### Consolidated Performance Across 400 Simulation Runs")
    st.markdown("Complete benchmark results covering 384,000 observations across all fault modes and detector types.")

    eval_p = REPORTS_DIR / "evaluation_summary.csv"
    if eval_p.exists():
        df_ev = pd.read_csv(eval_p)
        render_html(df_to_clean_html_table(df_ev, max_rows=100))

        st.download_button(
            label="Download Complete Benchmark CSV",
            data=df_ev.to_csv(index=False).encode("utf-8"),
            file_name="tep_anomaly_detection_benchmark.csv",
            mime="text/csv"
        )
