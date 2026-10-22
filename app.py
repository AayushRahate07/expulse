"""
Streamlit Results Viewer App: Tennessee Eastman Process Anomaly Detection.
Section 8 / Week 10 Deliverable.
"""

import sys
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PROCESSED_DIR, MODELS_DIR, FAULT_INTRO_SAMPLE_TESTING, SAMPLING_INTERVAL_MINUTES
from src.evaluation.metrics import evaluate_run_predictions
from src.explainability import compute_isolation_forest_shap, compute_pca_feature_attribution
from src.explainability.shap_explainer import rank_top_contributing_features
from src.explainability.pca_attribution import rank_top_pca_contributions


st.set_page_config(
    page_title="TEP Anomaly Detection Viewer",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 Multivariate Industrial Anomaly Detection - TEP Benchmark")
st.markdown(
    "**Comparative Evaluation**: Rule-Based Thresholds vs. PCA Reconstruction Error vs. Isolation Forest with TreeSHAP Attribution."
)


@st.cache_resource
def load_models():
    """Cache loaded models."""
    rule_m = joblib.load(MODELS_DIR / "rule_based_detector.joblib")
    pca_m = joblib.load(MODELS_DIR / "pca_detector.joblib")
    if_m = joblib.load(MODELS_DIR / "isolation_forest_detector.joblib")
    return {"Rule-Based": rule_m, "PCA Reconstruction": pca_m, "Isolation Forest": if_m}


@st.cache_data
def load_faulty_dataset():
    """Cache loaded faulty test dataset."""
    df = pd.read_parquet(DATA_PROCESSED_DIR / "faulty_testing.parquet")
    return df


models = load_models()
df_faulty = load_faulty_dataset()

# Sidebar Controls
st.sidebar.header("🕹️ Simulation Controls")

fault_options = [1, 4, 6, 8]
fault_descriptions = {
    1: "IDV(1) - A/C Feed Ratio Step Disturbance",
    4: "IDV(4) - Reactor Cooling Water Temp Step",
    6: "IDV(6) - A Feed Loss Step Disturbance",
    8: "IDV(8) - A, B, C Feed Composition Random Variations"
}

selected_fault = st.sidebar.selectbox(
    "Select Fault Type:",
    options=fault_options,
    format_func=lambda f: fault_descriptions[f]
)

selected_run = st.sidebar.slider("Select Simulation Run ID:", min_value=1, max_value=100, value=1)
selected_model = st.sidebar.radio("Select Model for Detailed View:", list(models.keys()))

# Subset dataset for selected run
df_run = df_faulty[(df_faulty["faultnumber"] == selected_fault) & (df_faulty["simulationrun"] == selected_run)].sort_values(by="sample")
feature_cols = [c for c in df_run.columns if c not in ["faultnumber", "simulationrun", "sample"]]
X_run = df_run[feature_cols].values
y_true = (df_run["sample"].values >= FAULT_INTRO_SAMPLE_TESTING).astype(int)

# Run Selected Model Inference
model = models[selected_model]
scores = model.decision_function(X_run)
preds = model.predict(X_run)
metrics = evaluate_run_predictions(y_true, preds, fault_intro_sample=FAULT_INTRO_SAMPLE_TESTING)

# Section 1: Headline Metrics Cards
st.subheader(f"📊 Performance Breakdown: {selected_model} on IDV({selected_fault}) (Run {selected_run})")
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Precision", f"{metrics['precision']:.3f}")
col2.metric("Recall", f"{metrics['recall']:.3f}")
col3.metric("F1-Score", f"{metrics['f1']:.3f}")

lead_time_str = f"{metrics['lead_time_minutes']:+.1f} min" if metrics["lead_time_minutes"] is not None else "N/A"
col4.metric("Lead Time (Minutes)", lead_time_str)

detected_str = f"Sample {metrics['first_sustained_sample']}" if metrics["first_sustained_sample"] else "Not Flagged"
col5.metric("First Sustained Flag", detected_str)

# Section 2: Interactive Time-Series Score Plot
st.subheader("📈 Anomaly Score Time-Series vs. Healthy Threshold")

df_plot = pd.DataFrame({
    "Sample": df_run["sample"].values,
    "Anomaly Score": scores,
    "Threshold": model.threshold_
})

import plotly.graph_objects as go

fig = go.Figure()

# Continuous Anomaly Score Line
fig.add_trace(go.Scatter(
    x=df_plot["Sample"],
    y=df_plot["Anomaly Score"],
    mode="lines",
    name="Anomaly Score",
    line=dict(color="#2b5c8f", width=2)
))

# Learned Threshold Line
fig.add_trace(go.Scatter(
    x=df_plot["Sample"],
    y=df_plot["Threshold"],
    mode="lines",
    name=f"Learned Threshold ({model.threshold_:.2f})",
    line=dict(color="#d9534f", width=2, dash="dash")
))

# Vertical Line for Fault Introduction
fig.add_vline(
    x=FAULT_INTRO_SAMPLE_TESTING,
    line_width=2,
    line_dash="dot",
    line_color="#f0ad4e",
    annotation_text="Fault Introduced (Sample 160 / 8h)",
    annotation_position="top right"
)

# Shaded background region for active fault period
fig.add_vrect(
    x0=FAULT_INTRO_SAMPLE_TESTING,
    x1=df_plot["Sample"].max(),
    fillcolor="rgba(240, 173, 78, 0.1)",
    layer="below",
    line_width=0
)

fig.update_layout(
    xaxis_title="Time Sample (3-Minute Intervals)",
    yaxis_title="Continuous Anomaly Score",
    height=450,
    margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# Section 3: Root-Cause Attribution Comparison View
st.subheader("🔍 Root-Cause Attribution Comparison: TreeSHAP vs. PCA Reconstruction")

anom_mask = df_run["sample"].values >= FAULT_INTRO_SAMPLE_TESTING
X_anom = X_run[anom_mask]

if len(X_anom) > 0:
    col_shap, col_pca = st.columns(2)

    with col_shap:
        st.markdown("### 🌲 Isolation Forest - TreeSHAP Top Features")
        shap_vals, feat_names = compute_isolation_forest_shap(models["Isolation Forest"], X_anom, feature_names=feature_cols)
        df_top_shap = rank_top_contributing_features(shap_vals, feat_names, top_k=5)
        
        fig_shap = go.Figure(go.Bar(
            x=df_top_shap["mean_abs_shap"],
            y=df_top_shap["feature"],
            orientation="h",
            marker_color="#5cb85c"
        ))
        fig_shap.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Mean |SHAP Value|", height=300)
        st.plotly_chart(fig_shap, use_container_width=True)

    with col_pca:
        st.markdown("### 📐 PCA Reconstruction Error Top Features")
        pca_errs, feat_names = compute_pca_feature_attribution(models["PCA Reconstruction"], X_anom, feature_names=feature_cols)
        df_top_pca = rank_top_pca_contributions(pca_errs, feat_names, top_k=5)
        
        fig_pca = go.Figure(go.Bar(
            x=df_top_pca["mean_reconstruction_error"],
            y=df_top_pca["feature"],
            orientation="h",
            marker_color="#428bca"
        ))
        fig_pca.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Mean Squared Reconstruction Error", height=300)
        st.plotly_chart(fig_pca, use_container_width=True)

# Section 4: Three-Way Benchmark Summary Table
st.subheader("📋 Complete Benchmark Evaluation Summary Across All 100 Runs")
eval_csv_path = PROJECT_ROOT / "reports" / "evaluation_summary.csv"

if eval_csv_path.exists():
    df_summary = pd.read_csv(eval_csv_path)
    st.dataframe(df_summary, use_container_width=True)
