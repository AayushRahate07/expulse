"""
CLI Script: Generate Publication-Ready Figures and Plots.
Section 8 / Week 11 Deliverable.
"""

import sys
import joblib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PROCESSED_DIR, MODELS_DIR, FAULT_INTRO_SAMPLE_TESTING, REPORTS_DIR
from src.explainability import compute_isolation_forest_shap, compute_pca_feature_attribution
from src.explainability.shap_explainer import rank_top_contributing_features
from src.explainability.pca_attribution import rank_top_pca_contributions
from src.utils import logger, timeit


# Set seaborn publication style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14
})


@timeit
def generate_publication_figures():
    figures_dir = REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading models and datasets for plot generation...")
    rule_m = joblib.load(MODELS_DIR / "rule_based_detector.joblib")
    pca_m = joblib.load(MODELS_DIR / "pca_detector.joblib")
    if_m = joblib.load(MODELS_DIR / "isolation_forest_detector.joblib")

    df_faulty = pd.read_parquet(DATA_PROCESSED_DIR / "faulty_testing.parquet")
    feature_cols = [c for c in df_faulty.columns if c not in ["faultnumber", "simulationrun", "sample"]]

    # --- Figure 1: Score Time-Series for IDV(4) ---
    logger.info("Generating Figure 1: Anomaly Score Time-Series...")
    df_idv4 = df_faulty[(df_faulty["faultnumber"] == 4) & (df_faulty["simulationrun"] == 1)].sort_values(by="sample")
    X_idv4 = df_idv4[feature_cols].values

    rule_scores = rule_m.decision_function(X_idv4)
    pca_scores = pca_m.decision_function(X_idv4)
    if_scores = if_m.decision_function(X_idv4)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(df_idv4["sample"], rule_scores, color="#2b5c8f", label="Rule-Based Score")
    axes[0].axhline(rule_m.threshold_, color="crimson", linestyle="--", label="Threshold")
    axes[0].set_ylabel("Rule Max Dev")
    axes[0].set_title("Rule-Based Anomaly Score")

    axes[1].plot(df_idv4["sample"], pca_scores, color="#428bca", label="PCA SPE Score")
    axes[1].axhline(pca_m.threshold_, color="crimson", linestyle="--", label="Threshold")
    axes[1].set_ylabel("PCA SPE")
    axes[1].set_title("PCA Reconstruction Error Score")

    axes[2].plot(df_idv4["sample"], if_scores, color="#5cb85c", label="Isolation Forest Score")
    axes[2].axhline(if_m.threshold_, color="crimson", linestyle="--", label="Threshold")
    axes[2].set_ylabel("Inverted Path Score")
    axes[2].set_title("Isolation Forest Anomaly Score")

    for ax in axes:
        ax.axvline(FAULT_INTRO_SAMPLE_TESTING, color="orange", linestyle=":", linewidth=2, label="Fault Introduced (Sample 160)")
        ax.axvspan(FAULT_INTRO_SAMPLE_TESTING, df_idv4["sample"].max(), color="orange", alpha=0.1)
        ax.legend(loc="upper left")

    axes[2].set_xlabel("Time Sample (3-Minute Intervals)")
    plt.tight_layout()
    fig.savefig(figures_dir / "figure_1_score_timeseries.png", dpi=300)
    plt.close()

    # --- Figure 2 & 3: Benchmark F1 and Lead Time Bar Charts ---
    logger.info("Generating Figure 2 & 3: Benchmark Metrics Comparison...")
    summary_path = REPORTS_DIR / "evaluation_summary.csv"
    if summary_path.exists():
        df_sum = pd.read_csv(summary_path)

        # Clean Lead Time column for numeric plotting
        df_sum["Lead_Mins_Num"] = df_sum["Lead Time (minutes)"].str.replace(" min", "").str.replace("+", "").astype(float)

        # Figure 2: F1-Score Comparison
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=df_sum, x="Fault Type", y="F1-Score", hue="Model", ax=ax, palette="Set2")
        ax.set_ylim(0, 1.1)
        ax.set_title("F1-Score Comparison Across Fault Types")
        plt.tight_layout()
        fig.savefig(figures_dir / "figure_2_benchmark_f1_comparison.png", dpi=300)
        plt.close()

        # Figure 3: Lead Time Comparison
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=df_sum, x="Fault Type", y="Lead_Mins_Num", hue="Model", ax=ax, palette="Set2")
        ax.set_ylabel("Lead Time (Minutes)")
        ax.set_title("Early Warning Lead Time (Minutes) Comparison")
        plt.tight_layout()
        fig.savefig(figures_dir / "figure_3_lead_time_comparison.png", dpi=300)
        plt.close()

    # --- Figure 4: Root-Cause Attribution Comparison ---
    logger.info("Generating Figure 4: Root-Cause Attribution Comparison...")
    anom_mask = df_idv4["sample"].values >= FAULT_INTRO_SAMPLE_TESTING
    X_anom = X_idv4[anom_mask]

    shap_vals, feat_names = compute_isolation_forest_shap(if_m, X_anom, feature_names=feature_cols)
    top_shap = rank_top_contributing_features(shap_vals, feat_names, top_k=5)

    pca_errs, feat_names = compute_pca_feature_attribution(pca_m, X_anom, feature_names=feature_cols)
    top_pca = rank_top_pca_contributions(pca_errs, feat_names, top_k=5)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    sns.barplot(data=top_shap, x="mean_abs_shap", y="feature", ax=axes[0], color="#5cb85c")
    axes[0].set_title("Isolation Forest - TreeSHAP Top 5 Features")
    axes[0].set_xlabel("Mean |SHAP Value|")

    sns.barplot(data=top_pca, x="mean_reconstruction_error", y="feature", ax=axes[1], color="#428bca")
    axes[1].set_title("PCA Top 5 Reconstruction Error Features")
    axes[1].set_xlabel("Mean Squared Error")

    plt.tight_layout()
    fig.savefig(figures_dir / "figure_4_idv4_attribution.png", dpi=300)
    plt.close()

    logger.info(f"All 4 publication figures generated and saved to '{figures_dir}'.")


if __name__ == "__main__":
    generate_publication_figures()
