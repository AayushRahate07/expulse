"""
CLI Script: Generate Publication-Ready Figures and Plots.
Section 8 / Week 11 Deliverable.
"""

import os
import sys
import tempfile
from pathlib import Path

# Ensure Matplotlib config directory is writable across environments
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_config"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:
    sns = None
    _HAS_SEABORN = False

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PROCESSED_DIR, MODELS_DIR, FAULT_INTRO_SAMPLE_TESTING, REPORTS_DIR
from src.explainability import compute_isolation_forest_shap, compute_pca_feature_attribution
from src.explainability.shap_explainer import rank_top_contributing_features
from src.explainability.pca_attribution import rank_top_pca_contributions
from src.utils import logger, timeit


# Set publication figure styling
if _HAS_SEABORN:
    sns.set_theme(style="whitegrid", palette="muted")
else:
    plt.style.use("tableau-colorblind10")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "axes.grid": True,
    "grid.alpha": 0.3
})


def _plot_grouped_bars(df: pd.DataFrame, x_col: str, y_col: str, hue_col: str, ax: plt.Axes, title: str, ylabel: str):
    """Plots grouped bars using seaborn if available, or matplotlib with clean formatting."""
    if _HAS_SEABORN:
        sns.barplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax, palette="Set2")
    else:
        x_vals = df[x_col].unique()
        hue_vals = df[hue_col].unique()
        palette = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"]
        x_indices = np.arange(len(x_vals))
        width = 0.8 / len(hue_vals)

        for i, hue in enumerate(hue_vals):
            sub_df = df[df[hue_col] == hue]
            mapping = dict(zip(sub_df[x_col], sub_df[y_col]))
            y_vals = [mapping.get(x, 0.0) for x in x_vals]
            ax.bar(
                x_indices + (i - len(hue_vals) / 2 + 0.5) * width,
                y_vals,
                width=width,
                label=str(hue),
                color=palette[i % len(palette)],
                edgecolor="white"
            )

        ax.set_xticks(x_indices)
        ax.set_xticklabels(x_vals)
        ax.legend(title=hue_col)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(x_col)


def _plot_horizontal_bar(df: pd.DataFrame, x_col: str, y_col: str, ax: plt.Axes, title: str, xlabel: str, color: str):
    """Plots horizontal ranking bar chart."""
    if _HAS_SEABORN:
        sns.barplot(data=df, x=x_col, y=y_col, ax=ax, color=color)
    else:
        y_pos = np.arange(len(df))
        ax.barh(y_pos, df[x_col].values, color=color, edgecolor="none", alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df[y_col].values)
        ax.invert_yaxis()

    ax.set_title(title)
    ax.set_xlabel(xlabel)


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

        # Parse numeric lead time column cleanly whether stored as float or string "+16.7 min"
        if pd.api.types.is_numeric_dtype(df_sum["Lead Time (minutes)"]):
            df_sum["Lead_Mins_Num"] = df_sum["Lead Time (minutes)"].fillna(0.0)
        else:
            cleaned_str = (
                df_sum["Lead Time (minutes)"]
                .astype(str)
                .str.replace(" min", "", regex=False)
                .str.replace("+", "", regex=False)
            )
            df_sum["Lead_Mins_Num"] = pd.to_numeric(cleaned_str, errors="coerce").fillna(0.0)

        # Ensure F1-Score is float
        df_sum["F1-Score"] = pd.to_numeric(df_sum["F1-Score"], errors="coerce")

        # Figure 2: F1-Score Comparison
        fig, ax = plt.subplots(figsize=(8, 5))
        _plot_grouped_bars(df_sum, x_col="Fault Type", y_col="F1-Score", hue_col="Model", ax=ax,
                           title="F1-Score Comparison Across Fault Types", ylabel="F1-Score")
        ax.set_ylim(0, 1.1)
        plt.tight_layout()
        fig.savefig(figures_dir / "figure_2_benchmark_f1_comparison.png", dpi=300)
        plt.close()

        # Figure 3: Lead Time Comparison
        fig, ax = plt.subplots(figsize=(8, 5))
        _plot_grouped_bars(df_sum, x_col="Fault Type", y_col="Lead_Mins_Num", hue_col="Model", ax=ax,
                           title="Early Warning Lead Time (Minutes) Comparison", ylabel="Lead Time (Minutes)")
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

    _plot_horizontal_bar(top_shap, x_col="mean_abs_shap", y_col="feature", ax=axes[0],
                         title="Isolation Forest - Attribution Top 5 Features",
                         xlabel="Mean |Attribution Value|", color="#5cb85c")

    _plot_horizontal_bar(top_pca, x_col="mean_reconstruction_error", y_col="feature", ax=axes[1],
                         title="PCA Top 5 Reconstruction Error Features",
                         xlabel="Mean Squared Error", color="#428bca")

    plt.tight_layout()
    fig.savefig(figures_dir / "figure_4_idv4_attribution.png", dpi=300)
    plt.close()

    logger.info(f"All 4 publication figures generated and saved to '{figures_dir}'.")


if __name__ == "__main__":
    generate_publication_figures()
