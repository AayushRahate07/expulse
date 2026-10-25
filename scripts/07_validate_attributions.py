"""
CLI Script: Quantitative Ground-Truth Attribution Validation Benchmark.
Evaluates Top-K Hit Rate (H@1, H@3, H@5), MRR, and paired Wilcoxon test
comparing TreeSHAP/Perturbation vs PCA Reconstruction Error against Downs & Vogel (1993).
Generates: reports/attribution_accuracy_benchmark.csv.
"""

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PROCESSED_DIR, MODELS_DIR, INCLUDED_FAULTS, FAULT_INTRO_SAMPLE_TESTING, REPORTS_DIR
from src.explainability.shap_explainer import compute_isolation_forest_shap, rank_top_contributing_features
from src.explainability.pca_attribution import compute_pca_feature_attribution, rank_top_pca_contributions
from src.explainability.validation import (
    evaluate_attribution_ranking,
    compare_attribution_methods_wilcoxon,
    get_ground_truth_variables
)
from src.utils import logger, timeit, format_dataframe_as_markdown


@timeit
def main():
    logger.info("Initializing Ground-Truth Attribution Validation Benchmark...")

    pca_model = joblib.load(MODELS_DIR / "pca_detector.joblib")
    if_model = joblib.load(MODELS_DIR / "isolation_forest_detector.joblib")

    df_faulty = pd.read_parquet(DATA_PROCESSED_DIR / "faulty_testing.parquet")
    feature_cols = [c for c in df_faulty.columns if c not in ["faultnumber", "simulationrun", "sample"]]
    fault_ids = [f for f in INCLUDED_FAULTS if f != 0]

    validation_records = []

    for fault_id in fault_ids:
        df_f = df_faulty[df_faulty["faultnumber"] == fault_id]
        runs = df_f["simulationrun"].unique()
        gt_vars = get_ground_truth_variables(fault_id)
        logger.info(f"Fault IDV({fault_id}) - Ground Truth physical sensors ({len(gt_vars)}): {sorted(gt_vars)}")

        # Collect run-level H@3 for Wilcoxon test
        pca_h3_runs = []
        if_h3_runs = []

        pca_mrr_runs = []
        if_mrr_runs = []
        pca_h1_runs, pca_h5_runs = [], []
        if_h1_runs, if_h5_runs = [], []

        # Evaluate across first 20 runs per fault to keep runtime responsive
        eval_runs = runs[:20]

        for run in eval_runs:
            df_run = df_f[df_f["simulationrun"] == run].sort_values(by="sample")
            anom_mask = df_run["sample"].values >= FAULT_INTRO_SAMPLE_TESTING
            X_anom = df_run.loc[anom_mask, feature_cols].values

            if len(X_anom) == 0:
                continue

            # 1. PCA Reconstruction Error Attribution
            pca_errs, _ = compute_pca_feature_attribution(pca_model, X_anom, feature_names=feature_cols)
            top_pca_df = rank_top_pca_contributions(pca_errs, feature_cols, top_k=10)
            pca_ranked = list(top_pca_df["feature"])
            pca_metrics = evaluate_attribution_ranking(pca_ranked, fault_id)

            pca_h1_runs.append(pca_metrics["H@1"])
            pca_h3_runs.append(pca_metrics["H@3"])
            pca_h5_runs.append(pca_metrics["H@5"])
            pca_mrr_runs.append(pca_metrics["MRR"])

            # 2. Isolation Forest Attribution
            if_vals, _ = compute_isolation_forest_shap(if_model, X_anom, feature_names=feature_cols)
            top_if_df = rank_top_contributing_features(if_vals, feature_cols, top_k=10)
            if_ranked = list(top_if_df["feature"])
            if_metrics = evaluate_attribution_ranking(if_ranked, fault_id)

            if_h1_runs.append(if_metrics["H@1"])
            if_h3_runs.append(if_metrics["H@3"])
            if_h5_runs.append(if_metrics["H@5"])
            if_mrr_runs.append(if_metrics["MRR"])

        # Run Wilcoxon signed-rank test
        wilcoxon_res = compare_attribution_methods_wilcoxon(pca_h3_runs, if_h3_runs)

        validation_records.append({
            "Fault Mode": f"IDV({fault_id})",
            "Model": "PCA Reconstruction",
            "H@1": round(float(np.mean(pca_h1_runs)), 3),
            "H@3": round(float(np.mean(pca_h3_runs)), 3),
            "H@5": round(float(np.mean(pca_h5_runs)), 3),
            "MRR": round(float(np.mean(pca_mrr_runs)), 3),
            "Wilcoxon p-val vs IF": f"{wilcoxon_res['p_value']:.4f}",
            "Statistical Winner": "PCA" if wilcoxon_res["mean_diff"] > 0 and wilcoxon_res["p_value"] < 0.05 else ("IF" if wilcoxon_res["mean_diff"] < 0 and wilcoxon_res["p_value"] < 0.05 else "Tie / Non-sig"),
        })

        validation_records.append({
            "Fault Mode": f"IDV({fault_id})",
            "Model": "Isolation Forest",
            "H@1": round(float(np.mean(if_h1_runs)), 3),
            "H@3": round(float(np.mean(if_h3_runs)), 3),
            "H@5": round(float(np.mean(if_h5_runs)), 3),
            "MRR": round(float(np.mean(if_mrr_runs)), 3),
            "Wilcoxon p-val vs IF": "-",
            "Statistical Winner": "-",
        })

    df_report = pd.DataFrame(validation_records)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = REPORTS_DIR / "attribution_accuracy_benchmark.csv"
    df_report.to_csv(out_csv, index=False)
    logger.info(f"Saved attribution validation report to '{out_csv}'.")

    print("\n" + "=" * 80)
    print("GROUND-TRUTH ATTRIBUTION ACCURACY BENCHMARK (TOP-K HIT RATE & MRR)")
    print("=" * 80)
    print(format_dataframe_as_markdown(df_report, index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
