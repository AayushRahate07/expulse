"""
CLI Script: Three-Way Comparative Evaluation of Rule-Based, PCA, and Isolation Forest Detectors.
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

from src.config import DATA_PROCESSED_DIR, MODELS_DIR, INCLUDED_FAULTS, FAULT_INTRO_SAMPLE_TESTING
from src.evaluation.metrics import evaluate_run_predictions, compute_binary_classification_metrics, compute_lead_time_metrics
from src.utils import logger, timeit, format_dataframe_as_markdown


@timeit
def main():
    faulty_test_path = DATA_PROCESSED_DIR / "faulty_testing.parquet"
    if not faulty_test_path.exists():
        logger.error("Processed faulty testing file missing. Run scripts/02_convert_parquet.py first.")
        sys.exit(1)

    logger.info("Loading trained model detectors...")
    rule_model = joblib.load(MODELS_DIR / "rule_based_detector.joblib")
    pca_model = joblib.load(MODELS_DIR / "pca_detector.joblib")
    if_model = joblib.load(MODELS_DIR / "isolation_forest_detector.joblib")

    models = {
        "Rule-Based": rule_model,
        "PCA Reconstruction": pca_model,
        "Isolation Forest": if_model
    }

    logger.info(f"Loading faulty testing dataset from '{faulty_test_path}'...")
    df_faulty = pd.read_parquet(faulty_test_path)

    feature_cols = [c for c in df_faulty.columns if c not in ["faultnumber", "simulationrun", "sample"]]
    fault_ids = [f for f in INCLUDED_FAULTS if f != 0]

    results = []

    for fault_id in fault_ids:
        df_f = df_faulty[df_faulty["faultnumber"] == fault_id]
        runs = df_f["simulationrun"].unique()
        logger.info(f"Evaluating Fault IDV({fault_id}) across {len(runs)} simulation runs...")

        for model_name, model in models.items():
            run_precisions = []
            run_recalls = []
            run_f1s = []
            run_lead_samples = []
            run_lead_mins = []

            for run in runs:
                df_run = df_f[df_f["simulationrun"] == run].sort_values(by="sample")
                X_run = df_run[feature_cols].values

                # Ground truth: sample >= FAULT_INTRO_SAMPLE_TESTING (160) is anomalous (1)
                y_true = (df_run["sample"].values >= FAULT_INTRO_SAMPLE_TESTING).astype(int)

                preds = model.predict(X_run)

                cls_m = compute_binary_classification_metrics(y_true, preds)
                lead_m = compute_lead_time_metrics(preds, fault_intro_sample=FAULT_INTRO_SAMPLE_TESTING)

                run_precisions.append(cls_m["precision"])
                run_recalls.append(cls_m["recall"])
                run_f1s.append(cls_m["f1"])

                if lead_m["lead_time_samples"] is not None:
                    run_lead_samples.append(lead_m["lead_time_samples"])
                    run_lead_mins.append(lead_m["lead_time_minutes"])

            avg_p = np.mean(run_precisions)
            avg_r = np.mean(run_recalls)
            avg_f1 = np.mean(run_f1s)
            avg_lead_samples = np.mean(run_lead_samples) if run_lead_samples else np.nan
            avg_lead_mins = np.mean(run_lead_mins) if run_lead_mins else np.nan

            results.append({
                "Fault Type": f"IDV({fault_id})",
                "Model": model_name,
                "Precision": round(float(avg_p), 3),
                "Recall": round(float(avg_r), 3),
                "F1-Score": round(float(avg_f1), 3),
                "Lead Time (samples)": round(float(avg_lead_samples), 1) if not np.isnan(avg_lead_samples) else None,
                "Lead Time (minutes)": round(float(avg_lead_mins), 1) if not np.isnan(avg_lead_mins) else None,
            })

    df_res = pd.DataFrame(results)

    # Save summary report CSV with clean typed columns
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(reports_dir / "evaluation_summary.csv", index=False)
    logger.info(f"Saved evaluation results to '{reports_dir / 'evaluation_summary.csv'}'.")

    # Format human-friendly display for CLI output
    df_display = df_res.copy()
    df_display["Precision"] = df_display["Precision"].map("{:.3f}".format)
    df_display["Recall"] = df_display["Recall"].map("{:.3f}".format)
    df_display["F1-Score"] = df_display["F1-Score"].map("{:.3f}".format)
    df_display["Lead Time (samples)"] = df_display["Lead Time (samples)"].apply(
        lambda v: f"{v:+.1f}" if pd.notnull(v) else "N/A"
    )
    df_display["Lead Time (minutes)"] = df_display["Lead Time (minutes)"].apply(
        lambda v: f"{v:+.1f} min" if pd.notnull(v) else "N/A"
    )

    print("\n" + "="*80)
    print("THREE-WAY COMPARATIVE EVALUATION SUMMARY TABLE")
    print("="*80)
    print(format_dataframe_as_markdown(df_display, index=False))
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
