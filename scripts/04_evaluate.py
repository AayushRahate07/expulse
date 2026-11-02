"""
CLI Script: Four-Way Comparative Evaluation of Rule-Based, PCA, Isolation Forest, and PCA-Rotated IF.
Reports standard TEP benchmark metrics:
- Pre-Fault False Alarm Rate (FAR % on samples 1..159)
- Post-Fault Detection Rate (FDR % on samples 160..960)
- Detection Delay (samples & minutes >= 0)
- F1-Score
"""

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PROCESSED_DIR, MODELS_DIR, INCLUDED_FAULTS, FAULT_INTRO_SAMPLE_TESTING, RANDOM_SEED
from src.evaluation.metrics import evaluate_run_predictions, compute_binary_classification_metrics, compute_lead_time_metrics, compute_run_far_fdr
from src.utils import logger, timeit, format_dataframe_as_markdown


@timeit
def main():
    faulty_test_path = DATA_PROCESSED_DIR / "faulty_testing.parquet"
    train_healthy_path = DATA_PROCESSED_DIR / "fault_free_training.parquet"
    if not faulty_test_path.exists() or not train_healthy_path.exists():
        logger.error("Processed dataset files missing. Run scripts/02_convert_parquet.py first.")
        sys.exit(1)

    logger.info("Loading trained model detectors...")
    rule_model = joblib.load(MODELS_DIR / "rule_based_detector.joblib")
    pca_model = joblib.load(MODELS_DIR / "pca_detector.joblib")
    if_model = joblib.load(MODELS_DIR / "isolation_forest_detector.joblib")

    # Fit PCA-Rotated IF on nominal training data to test the axis-aligned failure mechanism
    logger.info("Fitting PCA-Rotated Isolation Forest (on 35 PCA components)...")
    df_train_h = pd.read_parquet(train_healthy_path)
    feat_cols_train = [c for c in df_train_h.columns if c not in ["faultnumber", "simulationrun", "sample"]]
    X_train_h = df_train_h[feat_cols_train].values
    X_train_scaled = pca_model.scaler_.transform(X_train_h)
    Z_train = pca_model.pca_.transform(X_train_scaled)

    if_rot = IsolationForest(n_estimators=100, random_state=RANDOM_SEED)
    if_rot.fit(Z_train)
    rot_train_scores = -if_rot.score_samples(Z_train)
    q25, q75 = np.percentile(rot_train_scores, [25, 75])
    rot_thresh = float(q75 + 1.5 * (q75 - q25))

    class PCARotatedIFWrapper:
        """Lightweight wrapper for PCA-Rotated IF conforming to predict API."""
        def __init__(self, pca_detector, if_rot_model, threshold):
            self.pca = pca_detector
            self.model = if_rot_model
            self.threshold_ = threshold

        def decision_function(self, X):
            X_scaled = self.pca.scaler_.transform(X)
            Z = self.pca.pca_.transform(X_scaled)
            return -self.model.score_samples(Z)

        def predict(self, X):
            scores = self.decision_function(X)
            return (scores >= self.threshold_).astype(int)

    pca_rot_if_model = PCARotatedIFWrapper(pca_model, if_rot, rot_thresh)

    models = {
        "Rule-Based": rule_model,
        "PCA Reconstruction": pca_model,
        "Isolation Forest": if_model,
        "PCA-Rotated IF": pca_rot_if_model
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
            run_fars = []
            run_fdrs = []
            run_delay_samples = []
            run_delay_mins = []
            runs_detected = 0

            for run in runs:
                df_run = df_f[df_f["simulationrun"] == run].sort_values(by="sample")
                X_run = df_run[feature_cols].values

                # Ground truth: sample >= FAULT_INTRO_SAMPLE_TESTING (160) is anomalous (1)
                y_true = (df_run["sample"].values >= FAULT_INTRO_SAMPLE_TESTING).astype(int)

                preds = model.predict(X_run)

                cls_m = compute_binary_classification_metrics(y_true, preds)
                far_fdr_m = compute_run_far_fdr(preds, fault_intro_sample=FAULT_INTRO_SAMPLE_TESTING)
                delay_m = compute_lead_time_metrics(preds, fault_intro_sample=FAULT_INTRO_SAMPLE_TESTING)

                run_precisions.append(cls_m["precision"])
                run_recalls.append(cls_m["recall"])
                run_f1s.append(cls_m["f1"])
                run_fars.append(far_fdr_m["far"])
                run_fdrs.append(far_fdr_m["fdr"])

                if delay_m["detected"]:
                    runs_detected += 1
                    run_delay_samples.append(delay_m["detection_delay_samples"])
                    run_delay_mins.append(delay_m["detection_delay_minutes"])

            avg_p = np.mean(run_precisions)
            avg_r = np.mean(run_recalls)
            avg_f1 = np.mean(run_f1s)
            avg_far = np.mean(run_fars)
            avg_fdr = np.mean(run_fdrs)
            avg_delay_samples = np.mean(run_delay_samples) if run_delay_samples else np.nan
            avg_delay_mins = np.mean(run_delay_mins) if run_delay_mins else np.nan
            det_rate_pct = (runs_detected / len(runs)) * 100.0

            results.append({
                "Fault Type": f"IDV({fault_id})",
                "Model": model_name,
                "Pre-Fault FAR (%)": round(float(avg_far * 100.0), 2),
                "Post-Fault FDR (%)": round(float(avg_fdr * 100.0), 2),
                "Detection Delay (min)": round(float(avg_delay_mins), 1) if not np.isnan(avg_delay_mins) else None,
                "Detection Delay (samples)": round(float(avg_delay_samples), 1) if not np.isnan(avg_delay_samples) else None,
                "Runs Detected (%)": round(float(det_rate_pct), 1),
                "F1-Score": round(float(avg_f1), 3),
                # Backwards compatible aliases
                "Lead Time (minutes)": round(float(avg_delay_mins), 1) if not np.isnan(avg_delay_mins) else None,
                "Lead Time (samples)": round(float(avg_delay_samples), 1) if not np.isnan(avg_delay_samples) else None,
                "Precision": round(float(avg_p), 3),
                "Recall": round(float(avg_r), 3),
            })

    df_res = pd.DataFrame(results)

    # Save summary report CSV with clean typed columns
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(reports_dir / "evaluation_summary.csv", index=False)
    logger.info(f"Saved evaluation results to '{reports_dir / 'evaluation_summary.csv'}'.")

    # Format human-friendly display for CLI output
    display_cols = [
        "Fault Type", "Model", "Pre-Fault FAR (%)", "Post-Fault FDR (%)",
        "Detection Delay (min)", "Runs Detected (%)", "F1-Score"
    ]
    df_display = df_res[display_cols].copy()
    df_display["Pre-Fault FAR (%)"] = df_display["Pre-Fault FAR (%)"].map("{:.2f}%".format)
    df_display["Post-Fault FDR (%)"] = df_display["Post-Fault FDR (%)"].map("{:.2f}%".format)
    df_display["Detection Delay (min)"] = df_display["Detection Delay (min)"].apply(
        lambda v: f"{v:.1f} min" if pd.notnull(v) else "Not detected"
    )
    df_display["Runs Detected (%)"] = df_display["Runs Detected (%)"].map("{:.1f}%".format)
    df_display["F1-Score"] = df_display["F1-Score"].map("{:.3f}".format)

    print("\n" + "="*85)
    print("TEP STANDARDIZED COMPARATIVE BENCHMARK (MATCHED METRICS: FAR, FDR, DELAY)")
    print("="*85)
    print(format_dataframe_as_markdown(df_display, index=False))
    print("="*85 + "\n")


if __name__ == "__main__":
    main()
