"""
CLI Script: Benchmark Page's CUSUM Sequential Detection vs Heuristic Sliding Window.
Compares Lorden-optimal CUSUM against 3-consecutive-flag heuristic across 400 test runs.
Generates: reports/cusum_vs_consecutive_leadtime.csv.
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
from src.evaluation.cusum import CUSUMDetector, compute_cusum_lead_time
from src.evaluation.metrics import compute_lead_time_metrics
from src.utils import logger, timeit, format_dataframe_as_markdown


@timeit
def main():
    logger.info("Initializing CUSUM Sequential Detection Benchmark...")

    pca_model = joblib.load(MODELS_DIR / "pca_detector.joblib")
    df_train = pd.read_parquet(DATA_PROCESSED_DIR / "fault_free_training.parquet")
    feature_cols = [c for c in df_train.columns if c not in ["faultnumber", "simulationrun", "sample"]]

    # Calibrate CUSUM on nominal training scores
    X_train = df_train[feature_cols].values
    train_spe_scores = pca_model.decision_function(X_train)

    cusum = CUSUMDetector(k_factor=0.5)
    cusum.fit(train_spe_scores, target_arl0=2000.0)
    logger.info(
        f"CUSUM calibrated: mu_0={cusum.mu_0_:.4f}, sigma_0={cusum.sigma_0_:.4f}, "
        f"slack_k={cusum.k_:.4f}, threshold_h={cusum.h_:.4f}"
    )

    df_faulty = pd.read_parquet(DATA_PROCESSED_DIR / "faulty_testing.parquet")
    fault_ids = [f for f in INCLUDED_FAULTS if f != 0]

    benchmark_rows = []

    for fault_id in fault_ids:
        df_f = df_faulty[df_faulty["faultnumber"] == fault_id]
        runs = df_f["simulationrun"].unique()

        heuristic_lead_mins = []
        cusum_lead_mins = []

        for run in runs:
            df_run = df_f[df_f["simulationrun"] == run].sort_values(by="sample")
            X_run = df_run[feature_cols].values

            # Static model predictions & continuous anomaly scores
            spe_scores = pca_model.decision_function(X_run)
            binary_preds = pca_model.predict(X_run)

            # Method 1: Heuristic 3-consecutive-flag
            h_lead = compute_lead_time_metrics(binary_preds, fault_intro_sample=FAULT_INTRO_SAMPLE_TESTING)
            if h_lead["lead_time_minutes"] is not None:
                heuristic_lead_mins.append(h_lead["lead_time_minutes"])

            # Method 2: CUSUM sequential change-point
            c_lead = compute_cusum_lead_time(spe_scores, cusum, fault_intro_sample=FAULT_INTRO_SAMPLE_TESTING)
            if c_lead["lead_time_minutes"] is not None:
                cusum_lead_mins.append(c_lead["lead_time_minutes"])

        avg_heur = float(np.mean(heuristic_lead_mins)) if heuristic_lead_mins else np.nan
        avg_cusum = float(np.mean(cusum_lead_mins)) if cusum_lead_mins else np.nan
        delta = avg_cusum - avg_heur

        benchmark_rows.append({
            "Fault Mode": f"IDV({fault_id})",
            "Consecutive-Flag Lead (min)": round(avg_heur, 1),
            "CUSUM Lead (min)": round(avg_cusum, 1),
            "Delta Delay (min)": round(delta, 1),
            "Detection Advantage": "Earlier with CUSUM" if delta < 0 else ("Identical" if abs(delta) < 0.5 else "Heuristic"),
        })

    df_out = pd.DataFrame(benchmark_rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = REPORTS_DIR / "cusum_vs_consecutive_leadtime.csv"
    df_out.to_csv(out_csv, index=False)
    logger.info(f"Saved CUSUM comparison to '{out_csv}'.")

    print("\n" + "=" * 80)
    print("CUSUM VS. CONSECUTIVE-FLAG DETECTION DELAY BENCHMARK")
    print("=" * 80)
    print(format_dataframe_as_markdown(df_out, index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
