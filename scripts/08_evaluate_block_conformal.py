"""
CLI Script: Non-Overlapping Block Conformal Prediction Coverage Analysis.
Evaluates empirical Block False Alarm Rate (Block FAR) and coverage guarantees
on held-out healthy test data (96,000 observations) under serial autocorrelation.
Generates: reports/conformal_coverage_analysis.csv.
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

from src.config import DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR
from src.evaluation.conformal import BlockConformalCalibrator
from src.utils import logger, timeit, format_dataframe_as_markdown


@timeit
def main():
    logger.info("Initializing Block Conformal Calibration & Verification...")

    # Load PCA detector and healthy datasets
    pca_model = joblib.load(MODELS_DIR / "pca_detector.joblib")
    df_train_healthy = pd.read_parquet(DATA_PROCESSED_DIR / "fault_free_training.parquet")
    df_test_healthy = pd.read_parquet(DATA_PROCESSED_DIR / "fault_free_testing.parquet")

    feature_cols = [c for c in df_train_healthy.columns if c not in ["faultnumber", "simulationrun", "sample"]]

    X_train_h = df_train_healthy[feature_cols].values
    X_test_h = df_test_healthy[feature_cols].values

    # Continuous SPE scores
    train_spe = pca_model.decision_function(X_train_h)
    test_spe = pca_model.decision_function(X_test_h)

    # Evaluate for significance levels alpha in {0.01, 0.05, 0.10}
    alphas = [0.01, 0.05, 0.10]
    block_lengths = [10, 20, 30]  # 30 mins, 1 hour, 1.5 hours
    results = []

    for bl in block_lengths:
        for alpha in alphas:
            calibrator = BlockConformalCalibrator(block_length=bl, alpha=alpha)
            calibrator.fit(train_spe)

            cov_metrics = calibrator.evaluate_block_coverage(test_spe)

            # Compare against standard i.i.d. point-wise split conformal (naive point-wise threshold)
            naive_point_threshold = np.quantile(train_spe, 1.0 - alpha)
            naive_point_far = float(np.mean(test_spe >= naive_point_threshold))

            results.append({
                "Block Length (L samples)": f"{bl} ({bl * 3} min)",
                "Significance Level (alpha)": alpha,
                "Target Coverage": f"{(1.0 - alpha)*100:.1f}%",
                "Block Conformal FAR": f"{cov_metrics['empirical_block_far']*100:.2f}%",
                "Empirical Coverage": f"{cov_metrics['empirical_coverage']*100:.2f}%",
                "Coverage Guarantee Held": "YES" if cov_metrics["guarantee_held"] else "NO",
                "Naive Point-wise FAR": f"{naive_point_far*100:.2f}%",
            })

    df_conformal = pd.DataFrame(results)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = REPORTS_DIR / "conformal_coverage_analysis.csv"
    df_conformal.to_csv(out_csv, index=False)
    logger.info(f"Saved Block Conformal analysis to '{out_csv}'.")

    print("\n" + "=" * 85)
    print("BLOCK CONFORMAL COVERAGE & FALSE ALARM CONTROL ANALYSIS")
    print("=" * 85)
    print(format_dataframe_as_markdown(df_conformal, index=False))
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
