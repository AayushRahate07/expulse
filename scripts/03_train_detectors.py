"""
CLI Script: Train RuleBasedDetector, PCADetector, and IsolationForestDetector on Healthy Data.
"""

import sys
import joblib
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PROCESSED_DIR, MODELS_DIR
from src.models import RuleBasedDetector, PCADetector, IsolationForestDetector
from src.utils import logger, timeit, set_seed


@timeit
def main():
    set_seed(42)

    train_path = DATA_PROCESSED_DIR / "fault_free_training.parquet"
    test_path = DATA_PROCESSED_DIR / "fault_free_testing.parquet"

    if not train_path.exists() or not test_path.exists():
        logger.error("Processed fault-free dataset files not found. Run scripts/02_convert_parquet.py first.")
        sys.exit(1)

    logger.info(f"Loading fault-free training dataset from '{train_path}'...")
    df_train = pd.read_parquet(train_path)

    # Exclude metadata columns
    feature_cols = [c for c in df_train.columns if c not in ["faultnumber", "simulationrun", "sample"]]
    X_train = df_train[feature_cols].values
    logger.info(f"Feature matrix shape for training: {X_train.shape} across {len(feature_cols)} sensors.")

    # 1. Train Rule-Based Threshold Detector
    logger.info("--- Fitting RuleBasedDetector ---")
    rule_detector = RuleBasedDetector(k_std=3.0)
    rule_detector.fit(X_train)
    logger.info(f"RuleBasedDetector fitted. Threshold: {rule_detector.threshold_:.4f}")
    joblib.dump(rule_detector, MODELS_DIR / "rule_based_detector.joblib")

    # 2. Train PCA Reconstruction Error Detector
    logger.info("--- Fitting PCADetector (95% variance) ---")
    pca_detector = PCADetector(n_components=0.95)
    pca_detector.fit(X_train)
    logger.info(
        f"PCADetector fitted. Retained {pca_detector.pca_.n_components_} components explaining "
        f"{sum(pca_detector.pca_.explained_variance_ratio_)*100:.2f}% variance. Threshold: {pca_detector.threshold_:.4f}"
    )
    joblib.dump(pca_detector, MODELS_DIR / "pca_detector.joblib")

    # 3. Train Isolation Forest Detector
    logger.info("--- Fitting IsolationForestDetector ---")
    if_detector = IsolationForestDetector(n_estimators=100, random_state=42)
    if_detector.fit(X_train)
    logger.info(f"IsolationForestDetector fitted. Threshold: {if_detector.threshold_:.4f}")
    joblib.dump(if_detector, MODELS_DIR / "isolation_forest_detector.joblib")

    logger.info(f"All 3 detectors trained and serialized to '{MODELS_DIR}'.")


if __name__ == "__main__":
    main()
