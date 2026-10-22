"""
CLI Script: Explainability Comparison (TreeSHAP vs PCA Reconstruction Error Attribution).
"""

import sys
import joblib
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PROCESSED_DIR, MODELS_DIR, FAULT_INTRO_SAMPLE_TESTING, REPORTS_DIR
from src.explainability import compute_isolation_forest_shap, compute_pca_feature_attribution
from src.explainability.shap_explainer import rank_top_contributing_features
from src.explainability.pca_attribution import rank_top_pca_contributions
from src.utils import logger, timeit


@timeit
def main():
    faulty_test_path = DATA_PROCESSED_DIR / "faulty_testing.parquet"
    if not faulty_test_path.exists():
        logger.error("Processed faulty testing dataset missing.")
        sys.exit(1)

    logger.info("Loading trained PCA and Isolation Forest models...")
    pca_model = joblib.load(MODELS_DIR / "pca_detector.joblib")
    if_model = joblib.load(MODELS_DIR / "isolation_forest_detector.joblib")

    logger.info("Loading sample faulty run (IDV(4), run=1)...")
    df_faulty = pd.read_parquet(faulty_test_path)
    df_sample = df_faulty[(df_faulty["faultnumber"] == 4) & (df_faulty["simulationrun"] == 1)].sort_values(by="sample")

    feature_cols = [c for c in df_sample.columns if c not in ["faultnumber", "simulationrun", "sample"]]
    X_sample = df_sample[feature_cols].values

    # Focus attribution on anomalous samples after fault introduction
    anom_mask = df_sample["sample"].values >= FAULT_INTRO_SAMPLE_TESTING
    X_anom = X_sample[anom_mask]

    logger.info(f"Computing TreeSHAP for Isolation Forest on {len(X_anom)} anomalous samples...")
    shap_vals, feat_names = compute_isolation_forest_shap(if_model, X_anom, feature_names=feature_cols)
    top_shap = rank_top_contributing_features(shap_vals, feat_names, top_k=5)

    logger.info(f"Computing PCA Reconstruction Error attribution on {len(X_anom)} anomalous samples...")
    pca_errs, feat_names = compute_pca_feature_attribution(pca_model, X_anom, feature_names=feature_cols)
    top_pca = rank_top_pca_contributions(pca_errs, feat_names, top_k=5)

    print("\n" + "="*80)
    print("ROOT-CAUSE ATTRIBUTION COMPARISON FOR FAULT IDV(4)")
    print("="*80)
    print("--- Isolation Forest Top 5 TreeSHAP Features ---")
    print(top_shap.to_markdown(index=False))
    print("\n--- PCA Top 5 Reconstruction Error Features ---")
    print(top_pca.to_markdown(index=False))
    print("="*80 + "\n")

    top_shap.to_csv(REPORTS_DIR / "idv4_treeshap_attribution.csv", index=False)
    top_pca.to_csv(REPORTS_DIR / "idv4_pca_attribution.csv", index=False)
    logger.info(f"Saved attribution reports to '{REPORTS_DIR}'.")


if __name__ == "__main__":
    main()
