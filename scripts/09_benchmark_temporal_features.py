"""
CLI Script: Benchmark Temporal Feature Augmentation (R^50 -> R^200).
Evaluates rolling mean, volatility, and trend slope on detection delay and F1-scores.
Focuses on IDV(8) slow composition variation disturbance.
Generates: reports/temporal_features_benchmark.csv.
"""

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PROCESSED_DIR, MODELS_DIR, INCLUDED_FAULTS, FAULT_INTRO_SAMPLE_TESTING, REPORTS_DIR
from src.data.features import generate_temporal_augmented_features
from src.models import PCADetector
from src.evaluation.metrics import evaluate_run_predictions, compute_binary_classification_metrics, compute_lead_time_metrics
from src.utils import logger, timeit, format_dataframe_as_markdown


@timeit
def main():
    logger.info("Initializing Temporal Feature Augmentation Benchmark (R^50 -> R^200)...")

    # 1. Load nominal training data and augment
    df_train = pd.read_parquet(DATA_PROCESSED_DIR / "fault_free_training.parquet")
    feature_cols = [c for c in df_train.columns if c not in ["faultnumber", "simulationrun", "sample"]]

    logger.info("Augmenting nominal training data with rolling mean, std, and slope (w=15)...")
    X_train_aug, aug_feature_names = generate_temporal_augmented_features(df_train, feature_cols=feature_cols, window_size=15)
    logger.info(f"Augmented training shape: {X_train_aug.shape} (expanded from {len(feature_cols)} to {len(aug_feature_names)})")

    # Fit Augmented PCA Detector
    pca_aug = PCADetector(n_components=0.95)
    pca_aug.fit(X_train_aug)
    logger.info(
        f"Augmented PCA fitted. Retained {pca_aug.pca_.n_components_} components explaining "
        f"{sum(pca_aug.pca_.explained_variance_ratio_)*100:.2f}% variance. Threshold: {pca_aug.threshold_:.4f}"
    )

    # Load Static PCA Detector for baseline comparison
    pca_static = joblib.load(MODELS_DIR / "pca_detector.joblib")

    # 2. Evaluate across faulty runs for all faults, highlighting IDV(8)
    df_faulty = pd.read_parquet(DATA_PROCESSED_DIR / "faulty_testing.parquet")
    fault_ids = [f for f in INCLUDED_FAULTS if f != 0]

    comparison_results = []

    for fault_id in fault_ids:
        df_f = df_faulty[df_faulty["faultnumber"] == fault_id]
        runs = df_f["simulationrun"].unique()

        static_f1s, aug_f1s = [], []
        static_leads, aug_leads = [], []

        # Evaluate across first 25 runs per fault for benchmark speed
        eval_runs = runs[:25]

        for run in eval_runs:
            df_run = df_f[df_f["simulationrun"] == run].sort_values(by="sample")
            y_true = (df_run["sample"].values >= FAULT_INTRO_SAMPLE_TESTING).astype(int)

            # Static evaluation
            X_static = df_run[feature_cols].values
            preds_static = pca_static.predict(X_static)
            m_static = compute_binary_classification_metrics(y_true, preds_static)
            l_static = compute_lead_time_metrics(preds_static, fault_intro_sample=FAULT_INTRO_SAMPLE_TESTING)
            static_f1s.append(m_static["f1"])
            if l_static["lead_time_minutes"] is not None:
                static_leads.append(l_static["lead_time_minutes"])

            # Augmented evaluation
            X_aug, _ = generate_temporal_augmented_features(df_run, feature_cols=feature_cols, window_size=15)
            preds_aug = pca_aug.predict(X_aug)
            m_aug = compute_binary_classification_metrics(y_true, preds_aug)
            l_aug = compute_lead_time_metrics(preds_aug, fault_intro_sample=FAULT_INTRO_SAMPLE_TESTING)
            aug_f1s.append(m_aug["f1"])
            if l_aug["lead_time_minutes"] is not None:
                aug_leads.append(l_aug["lead_time_minutes"])

        avg_f1_static = float(np.mean(static_f1s))
        avg_f1_aug = float(np.mean(aug_f1s))
        avg_lead_static = float(np.mean(static_leads)) if static_leads else np.nan
        avg_lead_aug = float(np.mean(aug_leads)) if aug_leads else np.nan
        lead_delta = avg_lead_aug - avg_lead_static

        comparison_results.append({
            "Fault Mode": f"IDV({fault_id})",
            "Static F1": round(avg_f1_static, 3),
            "Augmented R^200 F1": round(avg_f1_aug, 3),
            "Static Lead (min)": round(avg_lead_static, 1),
            "Augmented Lead (min)": round(avg_lead_aug, 1),
            "Lead Time Delta (min)": round(lead_delta, 1),
            "Hypothesis Confirmed": "YES (Earlier Alert)" if lead_delta < 0 else "Comparable",
        })

    df_comp = pd.DataFrame(comparison_results)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = REPORTS_DIR / "temporal_features_benchmark.csv"
    df_comp.to_csv(out_csv, index=False)
    logger.info(f"Saved temporal feature benchmark to '{out_csv}'.")

    print("\n" + "=" * 85)
    print("TEMPORAL FEATURE AUGMENTATION (R^50 VS R^200) COMPARATIVE BENCHMARK")
    print("=" * 85)
    print(format_dataframe_as_markdown(df_comp, index=False))
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
