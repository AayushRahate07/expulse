"""
Unit Tests for Explainability and Root-Cause Attribution Modules.
"""

import numpy as np
import pandas as pd

from src.models import PCADetector, IsolationForestDetector
from src.explainability.pca_attribution import compute_pca_feature_attribution, rank_top_pca_contributions
from src.explainability.shap_explainer import compute_isolation_forest_shap, rank_top_contributing_features


def _generate_explain_data():
    """Generates synthetic healthy correlated training data and an anomalous test slice."""
    np.random.seed(42)
    # 200 samples with strong covariance structure
    base = np.random.normal(0, 1, (200, 1))
    X_healthy = np.hstack([base * (i + 1) + np.random.normal(0, 0.05, (200, 1)) for i in range(6)])
    
    # 20 anomalous samples where sensor_2 breaks the correlation subspace
    X_anom = np.hstack([base[:20] * (i + 1) for i in range(6)])
    X_anom[:, 2] += 15.0  # Large disturbance on sensor 2
    
    feature_names = [f"sensor_{i}" for i in range(6)]
    return X_healthy, X_anom, feature_names


def test_pca_feature_attribution():
    """Verifies PCA feature attribution computation and top-k ranking."""
    X_healthy, X_anom, feature_names = _generate_explain_data()

    detector = PCADetector(n_components=2)
    detector.fit(X_healthy)

    feat_errs, cols = compute_pca_feature_attribution(detector, X_anom, feature_names=feature_names)

    assert isinstance(feat_errs, np.ndarray)
    assert feat_errs.shape == X_anom.shape
    assert cols == feature_names

    top_df = rank_top_pca_contributions(feat_errs, cols, top_k=3)

    assert isinstance(top_df, pd.DataFrame)
    assert len(top_df) == 3
    assert list(top_df.columns) == ["feature", "mean_reconstruction_error"]
    # Sensor 2 broke the correlation, so it must be identified as top contributor
    assert top_df.iloc[0]["feature"] == "sensor_2"
    # Values should be monotonically decreasing
    assert top_df["mean_reconstruction_error"].is_monotonic_decreasing


def test_isolation_forest_attribution():
    """Verifies Isolation Forest attribution computation and top-k ranking."""
    X_healthy, X_anom, feature_names = _generate_explain_data()

    detector = IsolationForestDetector(n_estimators=30, random_state=42)
    detector.fit(X_healthy)

    attr_vals, cols = compute_isolation_forest_shap(detector, X_anom, feature_names=feature_names)

    assert isinstance(attr_vals, np.ndarray)
    assert attr_vals.shape == X_anom.shape
    assert cols == feature_names

    top_df = rank_top_contributing_features(attr_vals, cols, top_k=3)

    assert isinstance(top_df, pd.DataFrame)
    assert len(top_df) == 3
    assert list(top_df.columns) == ["feature", "mean_abs_shap"]
    assert top_df["mean_abs_shap"].is_monotonic_decreasing
