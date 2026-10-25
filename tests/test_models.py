"""
Unit Tests for Unified Scikit-Learn BaseEstimator Anomaly Detectors.
Includes tests for Hotelling's T^2, SPE, and Extended Isolation Forest (EIF).
"""

import numpy as np
from sklearn.base import BaseEstimator

from src.models import (
    RuleBasedDetector,
    PCADetector,
    IsolationForestDetector,
    ExtendedIsolationForestDetector,
)


def _generate_synthetic_data():
    """Generates synthetic multivariate normal healthy training data and outlier samples."""
    np.random.seed(42)
    X_healthy = np.random.normal(loc=0.0, scale=1.0, size=(100, 5))
    X_test = np.random.normal(loc=0.0, scale=1.0, size=(10, 5))
    X_test[8, :] += 10.0  # Large multivariate spike
    X_test[9, 2] += 15.0  # Large single-feature spike
    return X_healthy, X_test


def _assert_detector_contract(detector_cls, kwargs):
    """Verifies BaseEstimator inheritance, fit/predict API, and output shapes."""
    X_train, X_test = _generate_synthetic_data()
    detector = detector_cls(**kwargs)

    # 1. BaseEstimator inheritance
    assert isinstance(detector, BaseEstimator)

    # 2. Fit on healthy training data
    detector.fit(X_train)
    assert hasattr(detector, "threshold_")

    # 3. Decision function
    scores = detector.decision_function(X_test)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (len(X_test),)

    # 4. Binary predictions
    preds = detector.predict(X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(X_test),)
    assert set(preds).issubset({0, 1})

    # 5. Outlier detection check (sample index 8 and 9 should be anomalous)
    assert preds[8] == 1 or preds[9] == 1
    assert scores[8] > scores[0]


def test_rule_based_detector_interface():
    _assert_detector_contract(RuleBasedDetector, {"k_std": 3.0})


def test_pca_detector_interface():
    _assert_detector_contract(PCADetector, {"n_components": 0.95})


def test_pca_hotelling_t2_and_spe_dual_subspace():
    """
    Verifies that PCADetector monitors both SPE (residual) and Hotelling's T^2 (in-subspace).
    Specifically verifies that in-subspace overloads (blind to SPE) are caught by T^2.
    """
    np.random.seed(42)
    X_train = np.random.randn(200, 4)
    X_train[:, 1] = X_train[:, 0] * 3.0 + np.random.randn(200) * 0.05

    pca_det = PCADetector(n_components=2, metric="combined")
    pca_det.fit(X_train)

    assert hasattr(pca_det, "spe_threshold_")
    assert hasattr(pca_det, "t2_threshold_")
    assert pca_det.spe_threshold_ > 0
    assert pca_det.t2_threshold_ > 0

    # In-subspace overload: values surge along the exact correlation line
    X_in_subspace = np.array([[8.0, 24.0, 0.0, 0.0]])
    t2_val = pca_det.compute_t2(X_in_subspace)[0]
    combined_val = pca_det.compute_combined_index(X_in_subspace)[0]

    assert t2_val > pca_det.t2_threshold_
    assert combined_val > 1.0

    # Verify RBC attribution shape
    rbc = pca_det.feature_rbc_attribution(X_in_subspace)
    assert rbc.shape == (1, 4)


def test_isolation_forest_detector_interface():
    _assert_detector_contract(IsolationForestDetector, {"n_estimators": 50, "random_state": 42})


def test_extended_isolation_forest_detector_interface():
    """Verifies Extended Isolation Forest (EIF) with random hyperplanes on correlated data."""
    _assert_detector_contract(
        ExtendedIsolationForestDetector,
        {"n_estimators": 25, "max_samples": 64, "random_state": 42}
    )
