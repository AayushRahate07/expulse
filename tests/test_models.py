"""
Unit Tests for Unified Scikit-Learn BaseEstimator Anomaly Detectors.
"""

import numpy as np
import pytest
from sklearn.base import BaseEstimator

from src.models import RuleBasedDetector, PCADetector, IsolationForestDetector


@pytest.fixture
def synthetic_data():
    """Generates synthetic multivariate normal healthy training data and an outlier sample."""
    np.random.seed(42)
    # 100 nominal samples, 5 features
    X_healthy = np.random.normal(loc=0.0, scale=1.0, size=(100, 5))
    
    # 10 test samples: 8 nominal, 2 clear anomaly spikes
    X_test = np.random.normal(loc=0.0, scale=1.0, size=(10, 5))
    X_test[8, :] += 10.0  # Large multivariate spike
    X_test[9, 2] += 15.0  # Large single-feature spike
    
    return X_healthy, X_test


@pytest.mark.parametrize("detector_cls, kwargs", [
    (RuleBasedDetector, {"k_std": 3.0}),
    (PCADetector, {"n_components": 0.95}),
    (IsolationForestDetector, {"n_estimators": 50, "random_state": 42}),
])
def test_detector_interface(detector_cls, kwargs, synthetic_data):
    """Verifies BaseEstimator inheritance, fit/predict API, and output shapes."""
    X_train, X_test = synthetic_data
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
    assert scores[8] > scores[0]  # Anomaly score higher than nominal sample
