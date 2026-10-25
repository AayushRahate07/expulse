"""
Reproducibility Verification Test Suite (Section 8 / Week 12 Deliverable).
"""

import math
import numpy as np
from src.config import RANDOM_SEED
from src.models import IsolationForestDetector
from src.utils import set_seed

try:
    import pytest
    approx = pytest.approx
except ImportError:
    def approx(expected, rel=1e-5, abs=1e-5):
        class _Approx:
            def __eq__(self, other):
                return math.isclose(other, expected, rel_tol=rel, abs_tol=abs)
            def __repr__(self):
                return f"approx({expected})"
        return _Approx()


def test_model_reproducibility():
    """Verifies that setting random seed produces deterministic model thresholds and predictions."""
    np.random.seed(RANDOM_SEED)
    X_train = np.random.normal(0, 1, (100, 10))
    X_test = np.random.normal(0, 1, (20, 10))
    X_test[15:, :] += 5.0

    # Model 1 fit
    set_seed(RANDOM_SEED)
    if_1 = IsolationForestDetector(n_estimators=50, random_state=RANDOM_SEED)
    if_1.fit(X_train)
    scores_1 = if_1.decision_function(X_test)

    # Model 2 fit with same seed
    set_seed(RANDOM_SEED)
    if_2 = IsolationForestDetector(n_estimators=50, random_state=RANDOM_SEED)
    if_2.fit(X_train)
    scores_2 = if_2.decision_function(X_test)

    # Assert exact score equality
    np.testing.assert_allclose(scores_1, scores_2, atol=1e-5)
    assert if_1.threshold_ == approx(if_2.threshold_, abs=1e-5)
