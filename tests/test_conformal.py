"""
Unit Tests for Non-Overlapping Block Conformal Prediction.
"""

import numpy as np
from src.evaluation.conformal import BlockConformalCalibrator


def test_block_conformal_calibrator():
    """Tests non-overlapping block calibration, block maxima, and empirical coverage."""
    np.random.seed(42)
    # 2000 samples = 100 blocks of length 20
    cal_scores = np.random.exponential(scale=1.0, size=2000)

    calibrator = BlockConformalCalibrator(block_length=20, alpha=0.05)
    calibrator.fit(cal_scores)

    assert calibrator.n_blocks_ == 100
    assert len(calibrator.block_maxima_) == 100
    assert calibrator.q_hat_ > 0.0

    # Test online block evaluation
    test_scores = np.random.exponential(scale=1.0, size=200)
    p_vals, alarms = calibrator.evaluate_sequence(test_scores)

    assert len(p_vals) == 200
    assert len(alarms) == 200
    assert np.all(p_vals >= 0.0) and np.all(p_vals <= 1.0)

    # Coverage evaluation on independent test data
    held_out = np.random.exponential(scale=1.0, size=4000)
    cov_dict = calibrator.evaluate_block_coverage(held_out)

    assert "empirical_block_far" in cov_dict
    assert "empirical_coverage" in cov_dict
    # False alarm rate should be close to alpha=0.05
    assert cov_dict["empirical_block_far"] < 0.12
