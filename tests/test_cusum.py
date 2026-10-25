"""
Unit Tests for Page's CUSUM Sequential Change-Point Detection.
"""

import numpy as np
from src.evaluation.cusum import CUSUMDetector, compute_cusum_lead_time


def test_cusum_calibration_and_shift_detection():
    """Tests CUSUM parameter calibration and detection of a deliberate mean shift."""
    np.random.seed(42)
    # Healthy calibration scores (mean 2.0, std 0.5)
    healthy = np.random.normal(loc=2.0, scale=0.5, size=2000)

    detector = CUSUMDetector(k_factor=0.5)
    detector.fit(healthy, target_arl0=1000.0)

    assert detector.mu_0_ > 1.9 and detector.mu_0_ < 2.1
    assert detector.sigma_0_ > 0.4 and detector.sigma_0_ < 0.6
    assert detector.k_ > 0.0
    assert detector.h_ > detector.sigma_0_

    # Test sequence: 160 nominal samples followed by an abrupt out-of-control jump
    test_nominal = np.random.normal(loc=2.0, scale=0.5, size=160)
    test_fault = np.random.normal(loc=5.0, scale=0.5, size=100)
    test_seq = np.concatenate([test_nominal, test_fault])

    c_stat, alarms = detector.detect(test_seq)
    assert len(c_stat) == len(test_seq)
    assert len(alarms) == len(test_seq)

    first_alarm = detector.compute_first_detection(test_seq)
    assert first_alarm is not None
    # Alarm should trigger promptly after sample 160 (within 10 samples)
    assert first_alarm >= 155
    assert first_alarm <= 175

    lead_meta = compute_cusum_lead_time(test_seq, detector, fault_intro_sample=160)
    assert lead_meta["detected"] is True
    assert lead_meta["lead_time_samples"] is not None
