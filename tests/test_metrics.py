"""
Unit Tests for Evaluation Metrics and 3-Minute Lead Time Computation.
"""

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_binary_classification_metrics,
    compute_first_sustained_detection,
    compute_lead_time_metrics,
)


def test_binary_classification_metrics():
    """Verify Precision, Recall, F1, and FAR calculations."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 0, 0])

    metrics = compute_binary_classification_metrics(y_true, y_pred)
    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 1
    assert metrics["true_negatives"] == 3
    assert metrics["false_negatives"] == 2

    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 4)
    assert metrics["false_alarm_rate"] == pytest.approx(1 / 4)


def test_first_sustained_detection():
    """Verify sliding window for N consecutive flags."""
    preds = np.array([0, 1, 1, 0, 1, 1, 1, 0])
    # 3 consecutive ones start at index 4 (0-based)
    idx = compute_first_sustained_detection(preds, n_consecutive=3)
    assert idx == 4

    # No 4 consecutive ones
    idx_4 = compute_first_sustained_detection(preds, n_consecutive=4)
    assert idx_4 is None


def test_lead_time_3_minute_interval():
    """Verify lead time calculation in samples and minutes (Refinement 3)."""
    # 200 samples total, fault introduced at sample 160
    preds = np.zeros(200, dtype=int)
    # First sustained detection (3 consecutive ones) starts at 0-based index 164 (sample 165)
    preds[164:167] = 1

    lead_meta = compute_lead_time_metrics(
        predictions=preds,
        fault_intro_sample=160,
        sampling_interval_minutes=3,
        n_consecutive=3
    )

    assert lead_meta["detected"] is True
    assert lead_meta["first_sustained_sample"] == 165
    assert lead_meta["lead_time_samples"] == 5  # 165 - 160 = +5 samples
    assert lead_meta["lead_time_minutes"] == 15  # 5 * 3 = 15 minutes
    assert lead_meta["detection_delay_minutes"] == 15
