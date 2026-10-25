"""
Unit Tests for Temporal Feature Augmentation (R^50 -> R^200) and Multi-Scale Physical Windows.
"""

import numpy as np
import pandas as pd
from src.data.features import (
    generate_temporal_augmented_features,
    compute_rolling_linear_slope,
    get_sensor_dynamic_window,
    FAST_PRESSURE_POOL,
    INTERMEDIATE_FLOW_POOL,
    SLOW_THERMAL_POOL,
)


def test_compute_rolling_linear_slope():
    """Tests calculation of linear trend slope on known ramp and flat windows."""
    ramp = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    slope = compute_rolling_linear_slope(ramp)
    assert np.isclose(slope, 2.0, atol=1e-5)

    flat = np.array([5.0, 5.0, 5.0, 5.0])
    assert np.isclose(compute_rolling_linear_slope(flat), 0.0, atol=1e-5)


def test_generate_temporal_augmented_features_shape():
    """Verifies matrix shape expansion from P to 4*P features."""
    np.random.seed(42)
    data = {f"sensor_{i}": np.random.randn(50) for i in range(5)}
    df = pd.DataFrame(data)

    X_aug, names = generate_temporal_augmented_features(df, feature_cols=list(df.columns), window_size=5)

    assert X_aug.shape == (50, 20)
    assert len(names) == 20
    assert not np.isnan(X_aug).any()


def test_multiscale_dynamic_window_pools():
    """Verifies that sensors map to appropriate dynamic window sizes based on physics."""
    # Fast pressure sensors: w = 3 (9 min)
    assert get_sensor_dynamic_window("xmeas_7", multiscale=True) == 3
    assert get_sensor_dynamic_window("xmeas_13", multiscale=True) == 3

    # Intermediate flow/level sensors: w = 8 (24 min)
    assert get_sensor_dynamic_window("xmeas_1", multiscale=True) == 8
    assert get_sensor_dynamic_window("xmeas_12", multiscale=True) == 8

    # Slow thermal/analyzer sensors: w = 20 (60 min)
    assert get_sensor_dynamic_window("xmeas_9", multiscale=True) == 20
    assert get_sensor_dynamic_window("xmeas_21", multiscale=True) == 20
    assert get_sensor_dynamic_window("xmeas_23", multiscale=True) == 20

    # Test DataFrame with mixed pools
    df = pd.DataFrame({
        "xmeas_7": np.random.randn(40),  # fast
        "xmeas_1": np.random.randn(40),  # intermediate
        "xmeas_9": np.random.randn(40),  # slow
    })
    X_aug, names = generate_temporal_augmented_features(df, multiscale=True)
    assert X_aug.shape == (40, 12)
    assert "xmeas_7_roll_mean_3" in names
    assert "xmeas_1_roll_mean_8" in names
    assert "xmeas_9_roll_mean_20" in names
