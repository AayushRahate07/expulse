"""
Unit Tests for Temporal Feature Augmentation (R^50 -> R^200).
"""

import numpy as np
import pandas as pd
from src.data.features import generate_temporal_augmented_features, compute_rolling_linear_slope


def test_compute_rolling_linear_slope():
    """Tests calculation of linear trend slope on known ramp and flat windows."""
    # Ramp with slope 2.0: [0, 2, 4, 6, 8]
    ramp = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    slope = compute_rolling_linear_slope(ramp)
    assert np.isclose(slope, 2.0, atol=1e-5)

    # Flat line: [5, 5, 5, 5]
    flat = np.array([5.0, 5.0, 5.0, 5.0])
    assert np.isclose(compute_rolling_linear_slope(flat), 0.0, atol=1e-5)


def test_generate_temporal_augmented_features_shape():
    """Verifies matrix shape expansion from P to 4*P features."""
    np.random.seed(42)
    # 50 rows, 5 features
    data = {f"sensor_{i}": np.random.randn(50) for i in range(5)}
    df = pd.DataFrame(data)

    X_aug, names = generate_temporal_augmented_features(df, feature_cols=list(df.columns), window_size=5)

    # 5 sensors * 4 = 20 augmented features
    assert X_aug.shape == (50, 20)
    assert len(names) == 20
    assert not np.isnan(X_aug).any()
