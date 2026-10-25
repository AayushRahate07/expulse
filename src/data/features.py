"""
Temporal Feature Augmentation Module for Industrial Sensor Telemetry.
Computes rolling mean, volatility (std), and linear trend velocity (slope) over window w=15.
Expands sensor space from R^50 to R^200.
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd


def compute_rolling_linear_slope(window_data: np.ndarray) -> float:
    """
    Computes linear regression slope beta over a 1D window of length w:
        beta = sum((i - i_bar) * (y_i - y_bar)) / sum((i - i_bar)^2)
    """
    w = len(window_data)
    if w < 2:
        return 0.0
    x = np.arange(w, dtype=np.float64)
    x_mean = (w - 1.0) / 2.0
    x_diff = x - x_mean
    denom = np.sum(x_diff ** 2)
    y_diff = window_data - np.mean(window_data)
    return float(np.sum(x_diff * y_diff) / denom)


def generate_temporal_augmented_features(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    window_size: int = 15,
) -> Tuple[np.ndarray, List[str]]:
    """
    Augments 50 sensors to 200 features:
        - Original instantaneous value x_{j,t}
        - Rolling local mean mu_{j,t}^{(w)}
        - Rolling local volatility sigma_{j,t}^{(w)}
        - Rolling linear velocity beta_{j,t}^{(w)}
        
    Args:
        df: Input DataFrame containing sensor measurements.
        feature_cols: List of sensor columns. If None, auto-detected.
        window_size: Number of rolling samples (default w=15 samples = 45 mins).
        
    Returns:
        Augmented feature matrix X_augmented of shape (N, 4 * P), and list of 200 feature names.
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in ["faultnumber", "simulationrun", "sample"]]

    df_feats = df[feature_cols].copy()

    # 1. Instantaneous values (50)
    X_orig = df_feats.values

    # 2. Rolling Local Mean (50)
    # min_periods=1 ensures no NaNs at beginning of run
    rolling_mean_df = df_feats.rolling(window=window_size, min_periods=1).mean()
    X_mean = rolling_mean_df.values

    # 3. Rolling Local Volatility / Standard Deviation (50)
    rolling_std_df = df_feats.rolling(window=window_size, min_periods=2).std().fillna(0.0)
    X_std = rolling_std_df.values

    # 4. Rolling Linear Trend Slope (50)
    # Fast vectorized rolling slope via uniform convolution weights
    w = window_size
    i_arr = np.arange(w, dtype=np.float64)
    i_mean = (w - 1.0) / 2.0
    i_diff = i_arr - i_mean
    denom = np.sum(i_diff ** 2)
    weights = i_diff / denom  # 1D FIR filter for linear slope

    # Apply rolling linear trend filter per column
    n_samples, n_features = X_orig.shape
    X_slope = np.zeros_like(X_orig)

    # Vectorized convolution per sensor column
    for j in range(n_features):
        col_vals = X_orig[:, j]
        # Convolve with slope weights
        padded_vals = np.pad(col_vals, (w - 1, 0), mode="edge")
        slope_full = np.convolve(padded_vals, weights[::-1], mode="valid")
        X_slope[:, j] = slope_full[:n_samples]

    # Combine into R^200 feature vector
    X_augmented = np.hstack([X_orig, X_mean, X_std, X_slope])

    # Build descriptive feature names
    augmented_names = (
        [f"{col}" for col in feature_cols]
        + [f"{col}_roll_mean_{w}" for col in feature_cols]
        + [f"{col}_roll_std_{w}" for col in feature_cols]
        + [f"{col}_roll_slope_{w}" for col in feature_cols]
    )

    return X_augmented.astype(np.float32), augmented_names
