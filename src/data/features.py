"""
Temporal Feature Augmentation Module for Industrial Sensor Telemetry.
Computes rolling mean, volatility (std), and linear trend velocity (slope).
Supports both uniform windows (w=15) and physics-informed Multi-Scale Dynamic Windows:
  - Fast Hydraulic/Pressure Pool: w=3 (9 min)
  - Intermediate Flow/Level Pool: w=8 (24 min)
  - Slow Thermal/Composition Pool: w=20 (60 min)
Expands sensor space from R^50 to R^200.
"""

from typing import List, Tuple, Optional, Dict, Union
import numpy as np
import pandas as pd


# Physical Dynamic Time-Scale Pools based on TEP process unit time constants
# (Downs & Vogel 1993, McAvoy & Ye 1994)
FAST_PRESSURE_POOL = {
    "xmeas_7", "xmeas_13", "xmeas_16", "xmeas_20", "xmv_6"
}

INTERMEDIATE_FLOW_POOL = {
    "xmeas_1", "xmeas_2", "xmeas_3", "xmeas_4", "xmeas_5", "xmeas_6",
    "xmeas_8", "xmeas_10", "xmeas_12", "xmeas_14", "xmeas_15", "xmeas_17", "xmeas_19",
    "xmv_1", "xmv_2", "xmv_3", "xmv_4", "xmv_7", "xmv_8"
}

SLOW_THERMAL_POOL = {
    "xmeas_9", "xmeas_11", "xmeas_18", "xmeas_21", "xmeas_22",
    "xmeas_23", "xmeas_24", "xmeas_25", "xmeas_26", "xmeas_27", "xmeas_28",
    "xmeas_29", "xmeas_30", "xmeas_31", "xmeas_32", "xmeas_33", "xmeas_34",
    "xmeas_35", "xmeas_36", "xmeas_37", "xmeas_38",
    "xmv_10", "xmv_11"
}


def get_sensor_dynamic_window(sensor_tag: str, multiscale: bool = True, default_w: int = 15) -> int:
    """Returns the physics-informed rolling window size for a specific sensor channel."""
    if not multiscale:
        return default_w
    if sensor_tag in FAST_PRESSURE_POOL:
        return 3   # 9 minutes (fast hydraulic dynamics)
    elif sensor_tag in INTERMEDIATE_FLOW_POOL:
        return 8   # 24 minutes (level & flow integration)
    elif sensor_tag in SLOW_THERMAL_POOL:
        return 20  # 60 minutes (thermal lag & analyzer cycles)
    return default_w


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


def _compute_fir_slope(col_vals: np.ndarray, w: int) -> np.ndarray:
    """Fast vectorized 1D FIR filter for linear regression slope over rolling window w."""
    n_samples = len(col_vals)
    if w < 2 or n_samples == 0:
        return np.zeros_like(col_vals)
    i_arr = np.arange(w, dtype=np.float64)
    i_mean = (w - 1.0) / 2.0
    i_diff = i_arr - i_mean
    denom = np.sum(i_diff ** 2)
    weights = i_diff / denom
    padded_vals = np.pad(col_vals, (w - 1, 0), mode="edge")
    slope_full = np.convolve(padded_vals, weights[::-1], mode="valid")
    return slope_full[:n_samples]


def generate_temporal_augmented_features(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    window_size: Union[int, str] = 15,
    multiscale: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """
    Augments 50 sensors to 200 features:
        - Original instantaneous value x_{j,t}
        - Rolling local mean mu_{j,t}^{(w_j)}
        - Rolling local volatility sigma_{j,t}^{(w_j)}
        - Rolling linear velocity beta_{j,t}^{(w_j)}
        
    Args:
        df: Input DataFrame containing sensor measurements.
        feature_cols: List of sensor columns. If None, auto-detected.
        window_size: Uniform window size (default 15) or 'multiscale'.
        multiscale: If True, uses dynamic physics-informed windows (w=3, 8, 20).
        
    Returns:
        Augmented feature matrix X_augmented of shape (N, 4 * P), and list of 200 feature names.
    """
    if str(window_size).lower() == "multiscale":
        multiscale = True

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in ["faultnumber", "simulationrun", "sample"]]

    df_feats = df[feature_cols].copy()
    X_orig = df_feats.values
    n_samples, n_features = X_orig.shape

    X_mean = np.zeros_like(X_orig, dtype=np.float32)
    X_std = np.zeros_like(X_orig, dtype=np.float32)
    X_slope = np.zeros_like(X_orig, dtype=np.float32)

    mean_names = []
    std_names = []
    slope_names = []

    # Process each column with its tailored or uniform window
    for j, col in enumerate(feature_cols):
        w = get_sensor_dynamic_window(col, multiscale=multiscale, default_w=int(window_size) if isinstance(window_size, int) else 15)
        col_vals = df_feats[col].values

        # Rolling mean
        rolling_mean = df_feats[col].rolling(window=w, min_periods=1).mean().values
        X_mean[:, j] = rolling_mean

        # Rolling std
        rolling_std = df_feats[col].rolling(window=w, min_periods=2).std().fillna(0.0).values
        X_std[:, j] = rolling_std

        # Rolling slope (vectorized FIR)
        X_slope[:, j] = _compute_fir_slope(col_vals, w)

        mean_names.append(f"{col}_roll_mean_{w}")
        std_names.append(f"{col}_roll_std_{w}")
        slope_names.append(f"{col}_roll_slope_{w}")

    # Combine into R^200 feature vector
    X_augmented = np.hstack([X_orig, X_mean, X_std, X_slope])
    augmented_names = list(feature_cols) + mean_names + std_names + slope_names

    return X_augmented.astype(np.float32), augmented_names
