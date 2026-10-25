"""
TreeSHAP Attribution Module for Isolation Forest Anomaly Detector.
"""

from typing import Union, Tuple, Optional
import numpy as np
import pandas as pd

from src.models.isolation_forest import IsolationForestDetector
from src.utils import logger

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    shap = None
    _HAS_SHAP = False


def _compute_perturbation_attribution(
    detector: IsolationForestDetector,
    X_arr: np.ndarray,
    cols: list
) -> Tuple[np.ndarray, list]:
    """Fallback attribution using feature mean replacement perturbation."""
    n_samples, n_features = X_arr.shape
    baseline_score = detector.decision_function(X_arr)
    attribution_vals = np.zeros((n_samples, n_features), dtype=np.float64)

    for j in range(n_features):
        X_permuted = X_arr.copy()
        X_permuted[:, j] = np.mean(X_permuted[:, j])
        permuted_score = detector.decision_function(X_permuted)
        attribution_vals[:, j] = baseline_score - permuted_score

    return attribution_vals, cols


def compute_isolation_forest_shap(
    detector: IsolationForestDetector,
    X: Union[np.ndarray, pd.DataFrame],
    feature_names: Optional[list] = None
) -> Tuple[np.ndarray, list]:
    """
    Computes TreeSHAP values for samples X using fitted IsolationForestDetector.
    Falls back gracefully to perturbation-based attribution if SHAP is unavailable.
    
    Args:
        detector: Fitted IsolationForestDetector instance.
        X: Sample matrix (n_samples, n_features).
        feature_names: Optional list of feature names.
        
    Returns:
        Tuple of (shap_values matrix of shape (n_samples, n_features), feature_names list).
    """
    if detector.model_ is None or detector.scaler_ is None:
        raise ValueError("IsolationForestDetector must be fitted before computing SHAP values.")

    X_arr = np.asarray(X, dtype=np.float64)
    X_scaled = detector.scaler_.transform(X_arr)

    cols = feature_names if feature_names is not None else [f"feature_{i}" for i in range(X_arr.shape[1])]

    if not _HAS_SHAP:
        logger.warning("SHAP package not installed. Falling back to feature perturbation importance.")
        return _compute_perturbation_attribution(detector, X_arr, cols)

    try:
        explainer = shap.TreeExplainer(detector.model_)
        shap_vals = explainer.shap_values(X_scaled)
        
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
            
        logger.info(f"TreeSHAP values computed successfully for shape {shap_vals.shape}.")
        return shap_vals, cols
    except Exception as e:
        logger.warning(f"TreeSHAP calculation failed ({e}). Falling back to feature perturbation importance.")
        return _compute_perturbation_attribution(detector, X_arr, cols)


def rank_top_contributing_features(
    shap_values: np.ndarray,
    feature_names: list,
    top_k: int = 5
) -> pd.DataFrame:
    """
    Ranks top K feature contributions across flagged anomalous samples.
    """
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    df_rank = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
    return df_rank.head(top_k)
