"""
PCA Reconstruction Contribution Attribution Module.
"""

from typing import Union, Tuple, Optional
import numpy as np
import pandas as pd

from src.models.pca_detector import PCADetector
from src.utils import logger


def compute_pca_feature_attribution(
    detector: PCADetector,
    X: Union[np.ndarray, pd.DataFrame],
    feature_names: Optional[list] = None
) -> Tuple[np.ndarray, list]:
    """
    Computes per-variable squared reconstruction error matrix for samples X.
    
    Args:
        detector: Fitted PCADetector instance.
        X: Sample matrix (n_samples, n_features).
        feature_names: Optional list of feature names.
        
    Returns:
        Tuple of (reconstruction_error_matrix of shape (n_samples, n_features), feature_names list).
    """
    if detector.pca_ is None or detector.scaler_ is None:
        raise ValueError("PCADetector must be fitted before computing feature attribution.")

    X_arr = np.asarray(X, dtype=np.float64)
    feat_errors = detector.feature_reconstruction_error(X_arr)

    cols = feature_names if feature_names is not None else [f"feature_{i}" for i in range(X_arr.shape[1])]
    return feat_errors, cols


def rank_top_pca_contributions(
    feat_errors: np.ndarray,
    feature_names: list,
    top_k: int = 5
) -> pd.DataFrame:
    """
    Ranks top K feature reconstruction error contributions across flagged anomalous samples.
    """
    mean_feat_err = np.mean(feat_errors, axis=0)
    df_rank = pd.DataFrame({
        "feature": feature_names,
        "mean_reconstruction_error": mean_feat_err
    }).sort_values(by="mean_reconstruction_error", ascending=False).reset_index(drop=True)
    return df_rank.head(top_k)
