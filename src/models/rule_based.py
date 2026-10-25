"""
Rule-Based Per-Sensor Anomaly Detector implementing sklearn BaseEstimator API.
"""

from typing import Union, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class RuleBasedDetector(BaseEstimator, ClassifierMixin):
    """
    Rule-Based Baseline Detector.
    Fits independent upper/lower threshold bands (mean +- k * std) per feature on healthy training data.
    
    Decision Function:
        Calculates the maximum normalized out-of-bounds deviation across all features for each sample.
    Predict:
        Flags a sample as anomalous (1) if decision_function score exceeds the learned threshold.
    """

    def __init__(
        self,
        k_std: float = 3.0,
        threshold_percentile: Optional[float] = None,
        min_threshold: float = 1.0
    ) -> None:
        self.k_std = k_std
        self.threshold_percentile = threshold_percentile
        self.min_threshold = min_threshold
        self.means_: Optional[np.ndarray] = None
        self.stds_: Optional[np.ndarray] = None
        self.upper_bounds_: Optional[np.ndarray] = None
        self.lower_bounds_: Optional[np.ndarray] = None
        self.threshold_: float = 0.0

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None) -> "RuleBasedDetector":
        """
        Fits per-feature mean and std on healthy training data X.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        self.means_ = np.mean(X_arr, axis=0)
        self.stds_ = np.std(X_arr, axis=0, ddof=1)
        
        # Avoid division by zero for invariant features
        self.stds_[self.stds_ == 0.0] = 1e-8

        self.upper_bounds_ = self.means_ + self.k_std * self.stds_
        self.lower_bounds_ = self.means_ - self.k_std * self.stds_

        # Calculate decision scores on healthy training data
        train_scores = self.decision_function(X_arr)

        if self.threshold_percentile is not None:
            self.threshold_ = float(np.percentile(train_scores, self.threshold_percentile))
        else:
            # Default Q3 + 1.5 * IQR rule with floor to prevent 0-threshold collapse on nominal data
            q25, q75 = np.percentile(train_scores, [25, 75])
            iqr = q75 - q25
            self.threshold_ = float(max(self.min_threshold, q75 + 1.5 * iqr))

        return self

    def decision_function(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Computes continuous anomaly score for each sample:
        Max normalized deviation outside healthy bounds across all features.
        Score = 0 if sample is strictly inside all feature bands.
        """
        if self.means_ is None or self.stds_ is None:
            raise ValueError("RuleBasedDetector must be fitted before calling decision_function.")

        X_arr = np.asarray(X, dtype=np.float64)
        
        upper_dev = (X_arr - self.upper_bounds_) / self.stds_
        lower_dev = (self.lower_bounds_ - X_arr) / self.stds_

        # Take max positive deviation across upper and lower bounds
        dev = np.maximum(upper_dev, lower_dev)
        dev = np.maximum(0.0, dev)

        # Max deviation over all features for each sample
        scores = np.max(dev, axis=1)
        return scores

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Returns binary predictions: 0 for healthy/nominal, 1 for anomalous.
        """
        scores = self.decision_function(X)
        return (scores >= self.threshold_).astype(int)
