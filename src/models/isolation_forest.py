"""
Isolation Forest Anomaly Detector implementing sklearn BaseEstimator API.
"""

from typing import Union, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class IsolationForestDetector(BaseEstimator, ClassifierMixin):
    """
    Isolation Forest Anomaly Detector.
    Fits a StandardScaler and Isolation Forest on healthy training data.
    
    Decision Function:
        Computes continuous anomaly score derived from path length:
        Higher score = more anomalous (inverted sklearn raw score_samples).
    Predict:
        Flags sample as anomalous (1) if score >= threshold learned on healthy data.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: str = "auto",
        random_state: int = 42,
        threshold_percentile: Optional[float] = None
    ) -> None:
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.threshold_percentile = threshold_percentile
        self.scaler_: Optional[StandardScaler] = None
        self.model_: Optional[IsolationForest] = None
        self.threshold_: float = 0.0

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None) -> "IsolationForestDetector":
        """
        Scales features and fits IsolationForest on healthy training data X.
        Sets internal threshold on healthy decision scores.
        """
        X_arr = np.asarray(X, dtype=np.float64)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_arr)

        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state
        )
        self.model_.fit(X_scaled)

        # Compute decision scores on healthy training data
        train_scores = self.decision_function(X_arr)

        if self.threshold_percentile is not None:
            self.threshold_ = np.percentile(train_scores, self.threshold_percentile)
        else:
            q25, q75 = np.percentile(train_scores, [25, 75])
            iqr = q75 - q25
            self.threshold_ = q75 + 1.5 * iqr

        return self

    def decision_function(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Computes continuous anomaly score for each sample.
        Inverts score_samples so higher values indicate higher anomaly severity.
        """
        if self.scaler_ is None or self.model_ is None:
            raise ValueError("IsolationForestDetector must be fitted before calling decision_function.")

        X_arr = np.asarray(X, dtype=np.float64)
        X_scaled = self.scaler_.transform(X_arr)

        # Invert score_samples: score_samples is negative (smaller = more anomalous)
        # -score_samples makes higher = more anomalous
        raw_scores = self.model_.score_samples(X_scaled)
        scores = -raw_scores
        return scores

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Returns binary predictions: 0 for healthy/nominal, 1 for anomalous.
        """
        scores = self.decision_function(X)
        return (scores >= self.threshold_).astype(int)
