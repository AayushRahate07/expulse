"""
PCA Reconstruction Error Anomaly Detector implementing sklearn BaseEstimator API.
"""

from typing import Union, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCADetector(BaseEstimator, ClassifierMixin):
    """
    PCA Reconstruction Error Anomaly Detector.
    Fits a StandardScaler and PCA on healthy training data.
    
    Decision Function:
        Computes Squared Prediction Error (SPE / Reconstruction Error):
        SPE_i = || x_i - x_reconstructed_i ||^2
    Predict:
        Flags sample as anomalous (1) if SPE_i >= threshold learned on healthy data.
    """

    def __init__(
        self,
        n_components: Union[int, float] = 0.95,
        threshold_percentile: Optional[float] = None
    ) -> None:
        self.n_components = n_components
        self.threshold_percentile = threshold_percentile
        self.scaler_: Optional[StandardScaler] = None
        self.pca_: Optional[PCA] = None
        self.threshold_: float = 0.0

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None) -> "PCADetector":
        """
        Scales features and fits PCA on healthy training data X.
        Sets internal threshold on healthy reconstruction errors.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_arr)

        self.pca_ = PCA(n_components=self.n_components)
        self.pca_.fit(X_scaled)

        # Compute reconstruction error on healthy training set
        train_errors = self.decision_function(X_arr)

        if self.threshold_percentile is not None:
            self.threshold_ = np.percentile(train_errors, self.threshold_percentile)
        else:
            q25, q75 = np.percentile(train_errors, [25, 75])
            iqr = q75 - q25
            self.threshold_ = q75 + 1.5 * iqr

        return self

    def _transform_and_reconstruct(self, X_scaled: np.ndarray) -> np.ndarray:
        """Projects scaled X to latent subspace and reconstructs back to original feature space."""
        latent = self.pca_.transform(X_scaled)
        reconstructed = self.pca_.inverse_transform(latent)
        return reconstructed

    def decision_function(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Computes continuous reconstruction error (SPE) for each sample.
        """
        if self.scaler_ is None or self.pca_ is None:
            raise ValueError("PCADetector must be fitted before calling decision_function.")

        X_arr = np.asarray(X, dtype=np.float64)
        X_scaled = self.scaler_.transform(X_arr)
        X_rec = self._transform_and_reconstruct(X_scaled)

        # Squared reconstruction error per sample across features
        spe = np.sum((X_scaled - X_rec) ** 2, axis=1)
        return spe

    def feature_reconstruction_error(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Computes per-variable squared reconstruction error matrix of shape (n_samples, n_features).
        Used for attribution ranking.
        """
        if self.scaler_ is None or self.pca_ is None:
            raise ValueError("PCADetector must be fitted before calling feature_reconstruction_error.")

        X_arr = np.asarray(X, dtype=np.float64)
        X_scaled = self.scaler_.transform(X_arr)
        X_rec = self._transform_and_reconstruct(X_scaled)

        # Per-feature squared error
        feat_err = (X_scaled - X_rec) ** 2
        return feat_err

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Returns binary predictions: 0 for healthy/nominal, 1 for anomalous.
        """
        errors = self.decision_function(X)
        return (errors >= self.threshold_).astype(int)
