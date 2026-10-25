"""
Dual-Subspace PCA Anomaly Detector implementing Hotelling's T^2 and SPE (Q-Statistic).
Implements classical Jackson-Mudholkar and Qin combined index monitoring.
"""

from typing import Union, Optional, Literal
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCADetector(BaseEstimator, ClassifierMixin):
    """
    Dual-Subspace PCA Anomaly Detector (SPE + Hotelling's T^2).
    
    Monitors both:
    1. Residual Subspace (SPE / Q-statistic): Measures broken cross-sensor correlation:
       SPE_i = || x_scaled_i - x_hat_i ||^2
    2. Principal Subspace (Hotelling's T^2): Measures in-subspace magnitude overloads:
       T^2_i = sum_{j=1}^k (t_{ij}^2 / lambda_j)
    3. Combined Index (phi): Normalized multi-subspace monitoring:
       phi_i = (SPE_i / SPE_lim) + (T^2_i / T^2_lim)
       
    Args:
        n_components: Number or variance ratio of principal components (default 0.95).
        threshold_percentile: Quantile for threshold on healthy data (e.g. 99.0).
        metric: Anomaly metric to monitor: 'combined' (default), 'spe', or 't2'.
    """

    def __init__(
        self,
        n_components: Union[int, float] = 0.95,
        threshold_percentile: Optional[float] = None,
        metric: Literal["combined", "spe", "t2"] = "combined",
    ) -> None:
        self.n_components = n_components
        self.threshold_percentile = threshold_percentile
        self.metric = metric
        self.scaler_: Optional[StandardScaler] = None
        self.pca_: Optional[PCA] = None
        self.threshold_: float = 0.0
        self.spe_threshold_: float = 0.0
        self.t2_threshold_: float = 0.0

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None) -> "PCADetector":
        """
        Fits StandardScaler and PCA on healthy training data X.
        Computes empirical control limits for SPE, T^2, and the combined index phi.
        """
        X_arr = np.asarray(X, dtype=np.float64)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_arr)

        self.pca_ = PCA(n_components=self.n_components)
        self.pca_.fit(X_scaled)

        # 1. Compute SPE on healthy training data
        spe_train = self._compute_spe_internal(X_scaled)

        # 2. Compute Hotelling's T^2 on healthy training data
        t2_train = self._compute_t2_internal(X_scaled)

        # Calculate individual control limits on healthy training set
        if self.threshold_percentile is not None:
            self.spe_threshold_ = float(np.percentile(spe_train, self.threshold_percentile))
            self.t2_threshold_ = float(np.percentile(t2_train, self.threshold_percentile))
        else:
            # Robust IQR threshold
            q25_s, q75_s = np.percentile(spe_train, [25, 75])
            self.spe_threshold_ = float(q75_s + 1.5 * (q75_s - q25_s))

            q25_t, q75_t = np.percentile(t2_train, [25, 75])
            self.t2_threshold_ = float(q75_t + 1.5 * (q75_t - q25_t))

        # Avoid zero-division in degenerate cases
        self.spe_threshold_ = max(self.spe_threshold_, 1e-6)
        self.t2_threshold_ = max(self.t2_threshold_, 1e-6)

        # 3. Compute combined index phi on training set
        phi_train = (spe_train / self.spe_threshold_) + (t2_train / self.t2_threshold_)

        if self.metric == "combined":
            if self.threshold_percentile is not None:
                self.threshold_ = float(np.percentile(phi_train, self.threshold_percentile))
            else:
                q25_p, q75_p = np.percentile(phi_train, [25, 75])
                self.threshold_ = float(q75_p + 1.5 * (q75_p - q25_p))
        elif self.metric == "spe":
            self.threshold_ = self.spe_threshold_
        elif self.metric == "t2":
            self.threshold_ = self.t2_threshold_
        else:
            raise ValueError(f"Unknown metric '{self.metric}'. Choose from 'combined', 'spe', 't2'.")

        return self

    def _compute_spe_internal(self, X_scaled: np.ndarray) -> np.ndarray:
        """Computes SPE (Squared Prediction Error) on already scaled data."""
        latent = self.pca_.transform(X_scaled)
        reconstructed = self.pca_.inverse_transform(latent)
        spe = np.sum((X_scaled - reconstructed) ** 2, axis=1)
        return spe

    def _compute_t2_internal(self, X_scaled: np.ndarray) -> np.ndarray:
        """Computes Hotelling's T^2 statistic on already scaled data."""
        latent = self.pca_.transform(X_scaled)
        # T^2 = sum_j (t_j^2 / lambda_j)
        eigenvalues = np.maximum(self.pca_.explained_variance_, 1e-8)
        t2 = np.sum((latent ** 2) / eigenvalues, axis=1)
        return t2

    def compute_spe(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Computes continuous SPE (residual subspace error) for each sample."""
        if self.scaler_ is None or self.pca_ is None:
            raise ValueError("PCADetector must be fitted before calling compute_spe.")
        X_arr = np.asarray(X, dtype=np.float64)
        X_scaled = self.scaler_.transform(X_arr)
        return self._compute_spe_internal(X_scaled)

    def compute_t2(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Computes Hotelling's T^2 (in-subspace magnitude shift) for each sample."""
        if self.scaler_ is None or self.pca_ is None:
            raise ValueError("PCADetector must be fitted before calling compute_t2.")
        X_arr = np.asarray(X, dtype=np.float64)
        X_scaled = self.scaler_.transform(X_arr)
        return self._compute_t2_internal(X_scaled)

    def compute_combined_index(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Computes normalized combined index phi = (SPE / SPE_lim) + (T^2 / T^2_lim)."""
        spe = self.compute_spe(X)
        t2 = self.compute_t2(X)
        return (spe / self.spe_threshold_) + (t2 / self.t2_threshold_)

    def decision_function(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Computes the continuous anomaly score based on configured metric."""
        if self.metric == "combined":
            return self.compute_combined_index(X)
        elif self.metric == "spe":
            return self.compute_spe(X)
        elif self.metric == "t2":
            return self.compute_t2(X)
        else:
            raise ValueError(f"Unknown metric '{self.metric}'.")

    def feature_reconstruction_error(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Computes per-variable squared reconstruction error matrix of shape (n_samples, n_features).
        """
        if self.scaler_ is None or self.pca_ is None:
            raise ValueError("PCADetector must be fitted before calling feature_reconstruction_error.")
        X_arr = np.asarray(X, dtype=np.float64)
        X_scaled = self.scaler_.transform(X_arr)
        latent = self.pca_.transform(X_scaled)
        reconstructed = self.pca_.inverse_transform(latent)
        return (X_scaled - reconstructed) ** 2

    def feature_rbc_attribution(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Computes Reconstruction-Based Contribution (RBC, Alcala & Qin 2009).
        Eliminates collinearity smearing by normalizing per-feature residual error
        by the diagonal of the residual projection matrix (I - P P^T)_{jj}:
            RBC_j = e_j^2 / (1 - sum_m P_{jm}^2)
        """
        feat_err = self.feature_reconstruction_error(X)
        # Components shape: (n_components, n_features)
        # Loading matrix P has shape (n_features, n_components)
        p_matrix = self.pca_.components_.T  # (n_features, n_components)
        p_diag_sum = np.sum(p_matrix ** 2, axis=1)  # (n_features,)
        c_tilde_diag = np.maximum(1.0 - p_diag_sum, 1e-4)  # (I - P P^T)_{jj}
        # Normalize each column by c_tilde_diag
        return feat_err / c_tilde_diag

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Returns binary predictions: 0 for healthy/nominal, 1 for anomalous."""
        scores = self.decision_function(X)
        return (scores >= self.threshold_).astype(int)
