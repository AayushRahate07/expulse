"""
Extended Isolation Forest (EIF) Anomaly Detector.
Implements Hariri et al. (2019) with arbitrary-angle hyperplanes to eliminate
axis-aligned staircase artifacts on correlated sensor manifolds.
"""

from typing import Union, Optional, List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


def _c_factor(n: int) -> float:
    """Average path length of unsuccessful searches in a Binary Search Tree (BST)."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return float(2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n)


class _EIFNode:
    """Internal decision node or leaf node for Extended Isolation Tree."""
    __slots__ = ("left", "right", "normal", "intercept", "size", "is_leaf")

    def __init__(
        self,
        left: Optional["_EIFNode"] = None,
        right: Optional["_EIFNode"] = None,
        normal: Optional[np.ndarray] = None,
        intercept: Optional[float] = None,
        size: int = 1,
    ):
        self.left = left
        self.right = right
        self.normal = normal
        self.intercept = intercept
        self.size = size
        self.is_leaf = left is None and right is None


class ExtendedIsolationForestDetector(BaseEstimator, ClassifierMixin):
    """
    Extended Isolation Forest (EIF) Anomaly Detector.
    
    Instead of single-feature axis-aligned cuts (x_j <= c), EIF samples random
    normal vectors n ~ N(0, I) and random intercepts p in [min(x.n), max(x.n)],
    slicing feature space with arbitrary-angled hyperplanes: (x - p) . n <= 0.
    
    This eliminates the axis-aligned staircase artifact on 50D correlated sensor manifolds.
    
    Args:
        n_estimators: Number of isolation trees (default 100).
        max_samples: Subsample size per tree (default 256).
        threshold_percentile: Quantile for anomaly threshold on healthy data (default 99.0).
        random_state: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: Union[int, float] = 256,
        threshold_percentile: Optional[float] = None,
        random_state: Optional[int] = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.threshold_percentile = threshold_percentile
        self.random_state = random_state
        self.trees_: List[_EIFNode] = []
        self.threshold_: float = 0.5
        self.c_subsample_: float = 1.0

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None) -> "ExtendedIsolationForestDetector":
        """Builds ensemble of Extended Isolation Trees on healthy training data."""
        X_arr = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X_arr.shape

        rng = np.random.RandomState(self.random_state)

        # Determine subsample size
        if isinstance(self.max_samples, float):
            subsample_size = int(self.max_samples * n_samples)
        else:
            subsample_size = min(self.max_samples, n_samples)
        subsample_size = max(subsample_size, 2)

        self.c_subsample_ = _c_factor(subsample_size)
        max_depth = int(np.ceil(np.log2(max(subsample_size, 2))))

        self.trees_ = []
        for _ in range(self.n_estimators):
            sub_indices = rng.choice(n_samples, size=subsample_size, replace=False)
            X_sub = X_arr[sub_indices]
            tree = self._build_eif_tree(X_sub, current_depth=0, max_depth=max_depth, rng=rng)
            self.trees_.append(tree)

        # Set threshold on healthy training scores
        train_scores = self.decision_function(X_arr)
        if self.threshold_percentile is not None:
            self.threshold_ = float(np.percentile(train_scores, self.threshold_percentile))
        else:
            q25, q75 = np.percentile(train_scores, [25, 75])
            iqr = q75 - q25
            self.threshold_ = float(q75 + 1.5 * iqr)

        return self

    def _build_eif_tree(self, X: np.ndarray, current_depth: int, max_depth: int, rng: np.random.RandomState) -> _EIFNode:
        n_samples, n_features = X.shape
        if current_depth >= max_depth or n_samples <= 1:
            return _EIFNode(size=n_samples)

        # Sample random rotation vector from standard normal distribution
        normal = rng.normal(size=n_features)
        norm_val = np.linalg.norm(normal)
        if norm_val > 1e-8:
            normal = normal / norm_val

        # Project data along normal vector
        proj = X @ normal
        p_min, p_max = float(proj.min()), float(proj.max())
        if np.isclose(p_min, p_max):
            return _EIFNode(size=n_samples)

        # Uniform intercept along projection interval
        intercept = float(rng.uniform(p_min, p_max))
        left_mask = proj <= intercept
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return _EIFNode(size=n_samples)

        left_child = self._build_eif_tree(X[left_mask], current_depth + 1, max_depth, rng)
        right_child = self._build_eif_tree(X[right_mask], current_depth + 1, max_depth, rng)

        return _EIFNode(
            left=left_child,
            right=right_child,
            normal=normal,
            intercept=intercept,
            size=n_samples,
        )

    def _traverse_path(self, x: np.ndarray, node: _EIFNode, current_depth: int) -> float:
        if node.is_leaf:
            return float(current_depth + _c_factor(node.size))
        if np.dot(x, node.normal) <= node.intercept:
            return self._traverse_path(x, node.left, current_depth + 1)
        else:
            return self._traverse_path(x, node.right, current_depth + 1)

    def decision_function(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Computes anomaly score s(x) in [0, 1]. Higher score indicates greater anomaly.
            s(x) = 2^(- E[h(x)] / c(n))
        """
        if not self.trees_:
            raise ValueError("ExtendedIsolationForestDetector must be fitted before calling decision_function.")

        X_arr = np.asarray(X, dtype=np.float64)
        n_samples = len(X_arr)
        total_paths = np.zeros(n_samples, dtype=np.float64)

        for tree in self.trees_:
            for i in range(n_samples):
                total_paths[i] += self._traverse_path(X_arr[i], tree, current_depth=0)

        mean_paths = total_paths / len(self.trees_)
        c = self.c_subsample_ if self.c_subsample_ > 0 else 1.0

        # Anomaly score formula: s(x) = 2^(- mean_path / c)
        anomaly_scores = 2.0 ** (-mean_paths / c)
        return anomaly_scores

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Flags sample as anomalous (1) if anomaly score >= threshold."""
        scores = self.decision_function(X)
        return (scores >= self.threshold_).astype(int)
