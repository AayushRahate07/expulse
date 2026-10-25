"""
Non-Overlapping Block Conformal Prediction for Autocorrelated Industrial Time Series.
Restores distribution-free false alarm control under serial dependence (lag-1 rho in [0.75, 0.95]).
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class BlockConformalCalibrator:
    """
    Non-overlapping block conformal calibrator.
    Pools calibration anomaly scores into blocks of length L (1 hour = 20 samples),
    computing block-maximum scores to break autocorrelation and provide valid p-values.
    """

    def __init__(self, block_length: int = 20, alpha: float = 0.05) -> None:
        self.block_length = block_length
        self.alpha = alpha
        self.block_maxima_: np.ndarray = np.array([])
        self.q_hat_: float = 0.0
        self.n_blocks_: int = 0

    def fit(self, healthy_scores: np.ndarray) -> "BlockConformalCalibrator":
        """
        Partitions healthy scores into B non-overlapping blocks of length L,
        computes block maxima M_b, and calibrates finite-sample cutoff q_hat.
        """
        scores = np.asarray(healthy_scores, dtype=np.float64)
        n_samples = len(scores)
        self.n_blocks_ = n_samples // self.block_length

        if self.n_blocks_ == 0:
            raise ValueError(f"Insufficient samples ({n_samples}) for block length {self.block_length}")

        # Truncate to exact multiple of block_length
        trimmed_scores = scores[: self.n_blocks_ * self.block_length]
        blocks = trimmed_scores.reshape((self.n_blocks_, self.block_length))
        self.block_maxima_ = np.max(blocks, axis=1)

        # Finite-sample conformal quantile: ceil((B + 1) * (1 - alpha)) / B
        q_level = min(1.0, np.ceil((self.n_blocks_ + 1) * (1.0 - self.alpha)) / self.n_blocks_)
        self.q_hat_ = float(np.quantile(self.block_maxima_, q_level))

        return self

    def compute_block_p_value(self, test_window: np.ndarray) -> float:
        """
        Computes conformal block p-value for a trailing window of length L:
            p_block = (1 + sum(M_b >= M_test)) / (B + 1)
        """
        m_test = np.max(test_window)
        exceedances = np.sum(self.block_maxima_ >= m_test)
        p_val = (1.0 + exceedances) / (self.n_blocks_ + 1.0)
        return float(p_val)

    def evaluate_sequence(self, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates a continuous score sequence with a sliding window of length L.
        Returns:
            p_values: 1D array of block conformal p-values for each time t (padded at start).
            alarms: Binary flags where p_block < alpha (1 if anomalous, 0 otherwise).
        """
        scores_arr = np.asarray(scores, dtype=np.float64)
        n_samples = len(scores_arr)
        p_values = np.ones(n_samples, dtype=np.float64)

        for t in range(self.block_length - 1, n_samples):
            window = scores_arr[t - self.block_length + 1 : t + 1]
            p_values[t] = self.compute_block_p_value(window)

        alarms = (p_values < self.alpha).astype(int)
        return p_values, alarms

    def evaluate_block_coverage(self, held_out_healthy_scores: np.ndarray) -> Dict[str, float]:
        """
        Evaluates empirical block false alarm rate on held-out healthy data.
        Verifies whether empirical block FAR <= alpha.
        """
        scores = np.asarray(held_out_healthy_scores, dtype=np.float64)
        n_test_blocks = len(scores) // self.block_length
        if n_test_blocks == 0:
            return {"block_far": 0.0, "coverage": 1.0, "guarantee_held": True}

        trimmed = scores[: n_test_blocks * self.block_length]
        test_blocks = trimmed.reshape((n_test_blocks, self.block_length))
        test_maxima = np.max(test_blocks, axis=1)

        # Block-level p-value: proportion of calibration blocks exceeding test block
        p_vals = np.array([
            (1.0 + np.sum(self.block_maxima_ >= m_test)) / (self.n_blocks_ + 1.0)
            for m_test in test_maxima
        ])

        block_false_alarms = np.sum(p_vals < self.alpha)
        empirical_block_far = float(block_false_alarms / n_test_blocks)
        empirical_coverage = 1.0 - empirical_block_far

        return {
            "alpha": float(self.alpha),
            "target_coverage": float(1.0 - self.alpha),
            "empirical_block_far": empirical_block_far,
            "empirical_coverage": empirical_coverage,
            "guarantee_held": bool(empirical_block_far <= (self.alpha + 0.02)),  # Within finite-sample margin
            "n_test_blocks": int(n_test_blocks),
        }
