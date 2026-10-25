"""
Page's Cumulative Sum (CUSUM) Sequential Change-Point Detection Module.
Provides optimal sequential detection under Lorden's criterion (Lorden, 1971).
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np


class CUSUMDetector:
    """
    Two-sided or one-sided tabular CUSUM change-point detector.
    
    Accumulates positive deviations from nominal baseline:
        C_0 = 0
        C_t = max(0, C_{t-1} + (s_t - mu_0) - k)
        
    where:
        mu_0: Empirical baseline score on healthy data.
        sigma_0: Standard deviation of score on healthy data.
        k: Reference parameter (slack), typically 0.5 * sigma_0.
        h: Decision threshold (calibrated for desired ARL_0).
    """

    def __init__(
        self,
        k: Optional[float] = None,
        h: Optional[float] = None,
        k_factor: float = 0.5,
        h_factor: float = 5.0,
    ) -> None:
        self.k = k
        self.h = h
        self.k_factor = k_factor
        self.h_factor = h_factor
        self.mu_0_: float = 0.0
        self.sigma_0_: float = 1.0
        self.k_: float = 0.5
        self.h_: float = 5.0

    def fit(self, healthy_scores: np.ndarray, target_arl0: float = 2000.0) -> "CUSUMDetector":
        """
        Calibrates mu_0, sigma_0, slack k, and decision threshold h on nominal calibration data.
        
        Args:
            healthy_scores: 1D array of anomaly scores from nominal operation.
            target_arl0: Desired in-control Average Run Length (samples between false alarms).
        """
        scores = np.asarray(healthy_scores, dtype=np.float64)
        self.mu_0_ = float(np.mean(scores))
        self.sigma_0_ = float(np.std(scores, ddof=1))
        if self.sigma_0_ == 0.0:
            self.sigma_0_ = 1e-6

        # Set reference slack k
        if self.k is not None:
            self.k_ = float(self.k)
        else:
            self.k_ = float(self.k_factor * self.sigma_0_)

        # Calibrate decision threshold h to achieve target ARL0
        if self.h is not None:
            self.h_ = float(self.h)
        else:
            self.h_ = self._calibrate_threshold(scores, target_arl0=target_arl0)

        return self

    def _calibrate_threshold(self, scores: np.ndarray, target_arl0: float = 2000.0) -> float:
        """
        Grid search on calibration scores to find minimum threshold h ensuring ARL0 >= target_arl0.
        """
        h_candidates = np.linspace(1.0 * self.sigma_0_, 12.0 * self.sigma_0_, num=45)
        best_h = float(h_candidates[-1])

        for h_val in h_candidates:
            # Simulate CUSUM on calibration scores
            cusum_stat = 0.0
            false_alarms = 0
            n_samples = len(scores)

            for s in scores:
                cusum_stat = max(0.0, cusum_stat + (s - self.mu_0_) - self.k_)
                if cusum_stat >= h_val:
                    false_alarms += 1
                    cusum_stat = 0.0  # Reset after alarm

            empirical_arl = n_samples / max(1, false_alarms)
            if empirical_arl >= target_arl0:
                best_h = float(h_val)
                break

        return best_h

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """
        Computes running CUSUM statistic vector C_t across sequence of anomaly scores.
        """
        scores_arr = np.asarray(scores, dtype=np.float64)
        c_stat = np.zeros(len(scores_arr), dtype=np.float64)
        running_c = 0.0

        for t, s in enumerate(scores_arr):
            running_c = max(0.0, running_c + (s - self.mu_0_) - self.k_)
            c_stat[t] = running_c

        return c_stat

    def detect(self, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns running CUSUM statistic and binary change-point alarms (1 if C_t >= h_).
        """
        c_stat = self.transform(scores)
        alarms = (c_stat >= self.h_).astype(int)
        return c_stat, alarms

    def compute_first_detection(self, scores: np.ndarray) -> Optional[int]:
        """
        Finds 0-indexed sample of first change-point alarm C_t >= h_.
        """
        c_stat = self.transform(scores)
        crossings = np.where(c_stat >= self.h_)[0]
        return int(crossings[0]) if len(crossings) > 0 else None


def compute_cusum_lead_time(
    scores: np.ndarray,
    cusum_detector: CUSUMDetector,
    fault_intro_sample: int = 160,
    sampling_interval_minutes: int = 3,
) -> Dict[str, Any]:
    """
    Evaluates detection sample and lead time (delay) using calibrated CUSUM.
    """
    first_idx = cusum_detector.compute_first_detection(scores)
    if first_idx is None:
        return {
            "first_detection_sample": None,
            "lead_time_samples": None,
            "lead_time_minutes": None,
            "detected": False,
        }

    detected_sample = first_idx + 1  # 1-indexed
    lead_time_samples = detected_sample - fault_intro_sample
    lead_time_minutes = lead_time_samples * sampling_interval_minutes

    return {
        "first_detection_sample": detected_sample,
        "lead_time_samples": lead_time_samples,
        "lead_time_minutes": lead_time_minutes,
        "detected": True,
    }
