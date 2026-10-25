"""
Multi-Model Consensus Advisory Protocol for Tennessee Eastman Process.
Arbitrates between Rule-Based, PCA Subspace, and Isolation Forest detectors
into a tri-level operational plant alarm hierarchy.
"""

from typing import Dict, Any, List, Union, Tuple
import numpy as np


class ConsensusArbitrator:
    """
    Inference arbitrator mapping simultaneous multi-model detector flags
    into an actionable 3-tier plant alarm hierarchy:
    
    Level 3: Critical Plant Alarm (All models agree / Rule + PCA + IF)
             -> Catastrophic multi-unit disturbance; triggers full plant emergency trip.
    Level 2: Joint Subspace Anomaly (PCA active, Rule inactive)
             -> Subtle cross-unit correlation breakdown; triggers root-cause attribution loop.
    Level 1: Single Sensor Advisory (Rule active, PCA inactive)
             -> Probable instrument drift / transmitter failure; dispatches maintenance ticket.
    Level 0: Nominal (No sustained flags).
    """

    LEVEL_LABELS = {
        0: "Nominal (Normal Operation)",
        1: "Level 1: Single-Sensor Instrument Advisory",
        2: "Level 2: Joint Subspace Anomaly (Root-Cause Inspection)",
        3: "Level 3: Critical Plant Alarm (Emergency Trip Candidate)",
    }

    LEVEL_COLORS = {
        0: "#2ca02c",  # Green
        1: "#ff7f0e",  # Orange
        2: "#1f77b4",  # Blue
        3: "#d62728",  # Red
    }

    def __init__(self) -> None:
        pass

    @staticmethod
    def classify_sample(flag_rule: int, flag_pca: int, flag_if: int) -> int:
        """
        Classifies an individual sample into alarm level 0, 1, 2, or 3.
        """
        r, p, i = int(flag_rule), int(flag_pca), int(flag_if)

        # Level 3: All 3 agree or hard multi-model failure
        if (r == 1 and p == 1 and i == 1) or (r == 1 and p == 1):
            return 3

        # Level 2: Subspace / Multivariate anomaly without hard single-sensor breach
        if p == 1 and r == 0:
            return 2
        if i == 1 and r == 0:
            return 2

        # Level 1: Hard univariate breach without PCA subspace anomaly
        if r == 1 and p == 0:
            return 1

        return 0

    def evaluate_run(
        self,
        preds_rule: np.ndarray,
        preds_pca: np.ndarray,
        preds_if: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluates a complete simulation run across all 3 detector outputs.
        Returns:
            levels: Array of alarm levels (0, 1, 2, 3) for each sample.
            summary: Counts and percentages per alarm level.
            dominant_level: Maximum sustained severity observed.
        """
        preds_r = np.asarray(preds_rule, dtype=int)
        preds_p = np.asarray(preds_pca, dtype=int)
        preds_i = np.asarray(preds_if, dtype=int)

        n_samples = len(preds_r)
        levels = np.zeros(n_samples, dtype=int)

        for t in range(n_samples):
            levels[t] = self.classify_sample(preds_r[t], preds_p[t], preds_i[t])

        counts = {
            lvl: int(np.sum(levels == lvl))
            for lvl in [0, 1, 2, 3]
        }
        proportions = {
            lvl: float(counts[lvl] / n_samples)
            for lvl in [0, 1, 2, 3]
        }

        # Check post-fault dominant alarm state (after sample 160)
        post_fault_levels = levels[160:] if len(levels) > 160 else levels
        dominant = int(np.max(post_fault_levels)) if len(post_fault_levels) > 0 else 0

        return {
            "levels": levels,
            "counts": counts,
            "proportions": proportions,
            "dominant_level": dominant,
            "dominant_label": self.LEVEL_LABELS[dominant],
            "dominant_color": self.LEVEL_COLORS[dominant],
        }
