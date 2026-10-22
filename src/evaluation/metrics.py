"""
Evaluation Metrics: Precision, Recall, F1, False Alarm Rate, and 3-Minute Sampling Lead Time Analysis.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from src.config import SAMPLING_INTERVAL_MINUTES


def compute_binary_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes Precision, Recall, F1-Score, and False Alarm Rate (FAR).
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    precision = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    recall = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()
    far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": far,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def compute_first_sustained_detection(
    predictions: np.ndarray,
    n_consecutive: int = 3
) -> Optional[int]:
    """
    Finds the 0-indexed sample position of the first sequence of N consecutive anomaly flags (1s).
    
    Args:
        predictions: Binary prediction array (0 or 1).
        n_consecutive: Number of consecutive positive samples required for a sustained flag.
        
    Returns:
        Index of the first sample in the sustained sequence, or None if no sustained detection occurs.
    """
    if len(predictions) < n_consecutive:
        return None

    # Sliding window check for N consecutive ones
    for i in range(len(predictions) - n_consecutive + 1):
        if np.all(predictions[i : i + n_consecutive] == 1):
            return i

    return None


def compute_lead_time_metrics(
    predictions: np.ndarray,
    fault_intro_sample: int = 160,
    sampling_interval_minutes: int = SAMPLING_INTERVAL_MINUTES,
    n_consecutive: int = 3
) -> Dict[str, Optional[float]]:
    """
    Computes Early Warning Lead Time taking into account TEP's 3-minute sampling frequency (Refinement 3).
    
    Args:
        predictions: Binary prediction array for a single simulation run.
        fault_intro_sample: 1-indexed or 0-indexed sample index where the fault was introduced.
        sampling_interval_minutes: Minutes per sample (default 3 mins).
        n_consecutive: Number of consecutive flags required for sustained detection.
        
    Returns:
        Dictionary containing:
            - first_sustained_sample: Sample index of sustained detection
            - lead_time_samples: Sample difference (Detection Sample - Fault Intro Sample)
            - lead_time_minutes: Sample difference converted to minutes (samples * 3)
            - detection_delay_minutes: Delay in minutes after fault introduction (0 if detected immediately or early)
            - detected: Boolean indicating whether a sustained flag occurred
    """
    first_sustained_idx = compute_first_sustained_detection(predictions, n_consecutive=n_consecutive)

    if first_sustained_idx is None:
        return {
            "first_sustained_sample": None,
            "lead_time_samples": None,
            "lead_time_minutes": None,
            "detection_delay_minutes": None,
            "detected": False
        }

    # 1-indexed sample count aligned with dataset 'sample' column
    detected_sample = first_sustained_idx + 1
    lead_time_samples = detected_sample - fault_intro_sample
    lead_time_minutes = lead_time_samples * sampling_interval_minutes
    detection_delay_minutes = max(0, lead_time_minutes)

    return {
        "first_sustained_sample": detected_sample,
        "lead_time_samples": lead_time_samples,
        "lead_time_minutes": lead_time_minutes,
        "detection_delay_minutes": detection_delay_minutes,
        "detected": True
    }


def evaluate_run_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fault_intro_sample: int = 160
) -> Dict[str, Any]:
    """Combines binary classification metrics and lead time analysis for a simulation run."""
    class_metrics = compute_binary_classification_metrics(y_true, y_pred)
    lead_metrics = compute_lead_time_metrics(y_pred, fault_intro_sample=fault_intro_sample)
    return {**class_metrics, **lead_metrics}
