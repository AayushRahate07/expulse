"""
Evaluation Metrics: Precision, Recall, F1, False Alarm Rate, and 3-Minute Sampling Lead Time Analysis.
Addresses the pre-fault lead-time trap by strictly separating pre-fault false alarms from post-fault detection delay.
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
) -> Dict[str, Any]:
    """
    Computes Detection Delay (ADD) and Pre-Fault False Alarm metrics.
    
    Eliminates the pre-fault lead-time trap where false alarms prior to sample 160 
    are falsely credited as "early alerts". Evaluates strictly:
    1. Pre-fault window (t < fault_intro_sample): checks for false alarms and contamination.
    2. Post-fault window (t >= fault_intro_sample): measures true detection delay.
    
    Args:
        predictions: Binary prediction array for a single simulation run.
        fault_intro_sample: 1-indexed sample index where the fault was introduced (default 160).
        sampling_interval_minutes: Minutes per sample (default 3 mins).
        n_consecutive: Number of consecutive flags required for sustained detection.
        
    Returns:
        Dictionary containing:
            - first_sustained_sample: Sample index of sustained detection post-fault
            - lead_time_samples: Post-fault detection delay in samples (>= 0)
            - lead_time_minutes: Post-fault detection delay in minutes (>= 0)
            - detection_delay_minutes: Synonymous with lead_time_minutes
            - detected: Boolean indicating whether a sustained post-fault flag occurred
            - has_pre_fault_alarm: Boolean indicating whether a false alarm occurred before fault intro
            - time_to_first_false_alarm_sample: Sample index of first sustained pre-fault false alarm, or None
            - pre_fault_contaminated: Boolean indicating whether model was in active alarm state at fault onset
    """
    preds_arr = np.asarray(predictions, dtype=int)
    fault_idx = fault_intro_sample - 1  # 0-indexed fault intro position

    # 1. Pre-fault False Alarm Analysis (t < fault_intro_sample)
    pre_fault_preds = preds_arr[:fault_idx]
    pre_fault_sustained_idx = compute_first_sustained_detection(pre_fault_preds, n_consecutive=n_consecutive)
    has_pre_fault_alarm = pre_fault_sustained_idx is not None
    time_to_first_fa = (pre_fault_sustained_idx + 1) if has_pre_fault_alarm else None

    # Check if detector was contaminated across the transition boundary
    pre_fault_contaminated = False
    if fault_idx >= n_consecutive:
        pre_fault_contaminated = bool(np.all(preds_arr[fault_idx - n_consecutive : fault_idx] == 1))

    # 2. Post-fault Detection Delay (t >= fault_intro_sample)
    post_fault_preds = preds_arr[fault_idx:]
    post_fault_sustained_idx = compute_first_sustained_detection(post_fault_preds, n_consecutive=n_consecutive)

    if post_fault_sustained_idx is None:
        return {
            "first_sustained_sample": None,
            "lead_time_samples": None,
            "lead_time_minutes": None,
            "detection_delay_minutes": None,
            "detected": False,
            "has_pre_fault_alarm": has_pre_fault_alarm,
            "time_to_first_false_alarm_sample": time_to_first_fa,
            "pre_fault_contaminated": pre_fault_contaminated,
        }

    # 1-indexed detection sample at or after fault injection
    detected_sample = fault_intro_sample + post_fault_sustained_idx
    detection_delay_samples = post_fault_sustained_idx
    detection_delay_minutes = detection_delay_samples * sampling_interval_minutes

    return {
        "first_sustained_sample": detected_sample,
        "lead_time_samples": detection_delay_samples,
        "lead_time_minutes": detection_delay_minutes,
        "detection_delay_minutes": detection_delay_minutes,
        "detected": True,
        "has_pre_fault_alarm": has_pre_fault_alarm,
        "time_to_first_false_alarm_sample": time_to_first_fa,
        "pre_fault_contaminated": pre_fault_contaminated,
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
