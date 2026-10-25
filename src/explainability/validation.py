"""
Ground-Truth Attribution Validation Module for Tennessee Eastman Process.
Maps feature rankings against Downs & Vogel (1993) plant physical equations.
Includes Functional Process Group Attribution to eliminate collinearity credit dilution.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats

# Physical ground-truth mapping from Downs & Vogel (1993) and Rieth et al. (2017)
TEP_GROUND_TRUTH_ROOT_CAUSES = {
    1: {
        "primary": ["xmeas_1", "xmv_1"],  # A feed stream flow & valve
        "secondary": ["xmeas_4", "xmeas_6", "xmv_3"],  # Total feed flow, reactor feed, D feed valve
    },
    4: {
        "primary": ["xmv_10", "xmeas_9", "xmeas_21"],  # Cooling valve, reactor temp, cooling water outlet temp
        "secondary": ["xmeas_11", "xmeas_22"],  # Separator temp, separator cooling outlet temp
    },
    6: {
        "primary": ["xmeas_1", "xmv_1"],  # A feed flow rate & valve position
        "secondary": ["xmeas_6", "xmeas_7"],  # Total reactor feed & reactor pressure
    },
    8: {
        "primary": [
            "xmeas_23", "xmeas_24", "xmeas_25",
            "xmeas_26", "xmeas_27", "xmeas_28"
        ],  # Stream 6 & 9 analyzer components A-F
        "secondary": ["xmeas_1", "xmeas_2", "xmeas_3", "xmv_1", "xmv_2"],
    },
}

# Functional Process Groups (Addresses Collinear Credit-Sharing)
TEP_FUNCTIONAL_PROCESS_GROUPS = {
    "Feed_Reactant_Loop": {
        "sensors": {"xmeas_1", "xmeas_2", "xmeas_3", "xmeas_4", "xmv_1", "xmv_2", "xmv_3", "xmv_4"},
        "faults": {1, 6, 8},
        "description": "Feed streams 1-4 flows and reactant proportioning valves"
    },
    "Reactor_Thermal_Cooling_Loop": {
        "sensors": {"xmeas_9", "xmeas_21", "xmeas_22", "xmv_10", "xmv_11"},
        "faults": {4},
        "description": "Reactor temperature, jacket cooling water, condenser cooling"
    },
    "Reactor_Pressure_Recycle_Loop": {
        "sensors": {"xmeas_5", "xmeas_6", "xmeas_7", "xmeas_20", "xmv_6"},
        "faults": {6, 1},
        "description": "Reactor pressure, feed rate, recycle stream 8, compressor work, purge vent"
    },
    "Separation_Stripping_Unit": {
        "sensors": {
            "xmeas_11", "xmeas_12", "xmeas_13", "xmeas_14", "xmeas_15",
            "xmeas_16", "xmeas_17", "xmeas_18", "xmeas_19", "xmv_7", "xmv_8"
        },
        "faults": set(),
        "description": "Separator & stripper temps, levels, pressures, bottoms flow, steam"
    },
    "Stream_Composition_Analyzers": {
        "sensors": {f"xmeas_{i}" for i in range(23, 39)},
        "faults": {8},
        "description": "Online chromatograph analyzers for feed, purge, and product streams"
    },
}


def get_ground_truth_variables(fault_id: int, include_secondary: bool = True) -> Set[str]:
    """Returns set of known physical process root cause sensors for a given fault mode."""
    if fault_id not in TEP_GROUND_TRUTH_ROOT_CAUSES:
        return set()
    mapping = TEP_GROUND_TRUTH_ROOT_CAUSES[fault_id]
    gt = set(mapping["primary"])
    if include_secondary:
        gt.update(mapping.get("secondary", []))
    return gt


def compute_hit_at_k(
    ranked_features: List[str],
    ground_truth: Set[str],
    k: int = 3
) -> float:
    """
    Computes Top-K Hit Rate (H@K):
        H@K = |Top-K(a) ∩ G_fault| / min(K, |G_fault|)
    """
    if not ground_truth or k <= 0:
        return 0.0
    top_k = set(ranked_features[:k])
    hits = len(top_k.intersection(ground_truth))
    denominator = min(k, len(ground_truth))
    return float(hits / denominator) if denominator > 0 else 0.0


def compute_mrr(
    ranked_features: List[str],
    ground_truth: Set[str]
) -> float:
    """
    Computes Mean Reciprocal Rank (MRR):
        MRR = 1 / min_{v in G_fault} rank(v)
    1.0 means the first ranked feature is a physical root cause.
    """
    if not ground_truth:
        return 0.0
    for rank_idx, feat in enumerate(ranked_features, start=1):
        if feat in ground_truth:
            return 1.0 / rank_idx
    return 0.0


def compute_grouped_process_attribution(
    feature_importance_dict: Dict[str, float]
) -> List[Tuple[str, float]]:
    """
    Aggregates individual sensor attributions into functional process groups.
    Eliminates credit dilution across collinear manifolds (e.g. Stream 1, 4, and feed valve).
    
    Returns:
        List of (group_name, total_attribution) sorted in descending order.
    """
    group_scores = {group_name: 0.0 for group_name in TEP_FUNCTIONAL_PROCESS_GROUPS}

    for feat, score in feature_importance_dict.items():
        matched = False
        for group_name, group_meta in TEP_FUNCTIONAL_PROCESS_GROUPS.items():
            if feat in group_meta["sensors"]:
                group_scores[group_name] += float(score)
                matched = True
                break

    # Sort descending
    sorted_groups = sorted(group_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_groups


def evaluate_attribution_ranking(
    ranked_features: List[str],
    fault_id: int,
    k_list: Tuple[int, ...] = (1, 3, 5),
    include_secondary: bool = True
) -> Dict[str, float]:
    """
    Evaluates a ranked feature list against physical root causes.
    """
    gt = get_ground_truth_variables(fault_id, include_secondary=include_secondary)
    metrics = {
        f"H@{k}": compute_hit_at_k(ranked_features, gt, k=k)
        for k in k_list
    }
    metrics["MRR"] = compute_mrr(ranked_features, gt)
    return metrics


def compare_attribution_methods_wilcoxon(
    h3_method_a: List[float],
    h3_method_b: List[float]
) -> Dict[str, Any]:
    """
    Conducts paired Wilcoxon signed-rank test on H@3 across runs.
    """
    diffs = np.array(h3_method_a) - np.array(h3_method_b)
    if np.all(diffs == 0.0):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "superior_method": "Tie",
            "mean_diff": 0.0
        }

    try:
        res = stats.wilcoxon(diffs, zero_method="pratt")
        stat = float(res.statistic)
        p_val = float(res.pvalue)
    except Exception:
        stat, p_val = 0.0, 1.0

    mean_diff = float(np.mean(diffs))
    if p_val < 0.05:
        superior = "Method A" if mean_diff > 0 else "Method B"
    else:
        superior = "Inconclusive (p >= 0.05)"

    return {
        "statistic": stat,
        "p_value": p_val,
        "superior_method": superior,
        "mean_diff": mean_diff
    }
