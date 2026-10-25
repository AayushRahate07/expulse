"""
Unit Tests for Ground-Truth Attribution Validation (Top-K Hit Rate & MRR).
"""

from src.explainability.validation import (
    TEP_GROUND_TRUTH_ROOT_CAUSES,
    get_ground_truth_variables,
    compute_hit_at_k,
    compute_mrr,
    evaluate_attribution_ranking,
    compare_attribution_methods_wilcoxon
)


def test_ground_truth_mapping_and_hit_rate():
    """Verifies ground truth retrieval, H@K calculation, and MRR metrics."""
    # Test fault 4: primary includes xmv_10, xmeas_9, xmeas_21
    gt_vars = get_ground_truth_variables(4, include_secondary=True)
    assert "xmv_10" in gt_vars
    assert "xmeas_9" in gt_vars

    # Perfect ranking test: true causes ranked 1st and 2nd
    ranked_perfect = ["xmv_10", "xmeas_9", "xmeas_31", "xmeas_1", "xmeas_2"]
    h1 = compute_hit_at_k(ranked_perfect, gt_vars, k=1)
    h3 = compute_hit_at_k(ranked_perfect, gt_vars, k=3)
    mrr = compute_mrr(ranked_perfect, gt_vars)

    assert h1 == 1.0  # 1 hit out of min(1, 5) = 1.0
    assert h3 == 2 / 3  # 2 hits out of min(3, 5) = 0.667
    assert mrr == 1.0  # First cause at rank 1

    # Displaced ranking: true cause appears at rank 3
    ranked_delayed = ["xmeas_31", "xmeas_1", "xmv_10", "xmeas_2"]
    assert compute_mrr(ranked_delayed, gt_vars) == 1.0 / 3.0

    eval_dict = evaluate_attribution_ranking(ranked_perfect, fault_id=4)
    assert "H@1" in eval_dict
    assert "H@3" in eval_dict
    assert "MRR" in eval_dict


def test_wilcoxon_paired_test():
    """Verifies paired Wilcoxon test on H@3 across runs."""
    method_a = [1.0, 0.67, 0.67, 1.0, 0.67] * 4
    method_b = [0.0, 0.33, 0.0, 0.33, 0.0] * 4

    res = compare_attribution_methods_wilcoxon(method_a, method_b)
    assert res["p_value"] < 0.05
    assert res["superior_method"] == "Method A"
