"""
Unit Tests for Ground-Truth Attribution Validation (Top-K Hit Rate, MRR, and Grouped Attribution).
"""

from src.explainability.validation import (
    TEP_GROUND_TRUTH_ROOT_CAUSES,
    TEP_FUNCTIONAL_PROCESS_GROUPS,
    get_ground_truth_variables,
    compute_hit_at_k,
    compute_mrr,
    evaluate_attribution_ranking,
    compare_attribution_methods_wilcoxon,
    compute_grouped_process_attribution,
)


def test_ground_truth_mapping_and_hit_rate():
    """Verifies ground truth retrieval, H@K calculation, and MRR metrics."""
    gt_vars = get_ground_truth_variables(4, include_secondary=True)
    assert "xmv_10" in gt_vars
    assert "xmeas_9" in gt_vars

    ranked_perfect = ["xmv_10", "xmeas_9", "xmeas_31", "xmeas_1", "xmeas_2"]
    h1 = compute_hit_at_k(ranked_perfect, gt_vars, k=1)
    h3 = compute_hit_at_k(ranked_perfect, gt_vars, k=3)
    mrr = compute_mrr(ranked_perfect, gt_vars)

    assert h1 == 1.0
    assert h3 == 2 / 3
    assert mrr == 1.0

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


def test_grouped_process_attribution_resolves_dilution():
    """
    Verifies that collinear credit-sharing across individual feed tags
    (e.g., xmeas_1, xmeas_4, xmv_1) correctly aggregates into the Feed_Reactant_Loop.
    """
    # Credit diluted across collinear feed sensors
    diluted_scores = {
        "xmeas_1": 0.25,
        "xmeas_4": 0.20,
        "xmv_1": 0.25,
        "xmeas_9": 0.15,
        "xmeas_16": 0.15,
    }

    grouped = compute_grouped_process_attribution(diluted_scores)
    top_group, top_score = grouped[0]

    assert top_group == "Feed_Reactant_Loop"
    # Combined score: 0.25 + 0.20 + 0.25 = 0.70
    assert abs(top_score - 0.70) < 1e-5
