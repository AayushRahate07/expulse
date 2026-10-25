"""
Unit Tests for Multi-Model Consensus Advisory Protocol.
"""

import numpy as np
from src.models.consensus import ConsensusArbitrator


def test_consensus_arbitrator_classification_rules():
    """Tests classification of individual samples into Levels 0, 1, 2, 3."""
    arbitrator = ConsensusArbitrator()

    # Level 3: All 3 agree
    assert arbitrator.classify_sample(flag_rule=1, flag_pca=1, flag_if=1) == 3
    # Level 3: Rule + PCA agree
    assert arbitrator.classify_sample(flag_rule=1, flag_pca=1, flag_if=0) == 3

    # Level 2: PCA active, Rule inactive (subspace anomaly)
    assert arbitrator.classify_sample(flag_rule=0, flag_pca=1, flag_if=0) == 2
    assert arbitrator.classify_sample(flag_rule=0, flag_pca=1, flag_if=1) == 2

    # Level 1: Rule active, PCA inactive (single sensor drift)
    assert arbitrator.classify_sample(flag_rule=1, flag_pca=0, flag_if=0) == 1

    # Level 0: All inactive
    assert arbitrator.classify_sample(flag_rule=0, flag_pca=0, flag_if=0) == 0


def test_consensus_arbitrator_evaluate_run():
    """Verifies run-level evaluation, counts, proportions, and dominant alarm level."""
    arbitrator = ConsensusArbitrator()

    n = 200
    preds_rule = np.zeros(n, dtype=int)
    preds_pca = np.zeros(n, dtype=int)
    preds_if = np.zeros(n, dtype=int)

    # From sample 160 onwards, activate Level 3 emergency alarm
    preds_rule[160:] = 1
    preds_pca[160:] = 1
    preds_if[160:] = 1

    run_summary = arbitrator.evaluate_run(preds_rule, preds_pca, preds_if)

    assert len(run_summary["levels"]) == n
    assert run_summary["dominant_level"] == 3
    assert "Critical Plant Alarm" in run_summary["dominant_label"]
    assert run_summary["counts"][3] == 40
    assert run_summary["counts"][0] == 160
