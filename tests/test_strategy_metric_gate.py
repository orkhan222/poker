from __future__ import annotations

from copy import deepcopy

from poker_agent.strategy_metric_gate import evaluate_strategy_metric_gate


def _passing_metric_families() -> dict:
    return {
        "action_classification": {
            "required": True,
            "accuracy_only_approval_allowed": False,
            "metrics": {
                "accuracy": 0.72,
                "macro_f1": 0.56,
                "balanced_accuracy": 0.55,
            },
        },
        "calibration": {
            "required": True,
            "cross_entropy_only_approval_allowed": False,
            "diagnostic_loss_only_approval_allowed": False,
            "metrics": {
                "ece_10": 0.06,
                "cross_entropy": 0.42,
            },
        },
        "action_distribution": {
            "required": True,
            "larger_clean_real_gameplay_revalidation_required": False,
            "metrics": {"js_divergence": 0.02},
        },
        "bet_sizing": {
            "required": True,
            "final_high_realism_claim_allowed": True,
            "metrics": {"bet_size_mae": 0.18},
        },
        "simulation_return": {
            "required": True,
            "metrics": {
                "win_rate": 0.55,
                "expected_value_delta_vs_baseline": 0.21,
            },
        },
        "seed_stability": {
            "required": True,
            "metrics": {
                "full_training_seed_stability_required": True,
                "phase3_seed_stability_evaluated": True,
            },
            "full_production_scale_training_status": "COMPLETED",
            "phase3_training_proof_status": "TRAINING_PROOF_COMPLETED",
        },
    }


def test_strategy_metric_gate_allows_claim_only_for_complete_metric_bundle() -> None:
    gate = evaluate_strategy_metric_gate(_passing_metric_families())

    assert gate["gate_status"] == "PASS"
    assert gate["final_metric_bundle_passed"] is True
    assert gate["final_strategy_quality_claim_allowed"] is True
    assert gate["missing_or_failed_requirements"] == []


def test_strategy_metric_gate_blocks_accuracy_and_cross_entropy_shortcut() -> None:
    families = deepcopy(_passing_metric_families())
    families["calibration"]["cross_entropy_only_approval_allowed"] = True
    families["calibration"]["diagnostic_loss_only_approval_allowed"] = True

    gate = evaluate_strategy_metric_gate(families)

    assert gate["gate_status"] == "BLOCKED"
    assert gate["final_metric_bundle_passed"] is False
    assert gate["final_strategy_quality_claim_allowed"] is False
    assert "cross_entropy_only_not_allowed" in gate["missing_or_failed_requirements"]
    assert "diagnostic_loss_only_not_allowed" in gate["missing_or_failed_requirements"]
    assert gate["blocked_approval_shortcuts"]["accuracy_plus_cross_entropy"] is True
