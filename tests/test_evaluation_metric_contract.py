from __future__ import annotations

import json
from pathlib import Path

from poker_agent.evaluation_metric_contract import (
    build_evaluation_metric_contract,
    validate_evaluation_metric_contract,
)


def _write_reports(reports: Path) -> None:
    reports.mkdir()
    (reports / "today_acceptance_production_gate.json").write_text(
        json.dumps(
            {
                "valid_metrics": {
                    "accuracy": 0.72,
                    "macro_f1": 0.49,
                    "balanced_accuracy": 0.51,
                    "ece_10": 0.18,
                    "brier_loss": 0.45,
                    "cross_entropy": 0.84,
                }
            }
        ),
        encoding="utf-8",
    )
    (reports / "challenger_strategy_quality.json").write_text(
        json.dumps(
            {
                "challenger_result": {
                    "accuracy": 0.70,
                    "macro_f1": 0.48,
                    "balanced_accuracy": 0.50,
                    "calibration_ece_10": 0.16,
                }
            }
        ),
        encoding="utf-8",
    )
    (reports / "policy_acceptance.json").write_text(
        json.dumps(
            {
                "human_action_alignment": {"accuracy": 0.64, "macro_f1": 0.508},
                "human_likeness": {
                    "status": "PASS",
                    "timing_and_bet_size_status": "PASS",
                    "js_divergence": 0.0026,
                },
            }
        ),
        encoding="utf-8",
    )
    (reports / "behavioral_revalidation.json").write_text(
        json.dumps(
            {
                "current_validation_scope": {"action_distribution_status": "PASS", "js_divergence": 0.0026},
                "revalidation_boundary": {
                    "larger_clean_real_gameplay_revalidation_required": True,
                    "generalized_action_distribution_claim_allowed": False,
                },
            }
        ),
        encoding="utf-8",
    )
    (reports / "bet_timing_calibration.json").write_text(
        json.dumps(
            {
                "current_delivery_scope": {
                    "api_response_fields": ["bet_size", "wait_time_ms", "sizing_method", "timing_method"],
                    "timing_and_bet_size_status": "PASS",
                },
                "calibration_boundary": {
                    "requires_more_real_player_behavior_labels": True,
                    "final_high_realism_claim_allowed": False,
                },
            }
        ),
        encoding="utf-8",
    )
    (reports / "production_self_play.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "production_scale_status": "PASS",
                "paired_hands": 5000,
                "run_count": 20,
                "mean_policy_win_rate": 0.577,
                "min_policy_win_rate": 0.544,
                "max_policy_win_rate": 0.608,
                "mean_ev_delta_vs_baseline": 0.82,
            }
        ),
        encoding="utf-8",
    )
    (reports / "multi_agent_training_status.json").write_text(
        json.dumps(
            {
                "training_boundary": {
                    "full_production_scale_multi_agent_training_status": "NOT_COMPLETED",
                },
                "hardening_training_plan": {
                    "seed_stability_required": True,
                    "minimum_independent_training_seeds": 5,
                },
            }
        ),
        encoding="utf-8",
    )
    (reports / "phase3_open_spiel_arena.json").write_text(
        json.dumps(
            {
                "rl_training_proof_boundary": {
                    "status": "TRAINING_PROOF_NOT_COMPLETED",
                    "seed_stability_required": True,
                    "seed_stability_evaluated": False,
                }
            }
        ),
        encoding="utf-8",
    )


def test_evaluation_metric_contract_requires_full_metric_bundle(tmp_path: Path) -> None:
    _write_reports(tmp_path / "reports")

    payload = build_evaluation_metric_contract(tmp_path)

    assert payload["overall_status"] == "PASS"
    assert payload["boundary"] == "ACCURACY_ALONE_NOT_SUFFICIENT"
    assert payload["accuracy_alone_sufficient"] is False
    assert payload["metric_families"]["action_classification"]["metrics"]["accuracy"] == 0.72
    assert payload["metric_families"]["action_classification"]["metrics"]["macro_f1"] == 0.49
    assert payload["metric_families"]["calibration"]["metrics"]["ece_10"] == 0.18
    assert payload["metric_families"]["action_distribution"]["metrics"]["js_divergence"] == 0.0026
    assert payload["metric_families"]["bet_sizing"]["bet_size_mae_required_for_final_high_realism"] is True
    assert payload["metric_families"]["simulation_return"]["metrics"]["win_rate"] == 0.577
    assert payload["metric_families"]["simulation_return"]["metrics"]["expected_value_delta_vs_baseline"] == 0.82
    assert payload["metric_families"]["seed_stability"]["metrics"]["full_training_seed_stability_required"] is True
    assert payload["final_metric_bundle_passed"] is False
    assert payload["final_strategy_quality_claim_allowed"] is False
    assert payload["current_delivery_blocker"] is False
    assert payload["model_quality_risk"] is True
    assert all(case["result"] == "PASS" for case in payload["proof_cases"])


def test_evaluation_metric_contract_blocks_accuracy_only_claim(tmp_path: Path) -> None:
    _write_reports(tmp_path / "reports")
    payload = build_evaluation_metric_contract(tmp_path)
    payload["accuracy_alone_sufficient"] = True
    payload["final_strategy_quality_claim_allowed"] = True
    payload.pop("overall_status", None)

    invariants = validate_evaluation_metric_contract(payload)

    assert invariants["status"] == "FAIL"
    assert "accuracy_alone_must_not_be_sufficient" in invariants["violations"]
    assert "final_strategy_quality_claim_must_be_blocked_until_full_metric_bundle" in invariants["violations"]


def test_evaluation_metric_contract_endpoint_returns_contract() -> None:
    from poker_agent.service import evaluation_metric_contract_json

    payload = evaluation_metric_contract_json()

    assert payload["overall_status"] == "PASS"
    assert payload["accuracy_alone_sufficient"] is False
    assert payload["final_strategy_quality_claim_allowed"] is False
