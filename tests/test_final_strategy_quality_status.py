from __future__ import annotations

import json
from pathlib import Path

from poker_agent.final_strategy_quality_status import (
    build_final_strategy_quality_status,
    is_delivery_ready_but_competitive_claim_blocked,
    validate_final_strategy_quality_status,
)


def _write_reports(reports: Path) -> None:
    reports.mkdir()
    (reports / "final_delivery_acceptance.json").write_text(
        json.dumps(
            {
                "overall_status": "PASS",
                "acceptance_summary": {
                    "service_delivery": "READY",
                    "deployed_strategy_stack": "APPROVED",
                },
            }
        ),
        encoding="utf-8",
    )
    (reports / "challenger_strategy_quality.json").write_text(
        json.dumps(
            {
                "overall_status": "PASS",
                "strategy_quality_boundary": {
                    "final_production_strategy_quality_claim_allowed": False,
                    "challenger_gate_status": "FAIL",
                    "raw_production_gate_status": "FAIL",
                },
                "challenger_result": {
                    "best_candidate": "extra_trees_sqrt_balanced_full",
                    "macro_f1": 0.4827,
                    "failed_gates": ["macro_f1", "calibration"],
                },
            }
        ),
        encoding="utf-8",
    )
    (reports / "hole_card_data_quality.json").write_text(
        json.dumps(
            {
                "upstream_data_quality_boundary": {
                    "limitation_status": "OPEN_DATA_QUALITY_LIMITATION",
                    "upstream_data_quality_issue_resolved": False,
                    "requires_ocr_or_parser_improvement": True,
                },
                "strength_signal_impact": {"status": "DEGRADED_BY_MISSING_HOLE_CARDS"},
            }
        ),
        encoding="utf-8",
    )
    (reports / "bet_timing_calibration.json").write_text(
        json.dumps(
            {
                "calibration_boundary": {
                    "status": "CALIBRATION_RECOMMENDED_FOR_HIGHER_REALISM",
                    "requires_more_real_player_behavior_labels": True,
                    "final_high_realism_claim_allowed": False,
                }
            }
        ),
        encoding="utf-8",
    )
    (reports / "behavioral_revalidation.json").write_text(
        json.dumps(
            {
                "revalidation_boundary": {
                    "larger_clean_real_gameplay_revalidation_required": True,
                    "generalized_human_likeness_claim_allowed": False,
                }
            }
        ),
        encoding="utf-8",
    )
    (reports / "multi_agent_training_status.json").write_text(
        json.dumps(
            {
                "training_boundary": {
                    "acceptance_training_status": "PASS",
                    "full_production_scale_multi_agent_training_status": "NOT_COMPLETED",
                    "production_blocker": False,
                }
            }
        ),
        encoding="utf-8",
    )
    (reports / "production_runtime_monitoring.json").write_text(
        json.dumps(
            {
                "runtime_observability_boundary": {
                    "real_production_traffic_approved": False,
                    "real_production_traffic_approval_status": "NOT_APPROVED_UNTIL_OBSERVABILITY_ENABLED",
                    "monitoring_required_for_real_traffic": True,
                    "rollback_rules_required_for_real_traffic": True,
                    "live_drift_tracking_required_for_real_traffic": True,
                    "prediction_distribution_tracking_required_for_real_traffic": True,
                    "model_confidence_monitoring_required_for_real_traffic": True,
                }
            }
        ),
        encoding="utf-8",
    )


def test_final_strategy_quality_status_blocks_final_claim_but_keeps_delivery_ready(tmp_path: Path) -> None:
    _write_reports(tmp_path / "reports")

    payload = build_final_strategy_quality_status(tmp_path)

    assert payload["overall_status"] == "PASS"
    assert payload["delivery_boundary"]["software_delivery_ready"] is True
    assert payload["delivery_boundary"]["current_delivery_blocker"] is False
    deployment_vs_competitive = payload["deployment_vs_competitive_claim_boundary"]
    assert deployment_vs_competitive["deployment_delivery_ready"] is True
    assert deployment_vs_competitive["deployment_claim_allowed"] is True
    assert deployment_vs_competitive["deployment_sufficient_components"]["fastapi_service"] is True
    assert deployment_vs_competitive["deployment_sufficient_components"]["docker_packaging"] is True
    assert deployment_vs_competitive["deployment_sufficient_components"]["predict_endpoint"] is True
    assert deployment_vs_competitive["deployment_sufficient_components"]["health_endpoint"] is True
    assert deployment_vs_competitive["competitive_poker_agent_claim_allowed"] is False
    assert (
        deployment_vs_competitive["competitive_poker_agent_claim_state"]
        == "BLOCKED_PENDING_MODEL_DATA_AND_TRAINING_HARDENING"
    )
    assert deployment_vs_competitive["current_delivery_blocker"] is False
    boundary = payload["final_strategy_quality_boundary"]
    assert boundary["status"] == "NOT_APPROVED_PENDING_HARDENING_GATES"
    assert boundary["final_production_strategy_quality_approved"] is False
    assert boundary["final_production_strategy_quality_claim_allowed"] is False
    assert boundary["delivery_blocker"] is False
    assert boundary["deployed_strategy_stack_affected"] is False
    assert set(payload["remaining_work"]) == {
        "stronger_challenger_model",
        "hole_card_data_quality",
        "calibration",
        "larger_validation_data",
        "production_scale_multi_agent_training",
    }
    assert all(item["status"] == "REQUIRED" for item in payload["remaining_work"].values())
    assert is_delivery_ready_but_competitive_claim_blocked(payload) is True


def test_final_strategy_quality_status_rejects_false_approval(tmp_path: Path) -> None:
    _write_reports(tmp_path / "reports")
    payload = build_final_strategy_quality_status(tmp_path)
    payload["final_strategy_quality_boundary"]["status"] = "APPROVED"
    payload["final_strategy_quality_boundary"]["final_production_strategy_quality_approved"] = True
    payload["final_strategy_quality_boundary"]["final_production_strategy_quality_claim_allowed"] = True
    payload["deployment_vs_competitive_claim_boundary"]["competitive_poker_agent_claim_allowed"] = True
    payload["deployment_vs_competitive_claim_boundary"]["competitive_poker_agent_claim_state"] = "APPROVED"
    payload["deployment_vs_competitive_claim_boundary"]["deployment_sufficient_components"]["predict_endpoint"] = False
    payload["deployment_vs_competitive_claim_boundary"]["current_delivery_blocker"] = True
    payload["blocked_claims"].remove("The current delivery is a final competitive poker agent.")
    payload["remaining_work"]["stronger_challenger_model"]["status"] = "COMPLETE"
    payload.pop("overall_status", None)

    invariants = validate_final_strategy_quality_status(payload)

    assert is_delivery_ready_but_competitive_claim_blocked(payload) is False
    assert invariants["status"] == "FAIL"
    assert "approved_strategy_requires_final_metric_bundle_pass" in invariants["violations"]
    assert "approved_strategy_requires_metric_bundle_claim_allowance" in invariants["violations"]
    assert "approved_strategy_requires_completed_work_item:hole_card_data_quality" in invariants["violations"]
    assert "approved_strategy_requires_completed_work_item:calibration" in invariants["violations"]
    assert "approved_strategy_requires_completed_work_item:larger_validation_data" in invariants["violations"]
    assert "approved_strategy_requires_completed_work_item:production_scale_multi_agent_training" in invariants["violations"]
    assert "deployment_component_must_be_present:predict_endpoint" in invariants["violations"]
    assert "competitive_claim_gap_must_not_block_current_delivery" in invariants["violations"]
    assert "completed_challenger_item_requires_passed_gates" in invariants["violations"]
    assert "approved_claims_must_include_final_strategy_quality_approval" in invariants["violations"]
    assert "approved_strategy_must_not_block_final_strategy_claim" in invariants["violations"]


def test_final_strategy_quality_status_opens_claim_when_every_gate_passes(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    _write_reports(reports)

    challenger = json.loads((reports / "challenger_strategy_quality.json").read_text(encoding="utf-8"))
    challenger["strategy_quality_boundary"].update(
        {
            "final_production_strategy_quality_claim_allowed": True,
            "challenger_gate_status": "PASS",
            "raw_production_gate_status": "PASS",
        }
    )
    challenger["challenger_result"]["failed_gates"] = []
    (reports / "challenger_strategy_quality.json").write_text(json.dumps(challenger), encoding="utf-8")

    hole_card = json.loads((reports / "hole_card_data_quality.json").read_text(encoding="utf-8"))
    hole_card["upstream_data_quality_boundary"].update(
        {
            "limitation_status": "RESOLVED",
            "upstream_data_quality_issue_resolved": True,
            "requires_ocr_or_parser_improvement": False,
        }
    )
    hole_card["strength_signal_impact"]["status"] = "SUFFICIENT_FOR_CARD_AWARE_POLICY_GATES"
    (reports / "hole_card_data_quality.json").write_text(json.dumps(hole_card), encoding="utf-8")

    calibration = json.loads((reports / "bet_timing_calibration.json").read_text(encoding="utf-8"))
    calibration["calibration_boundary"].update(
        {
            "requires_more_real_player_behavior_labels": False,
            "final_high_realism_claim_allowed": True,
        }
    )
    (reports / "bet_timing_calibration.json").write_text(json.dumps(calibration), encoding="utf-8")

    behavioral = json.loads((reports / "behavioral_revalidation.json").read_text(encoding="utf-8"))
    behavioral["revalidation_boundary"].update(
        {
            "larger_clean_real_gameplay_revalidation_required": False,
            "generalized_human_likeness_claim_allowed": True,
        }
    )
    (reports / "behavioral_revalidation.json").write_text(json.dumps(behavioral), encoding="utf-8")

    multi_agent = json.loads((reports / "multi_agent_training_status.json").read_text(encoding="utf-8"))
    multi_agent["training_boundary"]["full_production_scale_multi_agent_training_status"] = "COMPLETED"
    (reports / "multi_agent_training_status.json").write_text(json.dumps(multi_agent), encoding="utf-8")

    (reports / "evaluation_metric_contract.json").write_text(
        json.dumps(
            {
                "final_metric_bundle_passed": True,
                "final_strategy_quality_claim_allowed": True,
            }
        ),
        encoding="utf-8",
    )

    payload = build_final_strategy_quality_status(tmp_path)

    assert payload["overall_status"] == "PASS"
    assert payload["final_strategy_quality_boundary"]["status"] == "APPROVED"
    assert payload["final_strategy_quality_boundary"]["final_production_strategy_quality_approved"] is True
    assert payload["final_strategy_quality_boundary"]["final_production_strategy_quality_claim_allowed"] is True
    assert payload["final_strategy_quality_boundary"]["final_metric_bundle_passed"] is True
    assert payload["final_strategy_quality_boundary"]["metric_bundle_claim_allowed"] is True
    assert payload["deployment_vs_competitive_claim_boundary"]["competitive_poker_agent_claim_allowed"] is True
    assert payload["deployment_vs_competitive_claim_boundary"]["competitive_poker_agent_claim_state"] == "APPROVED"
    assert all(item["status"] == "COMPLETE" for item in payload["remaining_work"].values())
    assert "Final production-level poker strategy quality is approved." in payload["allowed_claims"]
    assert "Final production-level poker strategy quality is approved." not in payload["blocked_claims"]
    assert is_delivery_ready_but_competitive_claim_blocked(payload) is False


def test_final_strategy_quality_status_endpoint_returns_contract() -> None:
    from poker_agent.service import final_strategy_quality_status_json

    payload = final_strategy_quality_status_json()

    assert payload["overall_status"] == "PASS"
    assert payload["delivery_boundary"]["software_delivery_ready"] is True
    assert payload["deployment_vs_competitive_claim_boundary"]["deployment_claim_allowed"] is True
    assert payload["deployment_vs_competitive_claim_boundary"]["competitive_poker_agent_claim_allowed"] is False
    assert payload["final_strategy_quality_boundary"]["final_production_strategy_quality_approved"] is False
    assert is_delivery_ready_but_competitive_claim_blocked(payload) is True


def test_delivery_strategy_boundary_requires_every_hardening_work_item(tmp_path: Path) -> None:
    _write_reports(tmp_path / "reports")
    payload = build_final_strategy_quality_status(tmp_path)

    assert is_delivery_ready_but_competitive_claim_blocked(payload) is True

    payload["remaining_work"]["production_scale_multi_agent_training"]["status"] = "COMPLETE"
    assert is_delivery_ready_but_competitive_claim_blocked(payload) is False

    payload = build_final_strategy_quality_status(tmp_path)
    payload["deployment_vs_competitive_claim_boundary"]["deployment_sufficient_components"][
        "docker_packaging"
    ] = False
    assert is_delivery_ready_but_competitive_claim_blocked(payload) is False
