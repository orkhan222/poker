from __future__ import annotations

import json
from pathlib import Path

from poker_agent.final_strategy_quality_status import (
    build_final_strategy_quality_status,
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


def test_final_strategy_quality_status_rejects_false_approval(tmp_path: Path) -> None:
    _write_reports(tmp_path / "reports")
    payload = build_final_strategy_quality_status(tmp_path)
    payload["final_strategy_quality_boundary"]["status"] = "APPROVED"
    payload["final_strategy_quality_boundary"]["final_production_strategy_quality_approved"] = True
    payload["final_strategy_quality_boundary"]["final_production_strategy_quality_claim_allowed"] = True
    payload["remaining_work"]["stronger_challenger_model"]["status"] = "COMPLETE"
    payload.pop("overall_status", None)

    invariants = validate_final_strategy_quality_status(payload)

    assert invariants["status"] == "FAIL"
    assert "final_production_strategy_quality_must_not_be_approved" in invariants["violations"]
    assert "final_production_strategy_quality_claim_must_be_blocked" in invariants["violations"]
    assert "final_strategy_quality_status_must_remain_not_approved" in invariants["violations"]
    assert "remaining_work_item_must_remain_required:stronger_challenger_model" in invariants["violations"]


def test_final_strategy_quality_status_endpoint_returns_contract() -> None:
    from poker_agent.service import final_strategy_quality_status_json

    payload = final_strategy_quality_status_json()

    assert payload["overall_status"] == "PASS"
    assert payload["delivery_boundary"]["software_delivery_ready"] is True
    assert payload["final_strategy_quality_boundary"]["final_production_strategy_quality_approved"] is False
