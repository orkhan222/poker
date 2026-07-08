from __future__ import annotations

import json
from pathlib import Path

from poker_agent.bet_timing_calibration import build_bet_timing_calibration, validate_bet_timing_calibration


def test_bet_timing_calibration_preserves_measured_current_scope_and_future_label_need(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "policy_acceptance.json").write_text(
        json.dumps({"human_likeness": {"status": "PASS", "timing_and_bet_size_status": "PASS"}}),
        encoding="utf-8",
    )
    (reports / "behavioral_revalidation.json").write_text(
        json.dumps(
            {
                "current_validation_scope": {"timing_and_bet_size_status": "PASS"},
                "metrics_to_revalidate": ["bet-size distribution similarity", "timing distribution similarity"],
            }
        ),
        encoding="utf-8",
    )

    payload = build_bet_timing_calibration(tmp_path)

    assert payload["overall_status"] == "PASS"
    assert payload["current_delivery_scope"]["implementation_status"] == "IMPLEMENTED_AND_MEASURED"
    assert payload["current_delivery_scope"]["bet_sizing_implemented"] is True
    assert payload["current_delivery_scope"]["timing_implemented"] is True
    assert payload["calibration_boundary"]["requires_more_real_player_behavior_labels"] is True
    assert payload["calibration_boundary"]["final_high_realism_claim_allowed"] is False
    assert payload["calibration_boundary"]["production_blocker_for_current_delivery"] is False
    assert payload["current_delivery_scope"]["timing_policy_type"] == "HEURISTIC_OR_TABLE_TEMPO_CALIBRATED"
    assert payload["current_delivery_scope"]["real_human_timing_label_quality"] == "TIMING_LABEL_QUALITY_UNCERTAIN"
    assert payload["current_delivery_scope"]["real_human_timing_labels_available"] is False
    assert payload["current_delivery_scope"]["timing_human_likeness_final_proof_allowed"] is False
    assert (
        payload["current_delivery_scope"]["timing_evidence_status"]
        == "HEURISTIC_TIMING_ONLY_NOT_FINAL_HUMAN_LIKENESS_PROOF"
    )
    assert (
        payload["timing_label_quality_boundary"]["boundary"]
        == "REAL_HUMAN_TIMING_LABELS_REQUIRED_FOR_FULL_HUMAN_LIKENESS_PROOF"
    )
    assert payload["timing_label_quality_boundary"]["timing_feature_available"] is True
    assert payload["timing_label_quality_boundary"]["requires_real_human_timing_labels"] is True
    assert payload["timing_label_quality_boundary"]["uses_real_human_timing_labels"] is False
    assert (
        payload["timing_label_quality_boundary"]["heuristic_timing_counts_as_full_human_likeness_proof"]
        is False
    )
    assert (
        payload["timing_label_quality_boundary"]["final_human_likeness_claim_allowed_from_timing_alone"]
        is False
    )
    assert payload["timing_label_quality_boundary"]["final_production_human_likeness_proof_allowed"] is False
    assert payload["timing_label_quality_boundary"]["current_delivery_blocker"] is False
    assert payload["timing_label_quality_boundary"]["model_quality_risk"] is True
    assert set(payload["timing_label_quality_boundary"]["required_timing_label_fields"]) == {
        "decision_start_ts",
        "decision_end_ts",
        "human_wait_time_ms",
        "street",
        "position",
        "facing_bet",
        "action",
    }
    proof_cases = {case["name"]: case for case in payload["proof_cases"]}
    assert proof_cases["base_contract_is_valid"]["observed_status"] == "PASS"
    assert proof_cases["blocks_final_timing_human_likeness_claim"]["observed_status"] == "FAIL"
    assert proof_cases["blocks_unreviewed_timing_label_availability_claim"]["observed_status"] == "FAIL"
    assert proof_cases["blocks_heuristic_timing_relabel_as_supervised"]["observed_status"] == "FAIL"
    assert proof_cases["blocks_heuristic_timing_as_full_human_likeness_proof"]["observed_status"] == "FAIL"
    assert proof_cases["blocks_missing_real_timing_label_contract"]["observed_status"] == "FAIL"


def test_bet_timing_calibration_blocks_false_final_realism_claim() -> None:
    payload = {
        "current_delivery_scope": {
            "api_response_fields": ["bet_size", "wait_time_ms", "sizing_method", "timing_method"],
            "bet_sizing_implemented": True,
            "timing_implemented": True,
            "measured": True,
            "timing_and_bet_size_status": "PASS",
        },
        "calibration_boundary": {
            "status": "CALIBRATION_COMPLETE",
            "requires_more_real_player_behavior_labels": False,
            "requires_bet_size_labels": False,
            "requires_decision_timing_labels": False,
            "requires_slice_level_calibration": False,
            "label_gap_status": "NO_GAP",
            "production_blocker_for_current_delivery": False,
            "final_high_realism_claim_allowed": True,
        },
        "timing_label_quality_boundary": {
            "status": "TIMING_LABELS_VERIFIED",
            "boundary": "NO_TIMING_LABEL_BOUNDARY",
            "timing_feature_available": True,
            "timing_policy_type": "LEARNED_FROM_REVIEWED_HUMAN_TIMING_LABELS",
            "real_human_timing_labels_available": True,
            "requires_real_human_timing_labels": False,
            "uses_real_human_timing_labels": True,
            "required_timing_label_fields": ["human_wait_time_ms"],
            "heuristic_timing_counts_as_full_human_likeness_proof": True,
            "final_human_likeness_claim_allowed_from_timing_alone": True,
            "final_production_human_likeness_proof_allowed": True,
            "current_delivery_blocker": False,
            "model_quality_risk": False,
        },
        "metrics_to_revalidate": [],
    }

    invariants = validate_bet_timing_calibration(payload)

    assert invariants["status"] == "FAIL"
    assert "final_high_realism_claim_must_remain_blocked" in invariants["violations"]
    assert "more_real_player_behavior_labels_must_remain_required" in invariants["violations"]
    assert "timing_label_quality_status_must_remain_uncertain" in invariants["violations"]
    assert "timing_boundary_must_not_claim_real_human_timing_labels_available" in invariants["violations"]
    assert "timing_final_production_human_likeness_proof_must_remain_blocked" in invariants["violations"]
    assert "timing_label_boundary_must_require_real_human_timing_labels" in invariants["violations"]
    assert "timing_boundary_must_require_real_human_timing_labels" in invariants["violations"]
    assert "timing_boundary_must_not_claim_real_human_timing_labels_are_used" in invariants["violations"]
    assert "timing_boundary_required_label_fields_must_be_complete" in invariants["violations"]
    assert "heuristic_timing_must_not_count_as_full_human_likeness_proof" in invariants["violations"]
    assert "timing_alone_must_not_allow_final_human_likeness_claim" in invariants["violations"]


def test_bet_timing_calibration_endpoint_returns_contract() -> None:
    from poker_agent.service import bet_timing_calibration_json
    from poker_agent.api_contract import api_contract

    contract = api_contract()["bet_timing_calibration"]
    payload = bet_timing_calibration_json()

    assert contract["endpoint"] == "/bet-timing-calibration.json"
    assert contract["timing_label_quality_status"] == "TIMING_LABEL_QUALITY_UNCERTAIN"
    assert (
        contract["timing_label_boundary"]["boundary"]
        == "REAL_HUMAN_TIMING_LABELS_REQUIRED_FOR_FULL_HUMAN_LIKENESS_PROOF"
    )
    assert contract["timing_label_boundary"]["requires_real_human_timing_labels"] is True
    assert contract["timing_label_boundary"]["uses_real_human_timing_labels"] is False
    assert (
        contract["timing_label_boundary"]["heuristic_timing_counts_as_full_human_likeness_proof"]
        is False
    )
    assert (
        contract["timing_label_boundary"]["final_human_likeness_claim_allowed_from_timing_alone"]
        is False
    )
    assert contract["final_production_human_likeness_proof_allowed"] is False
    assert payload["overall_status"] == "PASS"
    assert payload["current_delivery_scope"]["implementation_status"] == "IMPLEMENTED_AND_MEASURED"
    assert payload["calibration_boundary"]["requires_more_real_player_behavior_labels"] is True
    assert payload["timing_label_quality_boundary"]["status"] == "TIMING_LABEL_QUALITY_UNCERTAIN"
