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
        "metrics_to_revalidate": [],
    }

    invariants = validate_bet_timing_calibration(payload)

    assert invariants["status"] == "FAIL"
    assert "final_high_realism_claim_must_remain_blocked" in invariants["violations"]
    assert "more_real_player_behavior_labels_must_remain_required" in invariants["violations"]


def test_bet_timing_calibration_endpoint_returns_contract() -> None:
    from poker_agent.service import bet_timing_calibration_json

    payload = bet_timing_calibration_json()

    assert payload["overall_status"] == "PASS"
    assert payload["current_delivery_scope"]["implementation_status"] == "IMPLEMENTED_AND_MEASURED"
    assert payload["calibration_boundary"]["requires_more_real_player_behavior_labels"] is True
