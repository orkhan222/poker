from __future__ import annotations

import json
from pathlib import Path

from poker_agent.human_likeness_evidence import (
    REQUIRED_BEHAVIOR_DIMENSIONS,
    build_human_likeness_evidence,
    validate_human_likeness_evidence,
    write_human_likeness_evidence,
)


def _write_reports(reports: Path) -> None:
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "policy_acceptance.json").write_text(
        json.dumps({"human_likeness": {"status": "PASS", "js_divergence": 0.0026}}),
        encoding="utf-8",
    )
    (reports / "behavioral_revalidation.json").write_text(
        json.dumps(
            {
                "current_validation_scope": {
                    "action_distribution_status": "PASS",
                    "js_divergence": 0.0026,
                },
                "revalidation_boundary": {
                    "current_scope_claim_allowed": True,
                    "larger_clean_real_gameplay_revalidation_required": True,
                    "generalized_human_likeness_claim_allowed": False,
                },
            }
        ),
        encoding="utf-8",
    )
    (reports / "bet_timing_calibration.json").write_text(
        json.dumps(
            {
                "current_delivery_scope": {
                    "timing_and_bet_size_status": "PASS",
                    "bet_size_mae": None,
                    "decision_time_mae": None,
                },
                "calibration_boundary": {
                    "requires_more_real_player_behavior_labels": True,
                },
                "timing_label_quality_boundary": {
                    "final_production_human_likeness_proof_allowed": False,
                    "real_human_timing_labels_available": False,
                    "requires_real_human_timing_labels": True,
                    "heuristic_timing_counts_as_full_human_likeness_proof": False,
                    "final_human_likeness_claim_allowed_from_timing_alone": False,
                },
            }
        ),
        encoding="utf-8",
    )


def test_human_likeness_evidence_blocks_distribution_only_claim(tmp_path: Path) -> None:
    _write_reports(tmp_path / "reports")

    payload = build_human_likeness_evidence(tmp_path)

    assert payload["overall_status"] == "PASS"
    assert payload["current_scope_action_distribution_passed"] is True
    assert payload["human_likeness_fully_proven"] is False
    assert payload["final_human_likeness_claim_allowed"] is False
    assert payload["current_delivery_blocker"] is False
    assert set(payload["required_behavior_dimensions"]) == set(REQUIRED_BEHAVIOR_DIMENSIONS)
    assert payload["behavior_dimensions"]["bet_sizing"]["final_proof_allowed"] is False
    assert payload["behavior_dimensions"]["timing"]["final_proof_allowed"] is False
    assert payload["behavior_dimensions"]["timing"]["requires_real_human_timing_labels"] is True
    assert payload["behavior_dimensions"]["timing"]["real_human_timing_labels_available"] is False
    assert payload["behavior_dimensions"]["timing"]["heuristic_timing_counts_as_full_human_likeness_proof"] is False
    assert payload["upstream_boundaries"]["timing_alone_final_claim_allowed"] is False
    assert payload["upstream_boundaries"]["heuristic_timing_counts_as_full_human_likeness_proof"] is False
    assert payload["behavior_dimensions"]["position_based_behavior"]["required"] is True
    assert payload["behavior_dimensions"]["street_level_strategy"]["required"] is True


def test_human_likeness_evidence_rejects_false_claims(tmp_path: Path) -> None:
    _write_reports(tmp_path / "reports")
    payload = build_human_likeness_evidence(tmp_path)
    payload["human_likeness_fully_proven"] = True
    payload["final_human_likeness_claim_allowed"] = True
    payload["behavior_dimensions"]["bet_sizing"]["final_proof_allowed"] = True
    payload["behavior_dimensions"]["timing"]["final_proof_allowed"] = True
    payload["behavior_dimensions"]["position_based_behavior"]["required"] = False
    payload["behavior_dimensions"]["street_level_strategy"]["required"] = False

    invariants = validate_human_likeness_evidence(payload)

    assert invariants["status"] == "FAIL"
    assert "human_likeness_must_not_be_marked_fully_proven" in invariants["violations"]
    assert "final_human_likeness_claim_must_remain_blocked" in invariants["violations"]
    assert "behavior_dimension_final_proof_must_be_blocked:bet_sizing" in invariants["violations"]
    assert "behavior_dimension_final_proof_must_be_blocked:timing" in invariants["violations"]
    assert "behavior_dimension_must_be_required:position_based_behavior" in invariants["violations"]
    assert "behavior_dimension_must_be_required:street_level_strategy" in invariants["violations"]


def test_write_human_likeness_evidence_outputs_reports(tmp_path: Path) -> None:
    _write_reports(tmp_path / "reports")

    payload = write_human_likeness_evidence(
        tmp_path,
        tmp_path / "reports" / "human_likeness_evidence.json",
        tmp_path / "reports" / "human_likeness_evidence.md",
    )

    assert payload["overall_status"] == "PASS"
    assert (tmp_path / "reports" / "human_likeness_evidence.json").exists()
    assert "Action distribution alone" in (tmp_path / "reports" / "human_likeness_evidence.md").read_text(
        encoding="utf-8"
    )
