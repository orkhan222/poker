from __future__ import annotations

import json
from pathlib import Path

from poker_agent.hole_card_data_quality import build_hole_card_data_quality, validate_hole_card_data_quality


def test_hole_card_data_quality_keeps_limitation_open_with_routed_mitigation(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "dataset_audit.json").write_text(
        json.dumps(
            {
                "players": {
                    "rows": 100,
                    "missing_hole_card_rate": 0.77,
                    "partial_hole_card_rate": 0.09,
                    "complete_hole_card_rate": 0.14,
                },
                "findings": [
                    {
                        "severity": "blocker",
                        "issue": "Hole-card coverage is too low for card-strength modeling.",
                        "recommendation": "Improve OCR/card extraction.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (reports / "today_acceptance_training.json").write_text(
        json.dumps({"selected_architecture": "routed_policy_bundle"}),
        encoding="utf-8",
    )

    payload = build_hole_card_data_quality(tmp_path)

    assert payload["overall_status"] == "PASS"
    assert payload["mitigation_boundary"]["mitigation_status"] == "MITIGATED_BY_ROUTED_POLICY_BUNDLE"
    assert payload["mitigation_boundary"]["fully_solves_upstream_data_quality_issue"] is False
    assert payload["upstream_data_quality_boundary"]["limitation_status"] == "OPEN_DATA_QUALITY_LIMITATION"
    assert payload["upstream_data_quality_boundary"]["upstream_data_quality_issue_resolved"] is False
    assert payload["upstream_data_quality_boundary"]["production_blocker_for_current_deployment"] is False


def test_hole_card_data_quality_blocks_false_resolution_claim() -> None:
    payload = {
        "coverage_snapshot": {
            "missing_hole_card_rate": 0.77,
            "complete_hole_card_rate": 0.14,
            "audit_finding": {"issue": "Hole-card coverage is too low."},
        },
        "mitigation_boundary": {
            "mitigation_status": "MITIGATED_BY_ROUTED_POLICY_BUNDLE",
            "routed_policy_bundle_handles_missingness": True,
            "fully_solves_upstream_data_quality_issue": True,
        },
        "upstream_data_quality_boundary": {
            "limitation_status": "RESOLVED",
            "upstream_status": "RESOLVED",
            "upstream_data_quality_issue_resolved": True,
            "requires_ocr_or_parser_improvement": False,
            "requires_larger_reviewed_card_labels": False,
            "production_blocker_for_current_deployment": False,
            "component_risk": False,
        },
    }

    invariants = validate_hole_card_data_quality(payload)

    assert invariants["status"] == "FAIL"
    assert "routed_policy_bundle_must_not_claim_to_fully_solve_upstream_data_quality" in invariants["violations"]
    assert "upstream_data_quality_issue_cannot_be_marked_resolved" in invariants["violations"]


def test_hole_card_data_quality_endpoint_returns_contract() -> None:
    from poker_agent.service import hole_card_data_quality_json

    payload = hole_card_data_quality_json()

    assert payload["overall_status"] == "PASS"
    assert payload["mitigation_boundary"]["mitigation_status"] == "MITIGATED_BY_ROUTED_POLICY_BUNDLE"
    assert payload["upstream_data_quality_boundary"]["upstream_data_quality_issue_resolved"] is False
