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
                "features": {
                    "critical_feature_zero_rates": {
                        "strength_proxy": 0.91,
                    }
                },
                "findings": [
                    {
                        "severity": "blocker",
                        "issue": "Hole-card coverage is too low for card-strength modeling.",
                        "recommendation": "Improve OCR/card extraction.",
                    },
                    {
                        "severity": "high",
                        "issue": "Critical feature `strength_proxy` is zero for most audited examples.",
                        "recommendation": "Add missingness-specific model paths.",
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
    (reports / "production_gate.json").write_text(
        json.dumps(
            {
                "gates": [
                    {
                        "name": "observed_hole_cards_macro_f1",
                        "observed": 0.39,
                        "threshold": 0.50,
                        "passed": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (reports / "raw_model_challenger.json").write_text(
        json.dumps(
            {
                "best_candidate": {
                    "gate": {"failed_gates": ["observed_hole_cards_macro_f1"]},
                    "valid_slice_metrics": {"observed_hole_cards": {"macro_f1": 0.43}},
                }
            }
        ),
        encoding="utf-8",
    )

    payload = build_hole_card_data_quality(tmp_path)

    assert payload["overall_status"] == "PASS"
    assert payload["mitigation_boundary"]["mitigation_status"] == "MITIGATED_BY_ROUTED_POLICY_BUNDLE"
    assert payload["mitigation_boundary"]["mitigation_scope"] == "RUNTIME_RISK_REDUCTION_NOT_DATA_REPAIR"
    assert payload["mitigation_boundary"]["requires_slice_specific_monitoring"] is True
    assert payload["mitigation_boundary"]["fully_solves_upstream_data_quality_issue"] is False
    assert payload["strength_signal_impact"]["status"] == "DEGRADED_BY_MISSING_HOLE_CARDS"
    assert payload["strength_signal_impact"]["strength_proxy_zero_rate"] == 0.91
    assert payload["strength_signal_impact"]["observed_hole_cards_macro_f1"] == 0.39
    assert payload["strength_signal_impact"]["challenger_observed_hole_cards_macro_f1"] == 0.43
    assert payload["strength_signal_impact"]["primary_hand_strength_signal_reliable_for_standalone_policy"] is False
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
        "strength_signal_impact": {
            "status": "DEGRADED_BY_MISSING_HOLE_CARDS",
            "strength_proxy_zero_rate": 0.91,
            "primary_hand_strength_signal_reliable_for_standalone_policy": False,
            "strength_proxy_audit_finding": {"issue": "strength_proxy is zero for most rows."},
        },
        "mitigation_boundary": {
            "mitigation_status": "MITIGATED_BY_ROUTED_POLICY_BUNDLE",
            "routed_policy_bundle_handles_missingness": True,
            "mitigation_scope": "RUNTIME_RISK_REDUCTION_NOT_DATA_REPAIR",
            "requires_slice_specific_monitoring": True,
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


def test_hole_card_data_quality_blocks_false_strength_signal_reliability() -> None:
    payload = {
        "coverage_snapshot": {
            "missing_hole_card_rate": 0.77,
            "complete_hole_card_rate": 0.14,
            "audit_finding": {"issue": "Hole-card coverage is too low."},
        },
        "strength_signal_impact": {
            "status": "RELIABLE",
            "strength_proxy_zero_rate": 0.91,
            "primary_hand_strength_signal_reliable_for_standalone_policy": True,
            "strength_proxy_audit_finding": None,
        },
        "mitigation_boundary": {
            "mitigation_status": "MITIGATED_BY_ROUTED_POLICY_BUNDLE",
            "routed_policy_bundle_handles_missingness": True,
            "mitigation_scope": "RUNTIME_RISK_REDUCTION_NOT_DATA_REPAIR",
            "requires_slice_specific_monitoring": True,
            "fully_solves_upstream_data_quality_issue": False,
        },
        "upstream_data_quality_boundary": {
            "limitation_status": "OPEN_DATA_QUALITY_LIMITATION",
            "upstream_status": "UPSTREAM_NOT_RESOLVED",
            "upstream_data_quality_issue_resolved": False,
            "requires_ocr_or_parser_improvement": True,
            "requires_larger_reviewed_card_labels": True,
            "production_blocker_for_current_deployment": False,
            "component_risk": True,
        },
    }

    invariants = validate_hole_card_data_quality(payload)

    assert invariants["status"] == "FAIL"
    assert "high_hole_card_missingness_must_degrade_strength_signal" in invariants["violations"]
    assert "high_strength_proxy_zero_rate_must_degrade_strength_signal" in invariants["violations"]
    assert "degraded_strength_signal_cannot_be_standalone_reliable" in invariants["violations"]
    assert "strength_proxy_audit_finding_must_remain_visible" in invariants["violations"]


def test_hole_card_data_quality_endpoint_returns_contract() -> None:
    from poker_agent.service import hole_card_data_quality_json

    payload = hole_card_data_quality_json()

    assert payload["overall_status"] == "PASS"
    assert payload["mitigation_boundary"]["mitigation_status"] == "MITIGATED_BY_ROUTED_POLICY_BUNDLE"
    assert payload["upstream_data_quality_boundary"]["upstream_data_quality_issue_resolved"] is False
