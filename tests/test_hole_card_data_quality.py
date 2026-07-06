from __future__ import annotations

import json
from pathlib import Path

from poker_agent.hole_card_data_quality import build_hole_card_data_quality, validate_hole_card_data_quality


def test_hole_card_data_quality_keeps_limitation_open_with_routed_mitigation(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "players.csv").write_text(
        "\n".join(
            [
                "hand_id,position,cards",
                "h1,BTN,AS KD",
                "h2,SB,",
                "h3,BB,AS",
                "h4,UTG,ZZ QS",
                "h5,CO,AS AS",
            ]
        ),
        encoding="utf-8",
    )
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
    direct_audit = payload["coverage_snapshot"]["direct_players_csv_audit"]
    assert payload["coverage_snapshot"]["coverage_source"] == "direct_players_csv"
    assert direct_audit["status"] == "PASS"
    assert direct_audit["rows_scanned"] == 5
    assert direct_audit["missing_rows"] == 1
    assert direct_audit["partial_rows"] == 2
    assert direct_audit["complete_rows"] == 2
    assert direct_audit["reliable_two_card_rows"] == 1
    assert direct_audit["invalid_card_rows"] == 1
    assert direct_audit["duplicate_card_rows"] == 1
    assert direct_audit["reliable_two_card_rate"] == 0.2
    assert direct_audit["risk_status"] == "HIGH_RISK"
    assert payload["mitigation_boundary"]["mitigation_status"] == "MITIGATED_BY_ROUTED_POLICY_BUNDLE"
    assert payload["mitigation_boundary"]["mitigation_scope"] == "RUNTIME_RISK_REDUCTION_NOT_DATA_REPAIR"
    assert payload["mitigation_boundary"]["requires_slice_specific_monitoring"] is True
    assert payload["mitigation_boundary"]["fully_solves_upstream_data_quality_issue"] is False
    assert payload["strength_signal_impact"]["status"] == "DEGRADED_BY_MISSING_HOLE_CARDS"
    assert payload["strength_signal_impact"]["strength_proxy_zero_rate"] == 0.91
    assert payload["strength_signal_impact"]["observed_hole_cards_macro_f1"] == 0.39
    assert payload["strength_signal_impact"]["challenger_observed_hole_cards_macro_f1"] == 0.43
    assert payload["strength_signal_impact"]["primary_hand_strength_signal_reliable_for_standalone_policy"] is False
    assert "strength_proxy" in payload["strength_signal_impact"]["affected_features"]
    assert "made_hand_score" in payload["strength_signal_impact"]["affected_features"]
    risk = payload["risk_contract"]
    assert risk["risk_id"] == "hole_card_data_risk"
    assert risk["root_cause"] == "ocr_hole_card_extraction_missing_or_unreliable"
    assert risk["primary_dataset_column"] == "players.cards"
    assert risk["weakens_primary_poker_signal"] is True
    assert risk["affected_signal"] == "private_card_strength_and_texture"
    assert risk["feature_policy"]["missing_or_invalid_cards"] == "flag_and_route"
    assert risk["feature_policy"]["do_not_impute_unknown_cards_as_known_private_cards"] is True
    assert risk["feature_policy"]["do_not_treat_missing_cards_as_reliable_zero_strength"] is True
    assert risk["feature_policy"]["train_observed_card_and_public_context_slices_separately"] is True
    assert risk["current_delivery_blocker"] is False
    assert risk["final_strategy_quality_claim_blocker"] is True
    assert payload["upstream_data_quality_boundary"]["limitation_status"] == "OPEN_DATA_QUALITY_LIMITATION"
    assert payload["upstream_data_quality_boundary"]["upstream_data_quality_issue_resolved"] is False
    assert payload["upstream_data_quality_boundary"]["production_blocker_for_current_deployment"] is False
    assert payload["promotion_boundary"]["standalone_policy_promotion_allowed"] is False
    assert payload["promotion_boundary"]["model_promotion_blocker"] is True
    assert payload["promotion_boundary"]["current_deployment_blocker"] is False


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
        "promotion_boundary": {
            "standalone_policy_promotion_allowed": False,
            "model_promotion_blocker": True,
            "current_deployment_blocker": False,
            "requires_reliable_two_card_rate": 0.80,
            "requires_invalid_card_rate_below": 0.02,
            "requires_reviewed_card_label_set": True,
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
        "promotion_boundary": {
            "standalone_policy_promotion_allowed": False,
            "model_promotion_blocker": True,
            "current_deployment_blocker": False,
            "requires_reliable_two_card_rate": 0.80,
            "requires_invalid_card_rate_below": 0.02,
            "requires_reviewed_card_label_set": True,
        },
    }

    invariants = validate_hole_card_data_quality(payload)

    assert invariants["status"] == "FAIL"
    assert "high_hole_card_missingness_must_degrade_strength_signal" in invariants["violations"]
    assert "high_strength_proxy_zero_rate_must_degrade_strength_signal" in invariants["violations"]
    assert "degraded_strength_signal_cannot_be_standalone_reliable" in invariants["violations"]
    assert "strength_proxy_audit_finding_must_remain_visible" in invariants["violations"]


def test_hole_card_data_quality_blocks_false_standalone_promotion() -> None:
    payload = {
        "coverage_snapshot": {
            "missing_hole_card_rate": 0.77,
            "complete_hole_card_rate": 0.14,
            "audit_finding": {"issue": "Hole-card coverage is too low."},
            "direct_players_csv_audit": {
                "status": "PASS",
                "rows_scanned": 100,
                "missing_hole_card_rate": 0.77,
                "reliable_two_card_rate": 0.14,
                "invalid_card_rate": 0.03,
                "malformed_examples": [{"row_number": 5, "cards": "ZZ QQ"}],
            },
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
        "promotion_boundary": {
            "standalone_policy_promotion_allowed": True,
            "model_promotion_blocker": False,
            "current_deployment_blocker": True,
            "requires_reliable_two_card_rate": 0.80,
            "requires_invalid_card_rate_below": 0.02,
            "requires_reviewed_card_label_set": False,
        },
    }

    invariants = validate_hole_card_data_quality(payload)

    assert invariants["status"] == "FAIL"
    assert "hole_card_risk_must_block_standalone_policy_promotion" in invariants["violations"]
    assert "hole_card_risk_must_remain_model_promotion_blocker" in invariants["violations"]
    assert "hole_card_risk_must_not_block_current_deployment" in invariants["violations"]
    assert "hole_card_promotion_must_require_reviewed_card_labels" in invariants["violations"]


def test_hole_card_data_quality_endpoint_returns_contract() -> None:
    from poker_agent.service import hole_card_data_quality_json

    payload = hole_card_data_quality_json()

    assert payload["overall_status"] == "PASS"
    assert payload["mitigation_boundary"]["mitigation_status"] == "MITIGATED_BY_ROUTED_POLICY_BUNDLE"
    assert payload["upstream_data_quality_boundary"]["upstream_data_quality_issue_resolved"] is False
