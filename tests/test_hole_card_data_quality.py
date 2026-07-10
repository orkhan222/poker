from __future__ import annotations

import json
from pathlib import Path

from poker_agent.hole_card_data_quality import (
    can_promote_standalone_policy_with_hole_cards,
    build_hole_card_data_quality,
    evaluate_hole_card_delivery_strategy_boundary,
    is_open_hole_card_data_quality_risk,
    validate_hole_card_data_quality,
)


def _risk_contract(final_claim_blocker: bool = True) -> dict:
    return {
        "risk_id": "hole_card_data_risk",
        "root_cause": "ocr_hole_card_extraction_missing_or_unreliable",
        "primary_dataset_column": "players.cards",
        "source_table": "players.csv",
        "source_field": "players.csv::cards",
        "cards_storage_boundary": {
            "players_csv_stores_hole_cards": True,
            "storage_does_not_imply_reliability": True,
            "card_values_are_ocr_or_recognition_derived": True,
            "missing_or_unreliable_cards_are_expected_dataset_conditions": True,
        },
        "weakens_primary_poker_signal": True,
        "affected_signal": "private_card_strength_and_texture",
        "affected_features": [
            "strength_proxy",
            "hole_card_observed_ratio",
            "hole_cards_missing",
            "card_texture_features",
            "made_hand_score",
            "draw_pressure",
        ],
        "hand_strength_feature_boundary": {
            "private_cards_are_primary_strategy_signal": True,
            "missing_or_invalid_hole_cards_limit_hand_strength_features": True,
            "hand_strength_features_must_be_slice_aware": True,
            "standalone_card_aware_policy_requires_reliable_two_card_coverage": True,
        },
        "feature_policy": {
            "missing_or_invalid_cards": "flag_and_route",
            "do_not_impute_unknown_cards_as_known_private_cards": True,
            "do_not_treat_missing_cards_as_reliable_zero_strength": True,
            "train_observed_card_and_public_context_slices_separately": True,
        },
        "current_delivery_blocker": False,
        "final_strategy_quality_claim_blocker": final_claim_blocker,
    }


def _players_csv_cards_contract() -> dict:
    return {
        "source_table": "players.csv",
        "source_field": "players.csv::cards",
        "semantic": "player_hole_cards",
        "quality_source": "ocr_or_card_recognition",
        "may_be_missing_or_unreliable": True,
        "must_not_be_treated_as_reliable_by_presence_alone": True,
    }


def _open_strength_signal(*, reliable: bool = False) -> dict:
    return {
        "status": "RELIABLE" if reliable else "DEGRADED_BY_MISSING_HOLE_CARDS",
        "affected_features": _risk_contract()["affected_features"],
        "strength_proxy_zero_rate": 0.91,
        "primary_signal_weakened_by_ocr_missingness": not reliable,
        "hand_strength_features_limited_by_card_quality": not reliable,
        "primary_hand_strength_signal_reliable_for_standalone_policy": reliable,
        "strength_proxy_audit_finding": None if reliable else {"issue": "strength_proxy is zero for most rows."},
    }


def _open_upstream_boundary() -> dict:
    return {
        "limitation_status": "OPEN_DATA_QUALITY_LIMITATION",
        "upstream_status": "UPSTREAM_NOT_RESOLVED",
        "upstream_data_quality_issue_resolved": False,
        "requires_ocr_or_parser_improvement": True,
        "requires_larger_reviewed_card_labels": True,
        "production_blocker_for_current_deployment": False,
        "component_risk": True,
        "players_csv_cards_are_not_reliability_guarantee": True,
        "hand_strength_signal_remains_limited_until_ocr_and_reviewed_labels_improve": True,
    }


def _delivery_strategy_boundary() -> dict:
    return {
        "risk_scope": "MODEL_QUALITY_RISK_NOT_SERVICE_DELIVERY_BLOCKER",
        "current_delivery_blocker": False,
        "service_delivery_claim_allowed": True,
        "deployed_routed_stack_delivery_allowed": True,
        "final_strategy_quality_claim_allowed": False,
        "final_strategy_quality_claim_blocked_by_hole_card_data_quality": True,
        "model_quality_risk": True,
        "component_risk": True,
        "requires_to_clear_final_strategy_claim": [
            "improved_ocr_or_card_parser",
            "larger_reviewed_hole_card_label_set",
            "reliable_two_card_coverage_gate",
            "observed_card_policy_slice_gate",
            "standalone_card_aware_policy_promotion_gate",
        ],
    }


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
    assert risk["source_table"] == "players.csv"
    assert risk["source_field"] == "players.csv::cards"
    assert risk["cards_storage_boundary"]["players_csv_stores_hole_cards"] is True
    assert risk["cards_storage_boundary"]["storage_does_not_imply_reliability"] is True
    assert risk["cards_storage_boundary"]["card_values_are_ocr_or_recognition_derived"] is True
    assert risk["cards_storage_boundary"]["missing_or_unreliable_cards_are_expected_dataset_conditions"] is True
    assert risk["weakens_primary_poker_signal"] is True
    assert risk["affected_signal"] == "private_card_strength_and_texture"
    assert risk["hand_strength_feature_boundary"]["private_cards_are_primary_strategy_signal"] is True
    assert risk["hand_strength_feature_boundary"]["missing_or_invalid_hole_cards_limit_hand_strength_features"] is True
    assert risk["hand_strength_feature_boundary"]["hand_strength_features_must_be_slice_aware"] is True
    assert (
        risk["hand_strength_feature_boundary"]["standalone_card_aware_policy_requires_reliable_two_card_coverage"]
        is True
    )
    assert risk["feature_policy"]["missing_or_invalid_cards"] == "flag_and_route"
    assert risk["feature_policy"]["do_not_impute_unknown_cards_as_known_private_cards"] is True
    assert risk["feature_policy"]["do_not_treat_missing_cards_as_reliable_zero_strength"] is True
    assert risk["feature_policy"]["train_observed_card_and_public_context_slices_separately"] is True
    assert risk["current_delivery_blocker"] is False
    assert risk["final_strategy_quality_claim_blocker"] is True
    players_csv_contract = payload["coverage_snapshot"]["players_csv_cards_contract"]
    assert players_csv_contract["source_table"] == "players.csv"
    assert players_csv_contract["source_field"] == "players.csv::cards"
    assert players_csv_contract["semantic"] == "player_hole_cards"
    assert players_csv_contract["quality_source"] == "ocr_or_card_recognition"
    assert players_csv_contract["may_be_missing_or_unreliable"] is True
    assert players_csv_contract["must_not_be_treated_as_reliable_by_presence_alone"] is True
    assert payload["strength_signal_impact"]["primary_signal_weakened_by_ocr_missingness"] is True
    assert payload["strength_signal_impact"]["hand_strength_features_limited_by_card_quality"] is True
    assert payload["upstream_data_quality_boundary"]["limitation_status"] == "OPEN_DATA_QUALITY_LIMITATION"
    assert payload["upstream_data_quality_boundary"]["upstream_data_quality_issue_resolved"] is False
    assert payload["upstream_data_quality_boundary"]["production_blocker_for_current_deployment"] is False
    assert payload["upstream_data_quality_boundary"]["players_csv_cards_are_not_reliability_guarantee"] is True
    assert (
        payload["upstream_data_quality_boundary"][
            "hand_strength_signal_remains_limited_until_ocr_and_reviewed_labels_improve"
        ]
        is True
    )
    assert payload["promotion_boundary"]["standalone_policy_promotion_allowed"] is False
    assert payload["promotion_boundary"]["model_promotion_blocker"] is True
    assert payload["promotion_boundary"]["current_deployment_blocker"] is False
    delivery_strategy = payload["delivery_strategy_quality_boundary"]
    assert delivery_strategy["risk_scope"] == "MODEL_QUALITY_RISK_NOT_SERVICE_DELIVERY_BLOCKER"
    assert delivery_strategy["current_delivery_blocker"] is False
    assert delivery_strategy["service_delivery_claim_allowed"] is True
    assert delivery_strategy["deployed_routed_stack_delivery_allowed"] is True
    assert delivery_strategy["final_strategy_quality_claim_allowed"] is False
    assert delivery_strategy["final_strategy_quality_claim_blocked_by_hole_card_data_quality"] is True
    assert delivery_strategy["model_quality_risk"] is True
    assert delivery_strategy["component_risk"] is True
    decision = evaluate_hole_card_delivery_strategy_boundary(payload)
    assert decision["status"] == "PASS"
    assert decision["service_delivery_ready"] is True
    assert decision["deployed_routed_stack_delivery_ready"] is True
    assert decision["open_hole_card_data_quality_risk"] is True
    assert decision["final_strategy_quality_claim_allowed"] is False
    assert decision["final_strategy_quality_claim_blocked"] is True
    assert decision["blocking_reason"] == "hole_card_data_quality_open"
    assert (
        decision["boundary"]
        == "DELIVERY_READY_FINAL_STRATEGY_CLAIM_BLOCKED_BY_HOLE_CARD_DATA_QUALITY"
    )
    assert payload["delivery_strategy_claim_decision"] == decision
    assert is_open_hole_card_data_quality_risk(payload) is True
    assert can_promote_standalone_policy_with_hole_cards(payload) is False


def test_hole_card_data_quality_blocks_false_resolution_claim() -> None:
    payload = {
        "risk_contract": _risk_contract(),
        "coverage_snapshot": {
            "players_csv_cards_contract": _players_csv_cards_contract(),
            "missing_hole_card_rate": 0.77,
            "complete_hole_card_rate": 0.14,
            "audit_finding": {"issue": "Hole-card coverage is too low."},
        },
        "strength_signal_impact": _open_strength_signal(),
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
        "delivery_strategy_quality_boundary": _delivery_strategy_boundary(),
    }

    invariants = validate_hole_card_data_quality(payload)

    assert invariants["status"] == "FAIL"
    assert "routed_policy_bundle_must_not_claim_to_fully_solve_upstream_data_quality" in invariants["violations"]
    assert "upstream_data_quality_issue_cannot_be_marked_resolved" in invariants["violations"]


def test_hole_card_data_quality_blocks_false_strength_signal_reliability() -> None:
    payload = {
        "risk_contract": _risk_contract(),
        "coverage_snapshot": {
            "players_csv_cards_contract": _players_csv_cards_contract(),
            "missing_hole_card_rate": 0.77,
            "complete_hole_card_rate": 0.14,
            "audit_finding": {"issue": "Hole-card coverage is too low."},
        },
        "strength_signal_impact": {
            "status": "RELIABLE",
            "affected_features": _risk_contract()["affected_features"],
            "strength_proxy_zero_rate": 0.91,
            "primary_signal_weakened_by_ocr_missingness": False,
            "hand_strength_features_limited_by_card_quality": False,
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
        "upstream_data_quality_boundary": _open_upstream_boundary(),
        "promotion_boundary": {
            "standalone_policy_promotion_allowed": False,
            "model_promotion_blocker": True,
            "current_deployment_blocker": False,
            "requires_reliable_two_card_rate": 0.80,
            "requires_invalid_card_rate_below": 0.02,
            "requires_reviewed_card_label_set": True,
        },
        "delivery_strategy_quality_boundary": _delivery_strategy_boundary(),
    }

    invariants = validate_hole_card_data_quality(payload)

    assert is_open_hole_card_data_quality_risk(payload) is False
    assert can_promote_standalone_policy_with_hole_cards(payload) is False
    assert invariants["status"] == "FAIL"
    assert "high_hole_card_missingness_must_degrade_strength_signal" in invariants["violations"]
    assert "high_strength_proxy_zero_rate_must_degrade_strength_signal" in invariants["violations"]
    assert "degraded_strength_signal_cannot_be_standalone_reliable" in invariants["violations"]
    assert "hole_card_ocr_missingness_must_weaken_primary_signal" in invariants["violations"]
    assert "hole_card_quality_must_limit_hand_strength_features" in invariants["violations"]
    assert "strength_proxy_audit_finding_must_remain_visible" in invariants["violations"]


def test_hole_card_data_quality_blocks_players_csv_presence_as_reliability_claim() -> None:
    risk = _risk_contract()
    risk["cards_storage_boundary"]["storage_does_not_imply_reliability"] = False
    risk["cards_storage_boundary"]["card_values_are_ocr_or_recognition_derived"] = False
    risk["hand_strength_feature_boundary"]["missing_or_invalid_hole_cards_limit_hand_strength_features"] = False
    payload = {
        "risk_contract": risk,
        "coverage_snapshot": {
            "players_csv_cards_contract": {
                **_players_csv_cards_contract(),
                "may_be_missing_or_unreliable": False,
                "must_not_be_treated_as_reliable_by_presence_alone": False,
            },
            "missing_hole_card_rate": 0.77,
            "complete_hole_card_rate": 0.14,
            "audit_finding": {"issue": "Hole-card coverage is too low."},
        },
        "strength_signal_impact": {
            **_open_strength_signal(),
            "primary_signal_weakened_by_ocr_missingness": False,
            "hand_strength_features_limited_by_card_quality": False,
        },
        "mitigation_boundary": {
            "mitigation_status": "MITIGATED_BY_ROUTED_POLICY_BUNDLE",
            "routed_policy_bundle_handles_missingness": True,
            "mitigation_scope": "RUNTIME_RISK_REDUCTION_NOT_DATA_REPAIR",
            "requires_slice_specific_monitoring": True,
            "fully_solves_upstream_data_quality_issue": False,
        },
        "upstream_data_quality_boundary": {
            **_open_upstream_boundary(),
            "players_csv_cards_are_not_reliability_guarantee": False,
            "hand_strength_signal_remains_limited_until_ocr_and_reviewed_labels_improve": False,
        },
        "promotion_boundary": {
            "standalone_policy_promotion_allowed": False,
            "model_promotion_blocker": True,
            "current_deployment_blocker": False,
            "requires_reliable_two_card_rate": 0.80,
            "requires_invalid_card_rate_below": 0.02,
            "requires_reviewed_card_label_set": True,
        },
        "delivery_strategy_quality_boundary": _delivery_strategy_boundary(),
    }

    invariants = validate_hole_card_data_quality(payload)

    assert invariants["status"] == "FAIL"
    assert "players_csv_cards_storage_must_not_imply_reliability" in invariants["violations"]
    assert "hole_card_values_must_remain_ocr_or_recognition_derived" in invariants["violations"]
    assert "missing_or_invalid_hole_cards_must_limit_hand_strength_features" in invariants["violations"]
    assert "players_csv_cards_contract_must_allow_missing_or_unreliable_values" in invariants["violations"]
    assert "players_csv_cards_presence_must_not_imply_reliability" in invariants["violations"]
    assert "hole_card_ocr_missingness_must_weaken_primary_signal" in invariants["violations"]
    assert "hole_card_quality_must_limit_hand_strength_features" in invariants["violations"]
    assert "players_csv_cards_must_not_be_reliability_guarantee" in invariants["violations"]
    assert "hand_strength_signal_limit_must_remain_until_data_repair" in invariants["violations"]


def test_hole_card_data_quality_blocks_false_standalone_promotion() -> None:
    payload = {
        "risk_contract": _risk_contract(),
        "coverage_snapshot": {
            "players_csv_cards_contract": _players_csv_cards_contract(),
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
        "strength_signal_impact": _open_strength_signal(),
        "mitigation_boundary": {
            "mitigation_status": "MITIGATED_BY_ROUTED_POLICY_BUNDLE",
            "routed_policy_bundle_handles_missingness": True,
            "mitigation_scope": "RUNTIME_RISK_REDUCTION_NOT_DATA_REPAIR",
            "requires_slice_specific_monitoring": True,
            "fully_solves_upstream_data_quality_issue": False,
        },
        "upstream_data_quality_boundary": _open_upstream_boundary(),
        "promotion_boundary": {
            "standalone_policy_promotion_allowed": True,
            "model_promotion_blocker": False,
            "current_deployment_blocker": True,
            "requires_reliable_two_card_rate": 0.80,
            "requires_invalid_card_rate_below": 0.02,
            "requires_reviewed_card_label_set": False,
        },
        "delivery_strategy_quality_boundary": _delivery_strategy_boundary(),
    }

    invariants = validate_hole_card_data_quality(payload)

    assert invariants["status"] == "FAIL"
    assert "hole_card_risk_must_block_standalone_policy_promotion" in invariants["violations"]
    assert "hole_card_risk_must_remain_model_promotion_blocker" in invariants["violations"]
    assert "hole_card_risk_must_not_block_current_deployment" in invariants["violations"]
    assert "hole_card_promotion_must_require_reviewed_card_labels" in invariants["violations"]


def test_hole_card_data_quality_blocks_delivery_strategy_boundary_drift() -> None:
    boundary = _delivery_strategy_boundary()
    boundary.update(
        {
            "current_delivery_blocker": True,
            "service_delivery_claim_allowed": False,
            "deployed_routed_stack_delivery_allowed": False,
            "final_strategy_quality_claim_allowed": True,
            "final_strategy_quality_claim_blocked_by_hole_card_data_quality": False,
            "model_quality_risk": False,
            "component_risk": False,
            "requires_to_clear_final_strategy_claim": ["improved_ocr_or_card_parser"],
        }
    )
    payload = {
        "risk_contract": _risk_contract(),
        "coverage_snapshot": {
            "players_csv_cards_contract": _players_csv_cards_contract(),
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
        "strength_signal_impact": _open_strength_signal(),
        "mitigation_boundary": {
            "mitigation_status": "MITIGATED_BY_ROUTED_POLICY_BUNDLE",
            "routed_policy_bundle_handles_missingness": True,
            "mitigation_scope": "RUNTIME_RISK_REDUCTION_NOT_DATA_REPAIR",
            "requires_slice_specific_monitoring": True,
            "fully_solves_upstream_data_quality_issue": False,
        },
        "upstream_data_quality_boundary": _open_upstream_boundary(),
        "promotion_boundary": {
            "standalone_policy_promotion_allowed": False,
            "model_promotion_blocker": True,
            "current_deployment_blocker": False,
            "requires_reliable_two_card_rate": 0.80,
            "requires_invalid_card_rate_below": 0.02,
            "requires_reviewed_card_label_set": True,
        },
        "delivery_strategy_quality_boundary": boundary,
    }

    invariants = validate_hole_card_data_quality(payload)

    assert invariants["status"] == "FAIL"
    assert "hole_card_risk_must_not_be_delivery_blocker" in invariants["violations"]
    assert "hole_card_risk_must_not_block_service_delivery_claim" in invariants["violations"]
    assert "hole_card_risk_must_not_block_deployed_routed_stack_delivery" in invariants["violations"]
    assert "hole_card_risk_must_block_final_strategy_quality_claim" in invariants["violations"]
    assert "hole_card_data_quality_must_block_final_strategy_quality_claim" in invariants["violations"]
    assert "hole_card_limitation_must_remain_model_quality_risk" in invariants["violations"]
    assert "hole_card_limitation_must_remain_component_risk" in invariants["violations"]
    assert "hole_card_final_strategy_claim_clearance_requirements_must_be_explicit" in invariants["violations"]
    decision = evaluate_hole_card_delivery_strategy_boundary(payload)
    assert decision["status"] == "FAIL"
    assert "hole_card_risk_cannot_block_current_service_delivery" in decision["violations"]
    assert "service_delivery_must_remain_allowed_with_open_hole_card_risk" in decision["violations"]
    assert "deployed_routed_stack_delivery_must_remain_allowed" in decision["violations"]
    assert "open_hole_card_risk_must_block_final_strategy_quality_claim" in decision["violations"]
    assert "final_strategy_quality_claim_must_name_hole_card_data_quality_blocker" in decision["violations"]
    assert "hole_card_boundary_must_remain_model_quality_risk" in decision["violations"]
    assert "hole_card_boundary_must_remain_component_risk" in decision["violations"]


def test_hole_card_data_quality_endpoint_returns_contract() -> None:
    from poker_agent.service import hole_card_data_quality_json

    payload = hole_card_data_quality_json()

    assert payload["overall_status"] == "PASS"
    assert payload["mitigation_boundary"]["mitigation_status"] == "MITIGATED_BY_ROUTED_POLICY_BUNDLE"
    assert payload["upstream_data_quality_boundary"]["upstream_data_quality_issue_resolved"] is False
    decision = payload["delivery_strategy_claim_decision"]
    assert decision["status"] == "PASS"
    assert decision["service_delivery_ready"] is True
    assert decision["deployed_routed_stack_delivery_ready"] is True
    assert decision["final_strategy_quality_claim_allowed"] is False
    assert decision["final_strategy_quality_claim_blocked"] is True
    assert is_open_hole_card_data_quality_risk(payload) is True
    assert can_promote_standalone_policy_with_hole_cards(payload) is False


def test_hole_card_data_quality_guard_allows_promotion_only_after_real_data_repair() -> None:
    payload = {
        "risk_contract": _risk_contract(final_claim_blocker=False),
        "coverage_snapshot": {
            "players_csv_cards_contract": _players_csv_cards_contract(),
            "direct_players_csv_audit": {
                "missing_hole_card_rate": 0.03,
                "reliable_two_card_rate": 0.92,
                "invalid_card_rate": 0.0,
            }
        },
        "strength_signal_impact": {
            "status": "RELIABLE",
            "affected_features": _risk_contract()["affected_features"],
            "strength_proxy_zero_rate": 0.05,
            "primary_signal_weakened_by_ocr_missingness": False,
            "hand_strength_features_limited_by_card_quality": False,
            "primary_hand_strength_signal_reliable_for_standalone_policy": True,
        },
        "mitigation_boundary": {
            "mitigation_status": "MITIGATED_BY_ROUTED_POLICY_BUNDLE",
            "mitigation_scope": "RUNTIME_RISK_REDUCTION_NOT_DATA_REPAIR",
            "fully_solves_upstream_data_quality_issue": True,
        },
        "upstream_data_quality_boundary": {
            "limitation_status": "RESOLVED",
            "upstream_data_quality_issue_resolved": True,
            "production_blocker_for_current_deployment": False,
            "component_risk": False,
            "players_csv_cards_are_not_reliability_guarantee": False,
            "hand_strength_signal_remains_limited_until_ocr_and_reviewed_labels_improve": False,
        },
        "promotion_boundary": {
            "standalone_policy_promotion_allowed": True,
            "model_promotion_blocker": False,
            "current_deployment_blocker": False,
        },
        "delivery_strategy_quality_boundary": _delivery_strategy_boundary(),
    }

    assert is_open_hole_card_data_quality_risk(payload) is False
    assert can_promote_standalone_policy_with_hole_cards(payload) is True

    payload["coverage_snapshot"]["direct_players_csv_audit"]["reliable_two_card_rate"] = 0.50
    assert is_open_hole_card_data_quality_risk(payload) is False
    assert can_promote_standalone_policy_with_hole_cards(payload) is False
