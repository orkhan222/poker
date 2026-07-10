from __future__ import annotations

import csv
from pathlib import Path

import pytest

from poker_agent.data_leakage_contract import (
    build_data_leakage_contract,
    evaluate_data_leakage_delivery_boundary,
    validate_data_leakage_contract,
)
from poker_agent.features import load_training_examples, request_to_features, visible_board_cards
from poker_agent.api_contract import api_contract
from poker_agent.leakage_guard import (
    assert_board_cards_visible_for_street,
    assert_no_final_board_snapshot_leakage,
    assert_no_outcome_feature_leakage,
    truncate_final_board_snapshot_for_decision,
)
from poker_agent.schemas import PredictionRequest


def test_data_leakage_contract_passes_on_current_project() -> None:
    project_root = Path(__file__).resolve().parents[1]

    payload = build_data_leakage_contract(project_root, max_examples=200)

    assert payload["overall_status"] == "PASS"
    assert payload["feature_name_audit"]["forbidden_feature_names_detected"] == []
    assert payload["prediction_request_audit"]["forbidden_feature_names_detected"] == []
    assert payload["model_artifact_audit"]["forbidden_model_features_detected"] == []
    assert payload["source_usage_audit"]["forbidden_source_usages"] == []
    assert payload["leakage_boundary"]["training_feature_use_allowed"] is False
    assert payload["leakage_boundary"]["dataset_schema_presence_allowed"] is True
    assert payload["leakage_boundary"]["direct_final_board_snapshot_feature_use_allowed"] is False
    assert payload["leakage_boundary"]["decision_time_visible_board_cards_allowed"] is True
    decision = payload["delivery_leakage_claim_decision"]
    assert decision["claim"] == "LEAKAGE_FREE_DECISION_TIME_FEATURES"
    assert decision["boundary"] == "POST_HAND_OUTCOME_FIELDS_FORBIDDEN_FOR_DECISION_TIME_FEATURES"
    assert decision["status"] == "PASS"
    assert decision["decision"] == "APPROVED"
    assert decision["claim_allowed"] is True
    assert decision["post_hand_outcome_fields_blocked"] is True
    assert decision["final_board_snapshot_direct_use_blocked"] is True
    assert decision["runtime_prediction_board_visibility_guard_enabled"] is True
    assert decision["street_truncation_function"] == "leakage_guard.truncate_final_board_snapshot_for_decision"
    assert decision["runtime_guard_function"] == "leakage_guard.assert_board_cards_visible_for_street"
    assert decision["detected_leakage_is_production_blocker"] is True
    assert decision["current_delivery_blocker"] is False
    assert set(decision["forbidden_post_hand_outcome_fields"]) == {
        "winner_positions",
        "stack_delta",
        "ending_stack",
        "dealer_winner",
        "dealer_pot",
        "pot_from_stacks",
    }
    assert decision["forbidden_final_snapshot_fields"] == ["hands.csv::board_cards"]
    risk = payload["leakage_risk_contract"]
    assert risk["risk_id"] == "post_outcome_feature_leakage"
    assert risk["root_cause"] == "post_hand_outcome_fields_available_in_raw_dataset_schema"
    assert risk["temporal_requirement"] == "features_must_be_observable_before_target_action"
    assert risk["feature_policy"]["raw_dataset_schema_presence"] == "allowed_for_audit_and_reporting_only"
    assert risk["feature_policy"]["training_feature_use"] == "forbidden"
    assert risk["feature_policy"]["prediction_request_use"] == "forbidden"
    assert risk["feature_policy"]["model_artifact_feature_use"] == "forbidden"
    assert risk["feature_policy"]["detected_violation"] == "production_blocker"
    assert risk["field_definitions"]["winner_positions"]["availability"] == "post_hand"
    assert risk["field_definitions"]["dealer_winner"]["availability"] == "post_hand"
    assert risk["field_definitions"]["dealer_pot"]["availability"] == "post_hand"
    assert risk["field_definitions"]["ending_stack"]["availability"] == "post_hand"
    assert risk["field_definitions"]["stack_delta"]["availability"] == "post_hand"
    assert risk["field_definitions"]["pot_from_stacks"]["availability"] == "post_hand_reconstruction"
    raw_schema = payload["raw_dataset_schema_audit"]
    assert raw_schema["status"] == "PASS"
    assert raw_schema["presence_is_not_feature_approval"] is True
    assert {
        item["field"] for item in raw_schema["outcome_fields_present_in_raw_schema"]
    }.issuperset({"winner_positions", "dealer_winner", "dealer_pot", "ending_stack", "stack_delta", "pot_from_stacks"})
    final_board = payload["final_board_snapshot_contract"]
    assert final_board["risk_id"] == "final_board_snapshot_leakage"
    assert final_board["root_cause"] == "hands_csv_board_cards_is_final_hand_snapshot"
    assert final_board["temporal_requirement"] == "board_features_must_be_truncated_to_cards_visible_at_target_street"
    assert final_board["raw_final_board_snapshot_fields"] == ["hands.csv::board_cards"]
    assert final_board["feature_policy"]["direct_training_feature_use"] == "forbidden"
    assert final_board["feature_policy"]["prediction_request_board_cards"] == "allowed_only_as_decision_time_visible_board"
    assert final_board["feature_policy"]["detected_violation"] == "production_blocker"
    assert final_board["required_mitigation"]["truncate_final_board_by_street"] is True
    assert final_board["required_mitigation"]["runtime_prediction_board_visibility_guard"] is True
    assert (
        final_board["required_mitigation"]["street_truncation_function"]
        == "leakage_guard.truncate_final_board_snapshot_for_decision"
    )
    assert (
        final_board["required_mitigation"]["runtime_guard_function"]
        == "leakage_guard.assert_board_cards_visible_for_street"
    )
    assert final_board["required_mitigation"]["preflop_visible_board_count"] == 0
    assert final_board["required_mitigation"]["flop_visible_board_count"] == 3
    assert final_board["required_mitigation"]["turn_visible_board_count"] == 4
    assert final_board["required_mitigation"]["river_visible_board_count"] == 5
    assert payload["raw_final_board_snapshot_fields"] == ["hands.csv::board_cards"]
    assert raw_schema["final_board_snapshot_presence_is_not_feature_approval"] is True
    assert raw_schema["final_board_snapshot_fields_present_in_raw_schema"] == [
        {
            "table": "hands.csv",
            "field": "board_cards",
            "source_field": "hands.csv::board_cards",
            "availability": "post_hand_final_snapshot",
            "presence_allowed": True,
            "allowed_use": "audit_and_street_truncation_only",
            "direct_training_feature_use_allowed": False,
        }
    ]


def test_data_leakage_contract_is_exposed_through_api_contract() -> None:
    from poker_agent.service import data_leakage_contract_json

    contract = api_contract()["data_leakage_contract"]
    payload = data_leakage_contract_json()

    assert contract["endpoint"] == "/data-leakage-contract.json"
    assert "ending_stack" in contract["forbidden_outcome_fields"]
    assert contract["raw_final_board_snapshot_fields"] == ["hands.csv::board_cards"]
    assert "visible at decision time" in contract["board_cards_boundary"]
    assert payload["overall_status"] == "PASS"
    assert payload["leakage_boundary"]["training_feature_use_allowed"] is False
    assert payload["delivery_leakage_claim_decision"]["claim_allowed"] is True
    assert payload["leakage_risk_contract"]["temporal_requirement"] == "features_must_be_observable_before_target_action"
    assert payload["final_board_snapshot_contract"]["feature_policy"]["direct_training_feature_use"] == "forbidden"


def test_delivery_leakage_claim_decision_blocks_outcome_and_final_board_features() -> None:
    project_root = Path(__file__).resolve().parents[1]
    payload = build_data_leakage_contract(project_root, max_examples=50)

    payload["feature_name_audit"] = {
        **payload["feature_name_audit"],
        "status": "FAIL",
        "forbidden_feature_names_detected": ["ending_stack"],
    }
    payload["prediction_request_audit"] = {
        **payload["prediction_request_audit"],
        "status": "FAIL",
        "forbidden_feature_names_detected": ["dealer_pot"],
    }
    payload["model_artifact_audit"] = {
        **payload["model_artifact_audit"],
        "status": "FAIL",
        "forbidden_model_features_detected": [{"path": "models/poker_policy.joblib", "feature": "stack_delta"}],
    }
    payload["source_usage_audit"] = {
        **payload["source_usage_audit"],
        "status": "FAIL",
        "forbidden_source_usages": [
            {
                "path": "poker_agent/features.py",
                "field": "winner_positions",
                "line_number": 1,
                "line": "features['winner_positions'] = row['winner_positions']",
            }
        ],
    }
    payload["leakage_boundary"] = {
        **payload["leakage_boundary"],
        "training_feature_use_allowed": True,
        "prediction_request_use_allowed": True,
        "direct_final_board_snapshot_feature_use_allowed": True,
    }
    payload["final_board_snapshot_contract"] = {
        **payload["final_board_snapshot_contract"],
        "feature_policy": {
            **payload["final_board_snapshot_contract"]["feature_policy"],
            "direct_training_feature_use": "allowed",
        },
    }

    decision = evaluate_data_leakage_delivery_boundary(payload)

    assert decision["status"] == "FAIL"
    assert decision["decision"] == "BLOCKED"
    assert decision["claim_allowed"] is False
    assert decision["current_delivery_blocker"] is False
    assert decision["detected_leakage_is_production_blocker"] is True
    assert "post_hand_outcome_training_feature_use_allowed" in decision["violations"]
    assert "post_hand_outcome_prediction_request_use_allowed" in decision["violations"]
    assert "direct_final_board_snapshot_feature_use_allowed" in decision["violations"]
    assert "direct_final_board_training_feature_use_not_forbidden" in decision["violations"]
    assert "forbidden_training_feature_names_detected" in decision["violations"]
    assert "forbidden_prediction_request_features_detected" in decision["violations"]
    assert "forbidden_model_artifact_features_detected" in decision["violations"]
    assert "forbidden_guarded_source_usage_detected" in decision["violations"]


def test_data_leakage_contract_blocks_false_safe_claims() -> None:
    payload = {
        "forbidden_outcome_fields": [
            "winner_positions",
            "stack_delta",
            "ending_stack",
            "dealer_winner",
            "dealer_pot",
            "pot_from_stacks",
        ],
        "leakage_risk_contract": {
            "risk_id": "post_outcome_feature_leakage",
            "root_cause": "post_hand_outcome_fields_available_in_raw_dataset_schema",
            "temporal_requirement": "features_must_be_observable_before_target_action",
            "forbidden_fields": [
                "winner_positions",
                "stack_delta",
                "ending_stack",
                "dealer_winner",
                "dealer_pot",
                "pot_from_stacks",
            ],
            "field_definitions": {
                "winner_positions": {"availability": "post_hand"},
                "stack_delta": {"availability": "post_hand"},
                "ending_stack": {"availability": "post_hand"},
                "dealer_winner": {"availability": "post_hand"},
                "dealer_pot": {"availability": "post_hand"},
                "pot_from_stacks": {"availability": "post_hand_reconstruction"},
            },
            "feature_policy": {
                "raw_dataset_schema_presence": "approved_for_training",
                "training_feature_use": "allowed",
                "prediction_request_use": "allowed",
                "model_artifact_feature_use": "allowed",
                "detected_violation": "warning_only",
            },
        },
        "leakage_boundary": {
            "status": "DATASET_ONLY_NOT_TRAINING_FEATURES",
            "decision_time_observability_required": False,
            "training_feature_use_allowed": True,
            "prediction_request_use_allowed": True,
            "model_artifact_feature_use_allowed": True,
            "dataset_schema_presence_allowed": False,
            "reporting_and_audit_use_allowed": False,
            "production_blocker_if_detected": False,
        },
        "feature_name_audit": {
            "status": "FAIL",
            "examples_scanned": 10,
            "forbidden_feature_names_detected": ["ending_stack"],
        },
        "prediction_request_audit": {
            "status": "FAIL",
            "forbidden_feature_names_detected": ["dealer_pot"],
        },
        "model_artifact_audit": {
            "status": "FAIL",
            "forbidden_model_features_detected": [{"path": "models/poker_policy.joblib", "feature": "stack_delta"}],
        },
        "source_usage_audit": {
            "status": "FAIL",
            "forbidden_source_usages": [
                {
                    "path": "poker_agent/features.py",
                    "field": "ending_stack",
                    "line_number": 10,
                    "line": "stack = safe_float(player.get('ending_stack'))",
                }
            ],
        },
        "raw_dataset_schema_audit": {
            "status": "PASS",
            "presence_is_not_feature_approval": False,
            "outcome_fields_present_in_raw_schema": [
                {
                    "table": "hands.csv",
                    "field": "winner_positions",
                    "presence_allowed": False,
                    "allowed_use": "training_feature",
                }
            ],
        },
    }

    invariants = validate_data_leakage_contract(payload)

    assert invariants["status"] == "FAIL"
    assert "outcome_fields_must_not_be_training_features" in invariants["violations"]
    assert "forbidden_training_feature_names_detected" in invariants["violations"]
    assert "forbidden_model_artifact_features_detected" in invariants["violations"]
    assert "forbidden_outcome_fields_used_in_guarded_source" in invariants["violations"]
    assert "raw_outcome_schema_presence_must_be_audit_only" in invariants["violations"]
    assert "risk_contract_training_feature_use_must_be_forbidden" in invariants["violations"]
    assert "risk_contract_prediction_request_use_must_be_forbidden" in invariants["violations"]
    assert "risk_contract_model_feature_use_must_be_forbidden" in invariants["violations"]
    assert "risk_contract_detected_violation_must_be_production_blocker" in invariants["violations"]
    assert "raw_schema_presence_must_not_equal_feature_approval" in invariants["violations"]
    assert "raw_outcome_field_presence_must_remain_allowed_for_audit" in invariants["violations"]
    assert "raw_outcome_field_allowed_use_must_be_audit_only" in invariants["violations"]
    assert "final_board_leakage_risk_id_must_be_explicit" in invariants["violations"]
    assert "direct_final_board_snapshot_must_not_be_feature" in invariants["violations"]
    assert "raw_final_board_presence_must_not_equal_feature_approval" in invariants["violations"]


def test_runtime_feature_guard_blocks_outcome_field_leakage() -> None:
    with pytest.raises(ValueError, match="ending_stack"):
        assert_no_outcome_feature_leakage(
            {
                "street=preflop": 1.0,
                "ending_stack": 100.0,
            },
            context="unit-test training features",
        )


def test_runtime_feature_guard_blocks_direct_final_board_snapshot_leakage() -> None:
    with pytest.raises(ValueError, match="hands.csv::board_cards"):
        assert_no_final_board_snapshot_leakage(
            ["hands.csv::board_cards"],
            context="unit-test training source fields",
        )


def test_final_board_snapshot_is_truncated_by_decision_street() -> None:
    final_board = ["AS", "KD", "QC", "2H", "7S"]

    assert truncate_final_board_snapshot_for_decision(final_board, "preflop") == []
    assert truncate_final_board_snapshot_for_decision(final_board, "flop") == ["AS", "KD", "QC"]
    assert truncate_final_board_snapshot_for_decision(final_board, "turn") == ["AS", "KD", "QC", "2H"]
    assert truncate_final_board_snapshot_for_decision(final_board, "river") == final_board
    assert visible_board_cards(final_board, "turn") == ["AS", "KD", "QC", "2H"]


def test_prediction_request_rejects_future_board_cards_for_street() -> None:
    assert_board_cards_visible_for_street(["AS", "KD", "QC"], "flop", context="unit-test board")

    with pytest.raises(ValueError, match="allows at most 3 visible board cards"):
        request_to_features(
            PredictionRequest(
                position="BTN",
                street="flop",
                hole_cards=["AH", "AD"],
                board_cards=["AS", "KD", "QC", "2H"],
                pot=10.0,
                to_call=2.0,
                stack=100.0,
                min_raise=4.0,
                player_count=6,
            )
        )


def test_training_features_do_not_fallback_to_ending_stack(tmp_path: Path) -> None:
    data_dir = tmp_path
    _write_csv(
        data_dir / "players.csv",
        ["hand_id", "position", "cards", "starting_stack", "ending_stack"],
        [{"hand_id": "h1", "position": "BTN", "cards": "AS KD", "starting_stack": "", "ending_stack": "100"}],
    )
    _write_csv(
        data_dir / "hands.csv",
        ["hand_id", "board_cards"],
        [{"hand_id": "h1", "board_cards": ""}],
    )
    _write_csv(
        data_dir / "actions.csv",
        ["hand_id", "frame_id", "player_position", "action", "street"],
        [{"hand_id": "h1", "frame_id": "1", "player_position": "BTN", "action": "fold", "street": "preflop"}],
    )
    _write_csv(
        data_dir / "stack_events.csv",
        ["hand_id", "frame_id", "player_position", "diff"],
        [],
    )

    examples = load_training_examples(data_dir, require_hole_cards=False, missing_hole_cards="flag")

    assert len(examples) == 1
    features, label = examples[0]
    assert label == "fold"
    assert features["stack"] == 0.0
    assert "ending_stack" not in features
    assert "stack_delta" not in features


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
