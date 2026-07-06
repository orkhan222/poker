from __future__ import annotations

import csv
from pathlib import Path

from poker_agent.actions_context_quality import (
    ACTION_CONTEXT_RISK_ID,
    ACTION_CONTEXT_ROOT_CAUSE,
    ACTION_CONTEXT_SOURCE_TABLE,
    EXPLICIT_BETTING_CONTEXT_STATUS,
    REQUIRED_EXPLICIT_ACTION_FIELDS,
    build_actions_context_quality,
    validate_actions_context_quality,
)
from poker_agent.api_contract import api_contract
from poker_agent.features import load_training_examples


def test_actions_context_quality_passes_with_open_dataset_limitation() -> None:
    project_root = Path(__file__).resolve().parents[1]

    payload = build_actions_context_quality(project_root, max_examples=200)
    schema = payload["actions_csv_schema_audit"]
    risk = payload["risk_contract"]
    mitigation = payload["derived_context_mitigation"]
    feature_audit = payload["training_feature_audit"]

    assert payload["overall_status"] == "PASS"
    assert risk["risk_id"] == ACTION_CONTEXT_RISK_ID
    assert risk["root_cause"] == ACTION_CONTEXT_ROOT_CAUSE
    assert risk["source_table"] == ACTION_CONTEXT_SOURCE_TABLE
    assert set(risk["missing_or_reconstructed_decision_fields"]) == set(REQUIRED_EXPLICIT_ACTION_FIELDS)
    assert set(risk["decision_time_context_policy"]) == set(REQUIRED_EXPLICIT_ACTION_FIELDS)
    assert risk["target_row_values_are_labels_not_features"] is True
    for field in ["amount", "to_call", "pot_before_action", "min_raise", "legal_actions"]:
        assert risk["decision_time_context_policy"][field]["target_row_value_allowed_as_feature"] is False
        assert risk["decision_time_context_policy"][field]["reconstruction_source"]
    assert risk["current_delivery_blocker"] is False
    assert risk["model_quality_risk"] is True
    assert risk["final_strategy_quality_claim_blocker_without_richer_action_context"] is True
    assert schema["explicit_context_status"] == EXPLICIT_BETTING_CONTEXT_STATUS
    assert set(REQUIRED_EXPLICIT_ACTION_FIELDS).issubset(set(schema["missing_explicit_context_fields"]))
    assert mitigation["does_not_fully_replace_explicit_context"] is True
    assert mitigation["target_action_context_leakage_guard"] is True
    assert mitigation["final_strategy_quality_claim_blocker_without_richer_action_context"] is True
    assert mitigation["current_delivery_blocker"] is False
    assert mitigation["model_quality_risk"] is True
    assert feature_audit["status"] == "PASS"
    assert "hand_action_order" in feature_audit["required_derived_features_present"]
    assert "call_price_ratio" in feature_audit["required_derived_features_present"]


def test_actions_context_quality_is_exposed_through_api_contract() -> None:
    from poker_agent.service import actions_context_quality_json

    contract = api_contract()["actions_context_quality"]
    payload = actions_context_quality_json()

    assert contract["endpoint"] == "/actions-context-quality.json"
    assert "to_call" in contract["missing_or_required_explicit_fields"]
    assert payload["overall_status"] == "PASS"
    assert payload["actions_csv_schema_audit"]["explicit_context_status"] == EXPLICIT_BETTING_CONTEXT_STATUS


def test_actions_context_quality_blocks_false_completion_claim() -> None:
    payload = {
        "risk_contract": {
            "risk_id": "wrong",
            "root_cause": "wrong",
            "source_table": "actions.csv",
            "missing_or_reconstructed_decision_fields": ["to_call"],
            "decision_time_context_policy": {
                "to_call": {
                    "required_semantics": "",
                    "reconstruction_source": "",
                    "target_row_value_allowed_as_feature": True,
                }
            },
            "target_row_values_are_labels_not_features": False,
            "mitigation_status": "RESOLVED",
            "current_delivery_blocker": False,
            "model_quality_risk": False,
            "final_strategy_quality_claim_blocker_without_richer_action_context": False,
        },
        "required_explicit_action_fields": list(REQUIRED_EXPLICIT_ACTION_FIELDS),
        "actions_csv_schema_audit": {
            "status": "PASS",
            "explicit_context_status": "COMPLETE",
            "rows_scanned": 10,
            "missing_explicit_context_fields": ["to_call"],
            "limitation_status": "RESOLVED",
        },
        "derived_context_mitigation": {
            "status": "IMPLEMENTED_FROM_PRE_ACTION_EVENT_STREAM",
            "implemented": True,
            "uses_target_action_amount_as_feature": True,
            "target_action_context_leakage_guard": False,
            "uses_future_outcome_fields": False,
            "does_not_fully_replace_explicit_context": False,
            "current_delivery_blocker": False,
            "model_quality_risk": False,
            "final_strategy_quality_claim_blocker_without_richer_action_context": False,
        },
        "training_feature_audit": {
            "status": "PASS",
            "examples_scanned": 10,
            "missing_required_derived_features": [],
        },
    }

    invariants = validate_actions_context_quality(payload)

    assert invariants["status"] == "FAIL"
    assert "actions_context_risk_id_must_match_contract" in invariants["violations"]
    assert "actions_context_policy_must_cover_every_required_field" in invariants["violations"]
    assert "actions_context_target_row_values_must_be_labels_not_features" in invariants["violations"]
    assert "actions_context_mitigation_status_must_require_reconstruction" in invariants["violations"]
    assert "actions_context_must_block_final_strategy_claim_without_richer_data" in invariants["violations"]
    assert "missing_explicit_fields_must_remain_marked_incomplete" in invariants["violations"]
    assert "target_action_amount_must_not_be_used_as_feature" in invariants["violations"]
    assert "target_action_context_leakage_guard_must_be_enabled" in invariants["violations"]
    assert "derived_context_must_not_claim_full_replacement" in invariants["violations"]
    assert "actions_context_limitation_must_remain_model_quality_risk" in invariants["violations"]


def test_training_examples_include_pre_action_order_context(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "players.csv",
        ["hand_id", "position", "cards", "starting_stack"],
        [
            {"hand_id": "h1", "position": "UTG", "cards": "AS KD", "starting_stack": "100"},
            {"hand_id": "h1", "position": "BTN", "cards": "QS QH", "starting_stack": "100"},
        ],
    )
    _write_csv(
        tmp_path / "hands.csv",
        ["hand_id", "board_cards"],
        [{"hand_id": "h1", "board_cards": ""}],
    )
    _write_csv(
        tmp_path / "actions.csv",
        ["hand_id", "frame_id", "player_position", "action", "street"],
        [
            {"hand_id": "h1", "frame_id": "10", "player_position": "UTG", "action": "raise", "street": "preflop"},
            {"hand_id": "h1", "frame_id": "20", "player_position": "BTN", "action": "call", "street": "preflop"},
        ],
    )
    _write_csv(
        tmp_path / "stack_events.csv",
        ["hand_id", "frame_id", "player_position", "diff"],
        [
            {"hand_id": "h1", "frame_id": "10", "player_position": "UTG", "diff": "-4"},
            {"hand_id": "h1", "frame_id": "20", "player_position": "BTN", "diff": "-4"},
        ],
    )

    examples = load_training_examples(tmp_path, require_hole_cards=False, missing_hole_cards="flag")

    assert len(examples) == 2
    first_features, first_label = examples[0]
    second_features, second_label = examples[1]
    assert first_label == "raise"
    assert first_features["hand_action_order"] == 0.0
    assert first_features["street_action_order"] == 0.0
    assert first_features["betting_context_reconstructed"] == 1.0
    assert second_label == "call"
    assert second_features["hand_action_order"] == 1.0
    assert second_features["street_action_order"] == 1.0
    assert second_features["facing_bet_or_raise"] == 1.0
    assert second_features["call_price_ratio"] > 0.0
    assert second_features["explicit_to_call_available"] == 0.0


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
