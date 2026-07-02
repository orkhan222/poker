from __future__ import annotations

import csv
from pathlib import Path

from poker_agent.api_contract import api_contract
from poker_agent.features import load_training_examples
from poker_agent.stack_event_context_quality import (
    build_stack_event_context_quality,
    validate_stack_event_context_quality,
)


def test_stack_event_context_quality_passes_on_current_project() -> None:
    project_root = Path(__file__).resolve().parents[1]

    payload = build_stack_event_context_quality(project_root, max_examples=200)

    assert payload["overall_status"] == "PASS"
    assert payload["raw_stack_event_boundary"]["raw_stack_events_are_direct_policy_features"] is False
    assert payload["raw_stack_event_boundary"]["decision_time_derivation_required"] is True
    assert payload["derived_context_mitigation"]["status"] == "IMPLEMENTED_FROM_PRE_ACTION_STACK_DELTAS"
    assert payload["derived_context_mitigation"]["uses_target_action_stack_delta_as_feature"] is False
    assert payload["training_feature_audit"]["status"] == "PASS"
    assert "reconstructed_effective_stack" in payload["training_feature_audit"]["required_stack_context_features_present"]
    assert "reconstructed_spr_after_call" in payload["training_feature_audit"]["required_stack_context_features_present"]
    assert (
        payload["training_feature_audit"]["sample_stack_context_feature_values"][
            "stack_event_target_bet_size_used_as_feature"
        ]
        == 0.0
    )
    proof_cases = {case["name"]: case for case in payload["proof_cases"]}
    assert proof_cases["base_contract_is_valid"]["passed"] is True
    assert proof_cases["blocks_raw_stack_events_as_direct_features"]["observed_status"] == "FAIL"
    assert proof_cases["blocks_target_action_stack_delta_feature_leakage"]["observed_status"] == "FAIL"
    assert proof_cases["blocks_delivery_blocker_reclassification"]["observed_status"] == "FAIL"
    assert proof_cases["blocks_model_quality_risk_removal"]["observed_status"] == "FAIL"


def test_stack_event_context_quality_is_exposed_through_api_contract() -> None:
    from poker_agent.service import stack_event_context_quality_json

    contract = api_contract()["stack_event_context_quality"]
    payload = stack_event_context_quality_json()

    assert contract["endpoint"] == "/stack-event-context-quality.json"
    assert "reconstructed_effective_stack" in contract["required_derived_features"]
    assert payload["overall_status"] == "PASS"
    assert payload["raw_stack_event_boundary"]["target_action_stack_delta_allowed_as_feature"] is False


def test_stack_event_context_quality_blocks_raw_event_overclaim() -> None:
    payload = {
        "stack_events_schema_audit": {
            "status": "PASS",
            "rows_scanned": 10,
            "negative_diff_rows": 3,
        },
        "raw_stack_event_boundary": {
            "status": "RAW_EVENTS_REQUIRE_DECISION_CONTEXT_DERIVATION",
            "raw_stack_events_are_direct_policy_features": True,
            "decision_time_derivation_required": False,
            "target_action_stack_delta_allowed_as_feature": True,
            "post_hand_stack_outcome_allowed_as_feature": True,
            "current_delivery_blocker": False,
            "model_quality_risk": False,
        },
        "derived_context_mitigation": {
            "status": "IMPLEMENTED_FROM_PRE_ACTION_STACK_DELTAS",
            "implemented": True,
            "uses_target_action_stack_delta_as_feature": True,
            "uses_post_hand_outcome_fields": True,
            "current_delivery_blocker": False,
            "model_quality_risk": False,
        },
        "training_feature_audit": {
            "status": "PASS",
            "examples_scanned": 10,
            "missing_required_stack_context_features": [],
            "sample_stack_context_feature_values": {
                "stack_event_target_bet_size_used_as_feature": 1.0,
            },
        },
    }

    invariants = validate_stack_event_context_quality(payload)

    assert invariants["status"] == "FAIL"
    assert "raw_stack_events_must_not_be_direct_policy_features" in invariants["violations"]
    assert "stack_events_must_require_decision_time_derivation" in invariants["violations"]
    assert "target_action_stack_delta_must_not_be_feature" in invariants["violations"]
    assert "derived_stack_context_must_not_use_target_action_delta" in invariants["violations"]
    assert "stack_event_context_gap_must_remain_model_quality_risk" in invariants["violations"]
    assert "target_stack_delta_leakage_guard_must_be_zero" in invariants["violations"]


def test_training_examples_include_stack_event_derived_pressure(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "players.csv",
        ["hand_id", "position", "cards", "starting_stack"],
        [
            {"hand_id": "h1", "position": "UTG", "cards": "AS KD", "starting_stack": "100"},
            {"hand_id": "h1", "position": "BTN", "cards": "QS QH", "starting_stack": "100"},
        ],
    )
    _write_csv(tmp_path / "hands.csv", ["hand_id", "board_cards"], [{"hand_id": "h1", "board_cards": ""}])
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
        ["hand_id", "frame_id", "player_position", "event", "stack", "diff", "stack_after_event"],
        [
            {"hand_id": "h1", "frame_id": "10", "player_position": "UTG", "event": "update_stack", "stack": "96", "diff": "-4", "stack_after_event": "96"},
            {"hand_id": "h1", "frame_id": "20", "player_position": "BTN", "event": "update_stack", "stack": "96", "diff": "-4", "stack_after_event": "96"},
        ],
    )

    examples = load_training_examples(tmp_path, require_hole_cards=False, missing_hole_cards="flag")

    assert len(examples) == 2
    second_features, second_label = examples[1]
    assert second_label == "call"
    assert second_features["stack_event_context_reconstructed"] == 1.0
    assert second_features["stack_event_target_bet_size_used_as_feature"] == 0.0
    assert second_features["reconstructed_effective_stack"] == 100.0
    assert second_features["reconstructed_current_street_bet_size"] == 4.0
    assert second_features["reconstructed_call_pressure"] > 0.0
    assert second_features["reconstructed_spr_after_call"] > 0.0


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
