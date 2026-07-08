from __future__ import annotations

import csv
from pathlib import Path

from poker_agent.api_contract import api_contract
from poker_agent.features import load_training_examples
from poker_agent.stack_event_context_quality import (
    STACK_CONTEXT_IMPLEMENTATION_MODULE,
    STACK_CONTEXT_PRE_ACTION_HELPER,
    STACK_CONTEXT_DERIVATION_POLICY,
    STACK_EVENT_RISK_ID,
    STACK_EVENT_ROOT_CAUSE,
    STACK_EVENT_SOURCE_TABLE,
    build_stack_event_context_quality,
    validate_stack_event_context_quality,
)


def test_stack_event_context_quality_passes_on_current_project() -> None:
    project_root = Path(__file__).resolve().parents[1]

    payload = build_stack_event_context_quality(project_root, max_examples=200)
    risk = payload["risk_contract"]

    assert payload["overall_status"] == "PASS"
    assert risk["risk_id"] == STACK_EVENT_RISK_ID
    assert risk["root_cause"] == STACK_EVENT_ROOT_CAUSE
    assert risk["source_table"] == STACK_EVENT_SOURCE_TABLE
    assert risk["implementation_module"] == STACK_CONTEXT_IMPLEMENTATION_MODULE
    assert risk["raw_events_are_source_data_not_policy_features"] is True
    assert risk["target_action_stack_delta_is_label_context_not_feature"] is True
    assert risk["current_delivery_blocker"] is False
    assert risk["model_quality_risk"] is True
    assert risk["final_strategy_quality_claim_blocker_without_explicit_stack_context"] is True
    assert set(risk["derivation_policy"]) == set(STACK_CONTEXT_DERIVATION_POLICY)
    for context_name in ["pot", "effective_stack", "spr", "bet_size", "pressure"]:
        policy = risk["derivation_policy"][context_name]
        assert policy["required_semantics"]
        assert policy["source"]
        assert policy["target_action_delta_allowed"] is False
        assert policy["derived_features"]
    assert payload["raw_stack_event_boundary"]["raw_stack_events_are_direct_policy_features"] is False
    assert payload["raw_stack_event_boundary"]["decision_time_derivation_required"] is True
    assert payload["derived_context_mitigation"]["status"] == "IMPLEMENTED_FROM_PRE_ACTION_STACK_DELTAS"
    assert payload["derived_context_mitigation"]["implementation_module"] == STACK_CONTEXT_IMPLEMENTATION_MODULE
    assert payload["derived_context_mitigation"]["pre_action_event_derivation_helper"] == STACK_CONTEXT_PRE_ACTION_HELPER
    assert payload["derived_context_mitigation"]["uses_target_action_stack_delta_as_feature"] is False
    assert payload["training_feature_audit"]["status"] == "PASS"
    assert "reconstructed_pot" in payload["training_feature_audit"]["required_stack_context_features_present"]
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
    assert contract["implementation_module"] == STACK_CONTEXT_IMPLEMENTATION_MODULE
    assert contract["pre_action_event_derivation_helper"] == STACK_CONTEXT_PRE_ACTION_HELPER
    assert "reconstructed_pot" in contract["required_derived_features"]
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
        "risk_contract": {
            "risk_id": "wrong",
            "root_cause": "wrong",
            "source_table": "stack_events.csv",
            "implementation_module": "wrong",
            "raw_events_are_source_data_not_policy_features": False,
            "target_action_stack_delta_is_label_context_not_feature": False,
            "current_delivery_blocker": False,
            "model_quality_risk": False,
            "final_strategy_quality_claim_blocker_without_explicit_stack_context": False,
            "derivation_policy": {
                "pot": {
                    "required_semantics": "",
                    "source": "",
                    "target_action_delta_allowed": True,
                    "derived_features": ["unknown_feature"],
                }
            },
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
            "implementation_module": "wrong",
            "pre_action_event_derivation_helper": "wrong",
            "uses_target_action_stack_delta_as_feature": True,
            "target_action_stack_delta_leakage_guard": False,
            "uses_post_hand_outcome_fields": True,
            "current_delivery_blocker": False,
            "model_quality_risk": False,
            "final_strategy_quality_claim_blocker_without_explicit_stack_context": False,
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
    assert "stack_event_risk_id_must_match_contract" in invariants["violations"]
    assert "stack_context_implementation_module_must_be_explicit" in invariants["violations"]
    assert "stack_events_must_be_source_data_not_policy_features" in invariants["violations"]
    assert "target_action_stack_delta_must_be_label_context_not_feature" in invariants["violations"]
    assert "stack_context_derivation_policy_must_cover_required_context" in invariants["violations"]
    assert "stack_context_policy_must_forbid_target_delta_for_pot" in invariants["violations"]
    assert "raw_stack_events_must_not_be_direct_policy_features" in invariants["violations"]
    assert "stack_events_must_require_decision_time_derivation" in invariants["violations"]
    assert "target_action_stack_delta_must_not_be_feature" in invariants["violations"]
    assert "derived_stack_context_must_not_use_target_action_delta" in invariants["violations"]
    assert "derived_stack_context_must_reference_stack_context_module" in invariants["violations"]
    assert "derived_stack_context_must_reference_pre_action_event_helper" in invariants["violations"]
    assert "target_action_stack_delta_leakage_guard_must_be_enabled" in invariants["violations"]
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
    assert second_features["reconstructed_pot"] == 4.0
    assert second_features["reconstructed_effective_stack"] == 100.0
    assert second_features["reconstructed_current_street_bet_size"] == 4.0
    assert second_features["reconstructed_call_pressure"] > 0.0
    assert second_features["reconstructed_spr_after_call"] > 0.0


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
