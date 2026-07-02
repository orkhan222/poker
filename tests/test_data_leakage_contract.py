from __future__ import annotations

import csv
from pathlib import Path

from poker_agent.data_leakage_contract import (
    build_data_leakage_contract,
    validate_data_leakage_contract,
)
from poker_agent.features import load_training_examples
from poker_agent.api_contract import api_contract


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


def test_data_leakage_contract_is_exposed_through_api_contract() -> None:
    from poker_agent.service import data_leakage_contract_json

    contract = api_contract()["data_leakage_contract"]
    payload = data_leakage_contract_json()

    assert contract["endpoint"] == "/data-leakage-contract.json"
    assert "ending_stack" in contract["forbidden_outcome_fields"]
    assert payload["overall_status"] == "PASS"
    assert payload["leakage_boundary"]["training_feature_use_allowed"] is False


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
    }

    invariants = validate_data_leakage_contract(payload)

    assert invariants["status"] == "FAIL"
    assert "outcome_fields_must_not_be_training_features" in invariants["violations"]
    assert "forbidden_training_feature_names_detected" in invariants["violations"]
    assert "forbidden_model_artifact_features_detected" in invariants["violations"]
    assert "forbidden_outcome_fields_used_in_guarded_source" in invariants["violations"]


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
