from __future__ import annotations

import csv
from pathlib import Path

import pytest

from poker_agent.action_normalization import assert_canonical_decision_action
from poker_agent.action_normalization import normalize_action, normalize_action_record, normalize_action_result
from poker_agent.features import load_training_examples
from poker_agent.normalized_action_contract import (
    CANONICAL_ACTIONS,
    build_normalized_action_contract,
    validate_normalized_action_contract,
    write_normalized_action_contract,
)


def test_noisy_ocr_actions_normalize_to_canonical_labels() -> None:
    cases = {
        "ra1se": "raise",
        "Plyr3 ra1se $4.50": "raise",
        "P1ayer7 ra1sed 4.50": "raise",
        "cail": "call",
        "ca1l": "call",
        "bett": "bet",
        "bettt": "bet",
        "all-in": "all_in",
        "a11-in": "all_in",
        "all in": "all_in",
        "checks": "check",
        "f0ld": "fold",
    }

    for raw, expected in cases.items():
        result = normalize_action_result(raw)
        assert result.canonical_action == expected
        assert result.is_decision_action is True
        assert normalize_action(raw) == expected


def test_raw_ocr_action_labels_are_rejected_without_normalization() -> None:
    with pytest.raises(ValueError, match="Non-canonical action label"):
        assert_canonical_decision_action("ra1se", context="unit test")

    assert assert_canonical_decision_action("raise", context="unit test") == "raise"


def test_action_record_enrichment_keeps_raw_label_and_adds_canonical_label() -> None:
    row = {
        "hand_id": "h1",
        "frame_id": "17",
        "player_position": "BTN",
        "action": "Plyr3 ra1se $4.50",
        "street": "preflop",
    }

    normalized = normalize_action_record(row)

    assert normalized["action"] == "Plyr3 ra1se $4.50"
    assert normalized["raw_action"] == "Plyr3 ra1se $4.50"
    assert normalized["canonical_action"] == "raise"
    assert normalized["action_normalization_status"] == "canonical"
    assert normalized["action_normalization_method"] in {
        "phrase",
        "token_alias",
        "ocr_digit_alias",
        "ocr_digit_canonical",
        "fuzzy",
    }
    assert normalized["action_normalization_confidence"] > 0.0
    assert normalized["is_decision_action"] is True


def test_training_examples_emit_only_canonical_action_labels(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "actions.csv",
        ["hand_id", "frame_id", "player_position", "action", "street"],
        [
            {"hand_id": "h1", "frame_id": "1", "player_position": "BTN", "action": "ra1se", "street": "preflop"},
            {"hand_id": "h1", "frame_id": "2", "player_position": "BB", "action": "cail", "street": "preflop"},
            {"hand_id": "h1", "frame_id": "3", "player_position": "BTN", "action": "bett", "street": "flop"},
            {"hand_id": "h1", "frame_id": "4", "player_position": "BB", "action": "all-in", "street": "flop"},
        ],
    )
    _write_csv(
        tmp_path / "players.csv",
        ["hand_id", "position", "cards", "starting_stack"],
        [
            {"hand_id": "h1", "position": "BTN", "cards": "AS KD", "starting_stack": "100"},
            {"hand_id": "h1", "position": "BB", "cards": "QS QH", "starting_stack": "100"},
        ],
    )
    _write_csv(tmp_path / "hands.csv", ["hand_id", "board_cards"], [{"hand_id": "h1", "board_cards": "2C 7D QS"}])
    _write_csv(
        tmp_path / "stack_events.csv",
        ["hand_id", "frame_id", "player_position", "diff"],
        [
            {"hand_id": "h1", "frame_id": "1", "player_position": "BTN", "diff": "-4"},
            {"hand_id": "h1", "frame_id": "2", "player_position": "BB", "diff": "-4"},
            {"hand_id": "h1", "frame_id": "3", "player_position": "BTN", "diff": "-6"},
            {"hand_id": "h1", "frame_id": "4", "player_position": "BB", "diff": "-40"},
        ],
    )

    examples = load_training_examples(
        tmp_path,
        require_hole_cards=False,
        missing_hole_cards="flag",
        merge_all_in=False,
    )
    labels = [label for _, label in examples]

    assert labels == ["raise", "call", "bet", "all_in"]


def test_normalized_action_contract_passes_for_project_dataset() -> None:
    project_root = Path(__file__).resolve().parents[1]

    payload = build_normalized_action_contract(project_root, max_rows=1000)

    assert payload["overall_status"] == "PASS"
    assert set(payload["canonical_actions"]) == set(CANONICAL_ACTIONS)
    assert payload["raw_ocr_action_must_not_be_training_label"] is True
    assert payload["actions_csv_audit"]["action_column_present"] is True
    assert payload["actions_csv_audit"]["canonical_decision_rows"] > 0
    assert payload["training_label_audit"]["status"] == "PASS"
    assert payload["training_label_audit"]["invalid_labels"] == []


def test_normalized_action_contract_rejects_false_raw_label_claim() -> None:
    payload = {
        "normalized_action_status": "MISSING",
        "raw_action_source_status": "CANONICAL",
        "canonical_actions": ["fold", "call"],
        "raw_ocr_action_must_not_be_training_label": False,
        "normalization_required_before_training": False,
        "normalization_required_before_evaluation": False,
        "normalization_required_before_policy_comparison": False,
        "current_delivery_blocker": False,
        "model_quality_risk": False,
        "actions_csv_audit": {
            "status": "PASS",
            "action_column_present": True,
            "rows_scanned": 10,
            "canonical_decision_rows": 0,
        },
        "training_label_audit": {
            "status": "FAIL",
            "invalid_labels": ["ra1se"],
        },
        "noisy_action_examples": [
            {"raw_action": "ra1se", "expected": "raise", "observed": "ra1se", "passed": False}
        ],
    }

    invariants = validate_normalized_action_contract(payload)

    assert invariants["status"] == "FAIL"
    assert "normalized_action_status_must_be_implemented" in invariants["violations"]
    assert "raw_ocr_action_must_not_be_training_label" in invariants["violations"]
    assert "training_labels_must_not_contain_raw_ocr_actions" in invariants["violations"]
    assert "noisy_action_example_must_normalize:ra1se" in invariants["violations"]


def test_write_normalized_action_contract_outputs_reports(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "data" / "actions.csv",
        ["hand_id", "frame_id", "player_position", "action", "street"],
        [
            {"hand_id": "h1", "frame_id": "1", "player_position": "BTN", "action": "ra1se", "street": "preflop"},
            {"hand_id": "h1", "frame_id": "2", "player_position": "BB", "action": "cail", "street": "preflop"},
        ],
    )
    _write_csv(
        tmp_path / "data" / "players.csv",
        ["hand_id", "position", "cards", "starting_stack"],
        [
            {"hand_id": "h1", "position": "BTN", "cards": "AS KD", "starting_stack": "100"},
            {"hand_id": "h1", "position": "BB", "cards": "QS QH", "starting_stack": "100"},
        ],
    )
    _write_csv(tmp_path / "data" / "hands.csv", ["hand_id", "board_cards"], [{"hand_id": "h1", "board_cards": ""}])
    _write_csv(
        tmp_path / "data" / "stack_events.csv",
        ["hand_id", "frame_id", "player_position", "diff"],
        [
            {"hand_id": "h1", "frame_id": "1", "player_position": "BTN", "diff": "-4"},
            {"hand_id": "h1", "frame_id": "2", "player_position": "BB", "diff": "-4"},
        ],
    )

    payload = write_normalized_action_contract(
        tmp_path,
        tmp_path / "reports" / "normalized_action_contract.json",
        tmp_path / "reports" / "normalized_action_contract.md",
    )

    assert payload["overall_status"] == "PASS"
    assert (tmp_path / "reports" / "normalized_action_contract.json").exists()
    assert "Raw OCR/dealer action text must be normalized" in (
        tmp_path / "reports" / "normalized_action_contract.md"
    ).read_text(encoding="utf-8")


def test_normalized_action_contract_endpoint_returns_contract() -> None:
    from poker_agent.service import normalized_action_contract_json

    payload = normalized_action_contract_json()

    assert payload["overall_status"] == "PASS"
    assert payload["normalized_action_status"] == "IMPLEMENTED"
    assert payload["raw_ocr_action_must_not_be_training_label"] is True


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
