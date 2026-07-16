from __future__ import annotations

import csv
from pathlib import Path
from tempfile import TemporaryDirectory

from poker_agent.data_validation import MISSING_OCR_CONFLICT_POLICY, validate_dataset
from poker_agent.dataset_schema import ACTION_FIELDS, HAND_FIELDS, PLAYER_FIELDS, STACK_FIELDS


def test_clean_dataset_validation_passes() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        write_validation_fixture(root, broken=False)

        result = validate_dataset(root)

    assert result["status"] == "PASS"
    assert result["checks"]["pot_conservation"]["status"] == "PASS"
    assert result["checks"]["stack_delta_consistency"]["status"] == "PASS"
    assert result["checks"]["duplicate_hand_detection"]["status"] == "PASS"
    assert result["checks"]["missing_ocr_conflict_policy"]["status"] == "PASS"
    assert result["policy"] == MISSING_OCR_CONFLICT_POLICY


def test_broken_dataset_validation_fails_all_contract_checks() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        write_validation_fixture(root, broken=True)

        result = validate_dataset(root)

    assert result["status"] == "FAIL"
    assert result["checks"]["pot_conservation"]["status"] == "FAIL"
    assert result["checks"]["stack_delta_consistency"]["status"] == "FAIL"
    assert result["checks"]["duplicate_hand_detection"]["status"] == "FAIL"
    assert result["checks"]["missing_ocr_conflict_policy"]["status"] == "FAIL"
    assert result["checks"]["pot_conservation"]["violation_rows"] >= 1
    assert result["checks"]["stack_delta_consistency"]["player_delta_violation_rows"] >= 1
    assert result["checks"]["duplicate_hand_detection"]["duplicate_groups"]["hand_id"]
    assert result["checks"]["missing_ocr_conflict_policy"]["conflict_groups"] >= 1
    assert result["checks"]["missing_ocr_conflict_policy"]["missing_chip_action_amount_rows"] >= 1


def write_validation_fixture(root: Path, *, broken: bool) -> None:
    hand_rows = [
        {
            "hand_id": "H1",
            "hand_index": "0",
            "local_hand_index": "0",
            "source_file": "sample",
            "table_id": "table_1",
            "game_type": "nl_holdem",
            "small_blind": "0.5",
            "big_blind": "1.0",
            "ante": "0",
            "button_position": "BTN",
            "start_frame": "1",
            "end_frame": "5",
            "board_cards": "",
            "total_actions": "1",
            "total_stack_events": "2",
            "winner_positions": "BB",
            "pot_from_stacks": "2.5",
            "pot_from_recognition": "2.5",
            "dealer_hand_number": "",
            "dealer_winner": "",
            "dealer_pot": "",
        }
    ]
    if broken:
        hand_rows.append(dict(hand_rows[0]))

    player_rows = [
        {
            "hand_id": "H1",
            "hand_index": "0",
            "local_hand_index": "0",
            "source_file": "sample",
            "position": "BTN",
            "nickname": "hero",
            "cards": "Ah Kd",
            "starting_stack": "100",
            "ending_stack": "97.5",
            "stack_delta": "-8.0" if broken else "-2.5",
        },
        {
            "hand_id": "H1",
            "hand_index": "0",
            "local_hand_index": "0",
            "source_file": "sample",
            "position": "BB",
            "nickname": "villain",
            "cards": "",
            "starting_stack": "100",
            "ending_stack": "102.5",
            "stack_delta": "2.5",
        },
    ]
    action_rows = [
        {
            "hand_id": "H1",
            "hand_index": "0",
            "local_hand_index": "0",
            "source_file": "sample",
            "table_id": "table_1",
            "game_type": "nl_holdem",
            "small_blind": "0.5",
            "big_blind": "1.0",
            "ante": "0",
            "button_position": "BTN",
            "frame_id": "3",
            "player_position": "BTN",
            "player_nickname": "hero",
            "action": "call",
            "action_amount": "" if broken else "2.5",
            "pot_before_action": "0",
            "pot_after_action": "9.0" if broken else "2.5",
            "legal_actions": "" if broken else "fold call raise all_in",
            "ocr_confidence": "" if broken else "0.95",
            "street": "preflop",
        }
    ]
    if broken:
        conflict = dict(action_rows[0])
        conflict["action"] = "raise"
        conflict["action_amount"] = "6.0"
        action_rows.append(conflict)

    stack_rows = [
        {
            "hand_id": "H1",
            "hand_index": "0",
            "local_hand_index": "0",
            "source_file": "sample",
            "frame_id": "3",
            "player_position": "BTN",
            "event": "update_stack",
            "stack": "97.5",
            "diff": "-2.5",
            "stack_after_event": "97.5",
        },
        {
            "hand_id": "H1",
            "hand_index": "0",
            "local_hand_index": "0",
            "source_file": "sample",
            "frame_id": "4",
            "player_position": "BB",
            "event": "update_stack",
            "stack": "102.5",
            "diff": "2.5",
            "stack_after_event": "102.5",
        },
    ]
    write_csv(root / "hands.csv", HAND_FIELDS, hand_rows)
    write_csv(root / "players.csv", PLAYER_FIELDS, player_rows)
    write_csv(root / "actions.csv", ACTION_FIELDS, action_rows)
    write_csv(root / "stack_events.csv", STACK_FIELDS, stack_rows)


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
