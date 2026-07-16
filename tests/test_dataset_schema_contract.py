from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from build_poker_dataset_optimized import ACTION_FIELDS, HAND_FIELDS, process_file, rows_for_hand
from poker_agent.dataset_schema import DATASET_SCHEMA_REQUIRED_FIELDS


def test_dataset_schema_fields_are_declared() -> None:
    assert set(DATASET_SCHEMA_REQUIRED_FIELDS["hands.csv"]).issubset(set(HAND_FIELDS))
    assert set(DATASET_SCHEMA_REQUIRED_FIELDS["actions.csv"]).issubset(set(ACTION_FIELDS))


def test_builder_emits_table_and_action_context_columns() -> None:
    events = [
        {
            "frame_id": 1,
            "event_name": "recognize_pot",
            "object_type": "table",
            "table_id": "table_7",
            "game_type": "nl_holdem",
            "event_value": {"pot": 10.0},
        },
        {
            "frame_id": 2,
            "event_name": "dealer_message",
            "object_type": "table",
            "event_value": {"text": "dealer limits going up blinds 0.5001.000 ante 0.100"},
        },
        {
            "frame_id": 3,
            "event_name": "ocr_action",
            "object_type": "player",
            "event_value": {
                "player_position": "BTN",
                "value": "call",
                "amount": 2.5,
                "confidence": 0.88,
                "button_position": "BTN",
                "legal_actions": ["fold", "call", "raise", "all_in"],
            },
        },
        {
            "frame_id": 4,
            "event_name": "recognize_cards",
            "object_type": "table",
            "event_value": {"value": "remove_cards"},
        },
    ]

    with TemporaryDirectory() as temp_dir:
        source = Path(temp_dir) / "sample_table.jsonl"
        source.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")

        hands = list(process_file(source))

    assert len(hands) == 1
    hand_row, _player_rows, action_rows, _stack_rows = rows_for_hand(hands[0], 0)

    assert hand_row["table_id"] == "table_7"
    assert hand_row["game_type"] == "nl_holdem"
    assert hand_row["small_blind"] == 0.5
    assert hand_row["big_blind"] == 1.0
    assert hand_row["ante"] == 0.1
    assert hand_row["button_position"] == "BTN"

    assert len(action_rows) == 1
    action = action_rows[0]
    assert action["table_id"] == "table_7"
    assert action["game_type"] == "nl_holdem"
    assert action["small_blind"] == 0.5
    assert action["big_blind"] == 1.0
    assert action["ante"] == 0.1
    assert action["button_position"] == "BTN"
    assert action["action_amount"] == 2.5
    assert action["pot_before_action"] == 10.0
    assert action["pot_after_action"] == 12.5
    assert action["legal_actions"] == "fold call raise all_in"
    assert action["ocr_confidence"] == 0.88
