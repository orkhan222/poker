from __future__ import annotations

HAND_FIELDS = (
    "hand_id",
    "hand_index",
    "local_hand_index",
    "source_file",
    "table_id",
    "game_type",
    "small_blind",
    "big_blind",
    "ante",
    "button_position",
    "start_frame",
    "end_frame",
    "board_cards",
    "total_actions",
    "total_stack_events",
    "winner_positions",
    "pot_from_stacks",
    "pot_from_recognition",
    "dealer_hand_number",
    "dealer_winner",
    "dealer_pot",
)

PLAYER_FIELDS = (
    "hand_id",
    "hand_index",
    "local_hand_index",
    "source_file",
    "position",
    "nickname",
    "cards",
    "starting_stack",
    "ending_stack",
    "stack_delta",
)

ACTION_FIELDS = (
    "hand_id",
    "hand_index",
    "local_hand_index",
    "source_file",
    "table_id",
    "game_type",
    "small_blind",
    "big_blind",
    "ante",
    "button_position",
    "frame_id",
    "player_position",
    "player_nickname",
    "action",
    "action_amount",
    "pot_before_action",
    "pot_after_action",
    "legal_actions",
    "ocr_confidence",
    "street",
)

STACK_FIELDS = (
    "hand_id",
    "hand_index",
    "local_hand_index",
    "source_file",
    "frame_id",
    "player_position",
    "event",
    "stack",
    "diff",
    "stack_after_event",
)

DATASET_SCHEMA_REQUIRED_FIELDS = {
    "hands.csv": (
        "table_id",
        "game_type",
        "small_blind",
        "big_blind",
        "ante",
        "button_position",
    ),
    "actions.csv": (
        "table_id",
        "game_type",
        "small_blind",
        "big_blind",
        "ante",
        "button_position",
        "action_amount",
        "pot_before_action",
        "pot_after_action",
        "legal_actions",
        "ocr_confidence",
    ),
}


def missing_dataset_fields(filename: str, observed_fields: set[str]) -> list[str]:
    required = DATASET_SCHEMA_REQUIRED_FIELDS.get(filename, ())
    return sorted(field for field in required if field not in observed_fields)
