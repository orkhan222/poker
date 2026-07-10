from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


FORBIDDEN_OUTCOME_FIELDS = (
    "winner_positions",
    "stack_delta",
    "ending_stack",
    "dealer_winner",
    "dealer_pot",
    "pot_from_stacks",
)

RAW_FINAL_BOARD_SNAPSHOT_FIELDS = (
    "hands.csv::board_cards",
)

VISIBLE_BOARD_COUNTS_BY_STREET = {
    "preflop": 0,
    "flop": 3,
    "turn": 4,
    "river": 5,
}

OUTCOME_FIELD_DEFINITIONS: dict[str, dict[str, str]] = {
    "winner_positions": {
        "source_table": "hands.csv",
        "availability": "post_hand",
        "reason": "Known only after the hand result is reconstructed.",
    },
    "dealer_winner": {
        "source_table": "hands.csv",
        "availability": "post_hand",
        "reason": "Dealer result text is emitted after the hand outcome.",
    },
    "dealer_pot": {
        "source_table": "hands.csv",
        "availability": "post_hand",
        "reason": "Payout amount is known after settlement, not before the target action.",
    },
    "ending_stack": {
        "source_table": "players.csv",
        "availability": "post_hand",
        "reason": "Final stack is observed after the hand has completed.",
    },
    "stack_delta": {
        "source_table": "players.csv",
        "availability": "post_hand",
        "reason": "Win/loss delta is an outcome-derived settlement field.",
    },
    "pot_from_stacks": {
        "source_table": "hands.csv",
        "availability": "post_hand_reconstruction",
        "reason": "Computed from final stack movements and therefore includes future outcome information.",
    },
}

FINAL_BOARD_FIELD_DEFINITIONS: dict[str, dict[str, str]] = {
    "hands.csv::board_cards": {
        "source_table": "hands.csv",
        "field": "board_cards",
        "availability": "post_hand_final_snapshot",
        "reason": (
            "The hands.csv board_cards value is the final community-card snapshot. "
            "For preflop, flop, and turn decisions it may include cards that were not visible yet."
        ),
    },
}


def forbidden_outcome_feature_names(feature_names: Iterable[str]) -> list[str]:
    return sorted(
        {
            name
            for name in (str(item) for item in feature_names)
            for forbidden in FORBIDDEN_OUTCOME_FIELDS
            if forbidden in name
        }
    )


def assert_no_outcome_feature_leakage(
    features_or_names: Mapping[str, Any] | Iterable[str],
    *,
    context: str,
) -> None:
    if isinstance(features_or_names, Mapping):
        feature_names = features_or_names.keys()
    else:
        feature_names = features_or_names
    detected = forbidden_outcome_feature_names(feature_names)
    if detected:
        joined = ", ".join(detected)
        raise ValueError(
            f"Outcome-field data leakage detected in {context}: {joined}. "
            "Only decision-time observable features may be used for training or prediction."
        )


def assert_no_final_board_snapshot_leakage(
    source_fields: Iterable[str],
    *,
    context: str,
) -> None:
    detected = sorted(set(str(field) for field in source_fields) & set(RAW_FINAL_BOARD_SNAPSHOT_FIELDS))
    if detected:
        joined = ", ".join(detected)
        raise ValueError(
            f"Final-board data leakage detected in {context}: {joined}. "
            "Use only community cards visible before the target action; final hand snapshots are audit data."
        )


def visible_board_count_for_street(street: str) -> int:
    normalized = str(street or "preflop").lower()
    if normalized not in VISIBLE_BOARD_COUNTS_BY_STREET:
        raise ValueError(f"Unsupported poker street for board visibility in leakage guard: {street!r}")
    return VISIBLE_BOARD_COUNTS_BY_STREET[normalized]


def truncate_final_board_snapshot_for_decision(
    board_cards: Iterable[str],
    street: str,
) -> list[str]:
    """Convert a final board snapshot into the board visible at the decision street."""
    visible_count = visible_board_count_for_street(street)
    cards = [str(card).strip() for card in board_cards if str(card).strip()]
    return cards[:visible_count]


def assert_board_cards_visible_for_street(
    board_cards: Iterable[str],
    street: str,
    *,
    context: str,
) -> None:
    """Reject request/training board cards that contain future community cards."""
    cards = [str(card).strip() for card in board_cards if str(card).strip()]
    max_visible = visible_board_count_for_street(street)
    if len(cards) > max_visible:
        raise ValueError(
            f"Final-board data leakage detected in {context}: street={street!r} allows at most "
            f"{max_visible} visible board cards, received {len(cards)}. "
            "Use only community cards visible at the submitted decision street."
        )
