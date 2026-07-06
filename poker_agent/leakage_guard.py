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
