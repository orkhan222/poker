from __future__ import annotations

from typing import Any

from poker_agent.evaluator import evaluate_policy

Example = tuple[dict[str, float], str]


def slice_examples(examples: list[Example]) -> dict[str, list[Example]]:
    slices: dict[str, list[Example]] = {
        "all": examples,
        "observed_hole_cards": [],
        "missing_hole_cards": [],
        "preflop": [],
        "postflop": [],
        "facing_bet": [],
        "not_facing_bet": [],
        "short_stack": [],
        "deep_stack": [],
        "high_aggression": [],
        "low_aggression": [],
    }
    for features, label in examples:
        target = (features, label)
        if features.get("hole_card_observed_ratio", 0.0) >= 1.0:
            slices["observed_hole_cards"].append(target)
        else:
            slices["missing_hole_cards"].append(target)
        if features.get("street_index", 0.0) == 0.0:
            slices["preflop"].append(target)
        else:
            slices["postflop"].append(target)
        if features.get("facing_bet_or_raise", features.get("has_call", 0.0)) > 0:
            slices["facing_bet"].append(target)
        else:
            slices["not_facing_bet"].append(target)
        if features.get("spr", 0.0) <= 6.0:
            slices["short_stack"].append(target)
        else:
            slices["deep_stack"].append(target)
        if features.get("street_aggression_ratio", 0.0) >= 0.35:
            slices["high_aggression"].append(target)
        else:
            slices["low_aggression"].append(target)
    return slices


def evaluate_policy_slices(
    model: Any,
    examples: list[Example],
    *,
    min_examples: int = 100,
) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for name, rows in slice_examples(examples).items():
        if len(rows) < min_examples:
            continue
        metrics[name] = evaluate_policy(model, rows)
    return metrics
