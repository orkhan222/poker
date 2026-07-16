from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from poker_agent.action_space import CANONICAL_ACTIONS
from poker_agent.features import public_context_features
from poker_agent.model import SoftmaxPolicy


FeatureExample = tuple[dict[str, float], str]
ExampleTransform = Callable[[list[FeatureExample]], list[FeatureExample]]


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    family: str
    trains_on_dataset: bool
    uses_private_cards: bool
    description: str


BASELINE_SPECS: dict[str, BaselineSpec] = {
    "rule": BaselineSpec(
        name="rule",
        family="deterministic_rules",
        trains_on_dataset=False,
        uses_private_cards=True,
        description="Hard-coded poker heuristics using hand-strength proxy, pot odds, SPR, and pressure features.",
    ),
    "imitation_learning": BaselineSpec(
        name="imitation_learning",
        family="behavior_cloning_softmax",
        trains_on_dataset=True,
        uses_private_cards=False,
        description="Supervised behavior cloning from human action labels using public/context features only.",
    ),
    "llm": BaselineSpec(
        name="llm",
        family="offline_prompt_policy",
        trains_on_dataset=False,
        uses_private_cards=False,
        description="LLM prompt baseline contract with deterministic local prompt-policy fallback for offline evaluation.",
    ),
    "end_to_end_policy": BaselineSpec(
        name="end_to_end_policy",
        family="full_state_softmax_policy",
        trains_on_dataset=True,
        uses_private_cards=True,
        description="End-to-end supervised policy baseline over the full engineered state feature vector.",
    ),
}


def list_baseline_specs() -> list[BaselineSpec]:
    return [BASELINE_SPECS[name] for name in ("rule", "imitation_learning", "llm", "end_to_end_policy")]


def baseline_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in list_baseline_specs())


def transform_examples_for_baseline(name: str, examples: list[FeatureExample]) -> list[FeatureExample]:
    _require_known_baseline(name)
    if name in {"imitation_learning", "llm"}:
        return [(public_context_features(features), label) for features, label in examples]
    return examples


def build_baseline_policy(
    name: str,
    train_examples: list[FeatureExample] | None = None,
    *,
    epochs: int = 12,
    learning_rate: float = 0.05,
    class_weighting: str = "sqrt_balanced",
    max_class_weight: float = 6.0,
) -> Any:
    _require_known_baseline(name)
    if name == "rule":
        return RuleBaselinePolicy()
    if name == "llm":
        return LLMPromptBaselinePolicy()
    if train_examples is None:
        raise ValueError(f"Baseline `{name}` requires training examples")
    examples = transform_examples_for_baseline(name, train_examples)
    if not examples:
        raise ValueError(f"Baseline `{name}` received no training examples")
    policy = SoftmaxPolicy()
    policy.fit(
        examples,
        epochs=epochs,
        learning_rate=learning_rate,
        class_weighting=class_weighting,
        max_class_weight=max_class_weight,
    )
    spec = BASELINE_SPECS[name]
    policy.metadata = {
        "baseline": spec.name,
        "baseline_family": spec.family,
        "uses_private_cards": spec.uses_private_cards,
        "description": spec.description,
    }
    return policy


class RuleBaselinePolicy:
    labels = list(CANONICAL_ACTIONS)
    metadata = {
        "baseline": "rule",
        "baseline_family": "deterministic_rules",
        "uses_private_cards": True,
    }

    def predict_from_features(self, features: dict[str, float]) -> tuple[str, dict[str, float]]:
        strength = float(features.get("strength_proxy", 0.0))
        to_call = float(features.get("to_call", features.get("amount_to_call", 0.0)))
        pot_odds = float(features.get("pot_odds", 0.0))
        spr = float(features.get("spr", 0.0))
        facing_pressure = to_call > 0 or float(features.get("facing_bet_or_raise", 0.0)) > 0

        if not facing_pressure:
            if strength >= 0.62 and spr >= 3.0:
                probabilities = _distribution(bet=0.46, check=0.30, raise_=0.10, call=0.05, fold=0.04, all_in=0.05)
            else:
                probabilities = _distribution(check=0.70, bet=0.14, call=0.05, fold=0.04, raise_=0.03, all_in=0.04)
        elif strength >= 0.72:
            probabilities = _distribution(raise_=0.44, call=0.28, all_in=0.10, fold=0.08, bet=0.05, check=0.05)
        elif strength >= 0.45 or pot_odds <= 0.22:
            probabilities = _distribution(call=0.48, fold=0.25, raise_=0.12, all_in=0.05, bet=0.05, check=0.05)
        else:
            probabilities = _distribution(fold=0.66, call=0.20, raise_=0.04, all_in=0.03, bet=0.03, check=0.04)
        return _select(probabilities)

    def predict_batch_from_features(self, feature_rows: list[dict[str, float]]) -> list[tuple[str, dict[str, float]]]:
        return [self.predict_from_features(features) for features in feature_rows]


class LLMPromptBaselinePolicy:
    labels = list(CANONICAL_ACTIONS)
    metadata = {
        "baseline": "llm",
        "baseline_family": "offline_prompt_policy",
        "uses_private_cards": False,
        "provider": "local_deterministic_prompt_policy",
    }

    def build_prompt(self, features: dict[str, float]) -> str:
        street = _active_flag(features, "street=", default="unknown")
        position = _active_flag(features, "position_group=", default="unknown")
        return (
            "Choose one poker action from fold/check/call/bet/raise/all_in. "
            f"street={street}; position={position}; "
            f"pot_odds={float(features.get('pot_odds', 0.0)):.3f}; "
            f"spr={float(features.get('spr', 0.0)):.3f}; "
            f"to_call={float(features.get('to_call', features.get('amount_to_call', 0.0))):.3f}; "
            f"aggression={float(features.get('street_aggression_ratio', 0.0)):.3f}."
        )

    def predict_from_features(self, features: dict[str, float]) -> tuple[str, dict[str, float]]:
        _ = self.build_prompt(features)
        pot_odds = float(features.get("pot_odds", 0.0))
        aggression = float(features.get("street_aggression_ratio", 0.0))
        to_call = float(features.get("to_call", features.get("amount_to_call", 0.0)))
        spr = float(features.get("spr", 0.0))

        if to_call <= 0:
            probabilities = _distribution(check=0.58, bet=0.24, raise_=0.06, call=0.04, fold=0.04, all_in=0.04)
        elif pot_odds <= 0.18 and aggression <= 0.45:
            probabilities = _distribution(call=0.45, raise_=0.18, fold=0.18, all_in=0.07, bet=0.06, check=0.06)
        elif spr <= 3.0 and pot_odds <= 0.28:
            probabilities = _distribution(call=0.40, all_in=0.18, raise_=0.15, fold=0.17, bet=0.05, check=0.05)
        else:
            probabilities = _distribution(fold=0.52, call=0.28, raise_=0.07, all_in=0.04, bet=0.04, check=0.05)
        return _select(probabilities)

    def predict_batch_from_features(self, feature_rows: list[dict[str, float]]) -> list[tuple[str, dict[str, float]]]:
        return [self.predict_from_features(features) for features in feature_rows]


def _distribution(
    *,
    fold: float,
    check: float,
    call: float,
    bet: float,
    raise_: float,
    all_in: float,
) -> dict[str, float]:
    raw = {
        "fold": fold,
        "check": check,
        "call": call,
        "bet": bet,
        "raise": raise_,
        "all_in": all_in,
    }
    total = sum(max(0.0, value) for value in raw.values()) or 1.0
    return {action: max(0.0, raw[action]) / total for action in CANONICAL_ACTIONS}


def _select(probabilities: dict[str, float]) -> tuple[str, dict[str, float]]:
    return max(probabilities, key=probabilities.get), probabilities


def _active_flag(features: dict[str, float], prefix: str, *, default: str) -> str:
    for name, value in sorted(features.items()):
        if name.startswith(prefix) and float(value) > 0:
            return name[len(prefix) :]
    return default


def _require_known_baseline(name: str) -> None:
    if name not in BASELINE_SPECS:
        raise ValueError(f"Unknown baseline `{name}`. Expected one of: {', '.join(baseline_names())}")
