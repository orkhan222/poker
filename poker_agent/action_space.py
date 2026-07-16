from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


CANONICAL_ACTIONS = ("fold", "check", "call", "bet", "raise", "all_in")
AGGRESSIVE_ACTIONS = ("bet", "raise", "all_in")

ACTION_ALIASES = {
    "all-in": "all_in",
    "all in": "all_in",
    "allin": "all_in",
    "jam": "all_in",
    "shove": "all_in",
    "calls": "call",
    "called": "call",
    "checks": "check",
    "checked": "check",
    "bets": "bet",
    "betted": "bet",
    "raises": "raise",
    "raised": "raise",
    "folds": "fold",
    "folded": "fold",
}


@dataclass(frozen=True)
class ActionSpace:
    legal_actions: tuple[str, ...]
    min_raise_to: float
    max_raise_to: float
    min_raise_by: float
    max_raise_by: float
    all_in_amount: float
    to_call: float
    stack: float

    @classmethod
    def from_state(
        cls,
        raw: dict[str, Any],
        *,
        to_call: float,
        stack: float,
        min_raise: float,
    ) -> "ActionSpace":
        to_call = _nonnegative_float(to_call)
        stack = _nonnegative_float(stack)
        min_raise_by = _first_nonnegative(
            raw,
            ("min_raise_by", "min_legal_raise_by", "minimum_raise_by"),
            default=min_raise,
        )
        all_in_amount = _first_nonnegative(
            raw,
            ("all_in_amount", "all_in", "effective_stack", "max_commitment"),
            default=stack,
        )
        max_raise_to = _first_nonnegative(
            raw,
            ("max_raise_to", "max_legal_raise_to", "maximum_raise_to", "max_raise"),
            default=all_in_amount,
        )
        max_raise_to = min(max_raise_to, all_in_amount) if all_in_amount > 0 else max_raise_to
        min_raise_to = _first_nonnegative(
            raw,
            ("min_raise_to", "min_legal_raise_to", "minimum_raise_to"),
            default=(to_call + min_raise_by if to_call > 0 else min_raise_by),
        )
        max_raise_by = _first_nonnegative(
            raw,
            ("max_raise_by", "max_legal_raise_by", "maximum_raise_by"),
            default=max(0.0, max_raise_to - to_call),
        )
        min_raise_by = min(min_raise_by, max_raise_by) if max_raise_by > 0 else min_raise_by

        legal_actions = _raw_legal_actions(raw)
        if not legal_actions:
            legal_actions = derive_legal_actions(
                to_call=to_call,
                stack=stack,
                min_raise_to=min_raise_to,
                max_raise_to=max_raise_to,
            )
        else:
            legal_actions = tuple(action for action in legal_actions if action in _physically_possible_actions(
                to_call=to_call,
                stack=stack,
                min_raise_to=min_raise_to,
                max_raise_to=max_raise_to,
            ))
            if not legal_actions:
                legal_actions = derive_legal_actions(
                    to_call=to_call,
                    stack=stack,
                    min_raise_to=min_raise_to,
                    max_raise_to=max_raise_to,
                )

        return cls(
            legal_actions=legal_actions,
            min_raise_to=min_raise_to,
            max_raise_to=max_raise_to,
            min_raise_by=min_raise_by,
            max_raise_by=max_raise_by,
            all_in_amount=all_in_amount,
            to_call=to_call,
            stack=stack,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_actions": list(CANONICAL_ACTIONS),
            "legal_actions": list(self.legal_actions),
            "min_raise_to": self.min_raise_to,
            "max_raise_to": self.max_raise_to,
            "min_raise_by": self.min_raise_by,
            "max_raise_by": self.max_raise_by,
            "all_in_amount": self.all_in_amount,
            "to_call": self.to_call,
            "stack": self.stack,
        }


def normalize_action(action: Any) -> str:
    text = " ".join(str(action or "").strip().lower().replace("_", " ").split())
    if not text:
        return "unknown"
    canonical = ACTION_ALIASES.get(text) or ACTION_ALIASES.get(text.replace(" ", "_"))
    if canonical:
        return canonical
    if "fold" in text:
        return "fold"
    if "check" in text:
        return "check"
    if "call" in text:
        return "call"
    if "raise" in text:
        return "raise"
    if "bet" in text:
        return "bet"
    if "all" in text or "shove" in text or "jam" in text:
        return "all_in"
    return text.replace(" ", "_")


def derive_legal_actions(
    *,
    to_call: float,
    stack: float,
    min_raise_to: float,
    max_raise_to: float,
) -> tuple[str, ...]:
    possible = _physically_possible_actions(
        to_call=to_call,
        stack=stack,
        min_raise_to=min_raise_to,
        max_raise_to=max_raise_to,
    )
    ordered = ("fold", "call", "raise", "all_in") if to_call > 0 else ("check", "bet", "all_in")
    return tuple(action for action in ordered if action in possible)


def constrain_probabilities(
    raw_probabilities: dict[str, float],
    action_space: ActionSpace,
) -> tuple[str, dict[str, float], list[str]]:
    warnings: list[str] = []
    values = {action: 0.0 for action in CANONICAL_ACTIONS}
    for raw_action, raw_value in raw_probabilities.items():
        action = normalize_action(raw_action)
        if action in values:
            values[action] += max(0.0, float(raw_value or 0.0))

    legal = set(action_space.legal_actions)
    illegal_mass = sum(value for action, value in values.items() if action not in legal)
    if illegal_mass > 0:
        warnings.append("Illegal action probability mass was removed by the legal action mask.")
    for action in CANONICAL_ACTIONS:
        if action not in legal:
            values[action] = 0.0

    total = sum(values.values())
    if total <= 0:
        fallback = fallback_action(action_space)
        values[fallback] = 1.0
        total = 1.0
        warnings.append("Model produced no legal action mass; used deterministic legal fallback.")

    normalized = {action: values[action] / total for action in CANONICAL_ACTIONS}
    selected = max(action_space.legal_actions, key=lambda action: normalized.get(action, 0.0))
    return selected, normalized, warnings


def action_amounts(action: str, action_space: ActionSpace) -> dict[str, Any]:
    action = normalize_action(action)
    if action in {"fold", "check"}:
        return {"bet_size": 0.0, "raise_to": None, "raise_by": None, "sizing_method": "no_chip_commitment"}
    if action == "call":
        return {
            "bet_size": min(action_space.to_call, action_space.stack),
            "raise_to": None,
            "raise_by": None,
            "sizing_method": "call_amount",
        }
    if action == "all_in":
        return {
            "bet_size": action_space.all_in_amount,
            "raise_to": action_space.all_in_amount,
            "raise_by": max(0.0, action_space.all_in_amount - action_space.to_call),
            "sizing_method": "all_in",
        }
    if action in {"bet", "raise"}:
        raise_to = min(max(action_space.min_raise_to, 0.0), action_space.max_raise_to)
        raise_by = min(max(action_space.min_raise_by, 0.0), action_space.max_raise_by)
        return {
            "bet_size": raise_to,
            "raise_to": raise_to if action == "raise" else None,
            "raise_by": raise_by if action == "raise" else None,
            "sizing_method": "legal_minimum",
        }
    return {"bet_size": 0.0, "raise_to": None, "raise_by": None, "sizing_method": "unknown_action"}


def fallback_action(action_space: ActionSpace) -> str:
    for action in ("check", "call", "fold", "all_in", "bet", "raise"):
        if action in action_space.legal_actions:
            return action
    return action_space.legal_actions[0] if action_space.legal_actions else "fold"


def _raw_legal_actions(raw: dict[str, Any]) -> tuple[str, ...]:
    raw_actions = raw.get("legal_actions") or raw.get("legal_action_mask")
    if raw_actions is None:
        return ()
    if isinstance(raw_actions, str):
        pieces: Iterable[Any] = raw_actions.replace(",", " ").split()
    elif isinstance(raw_actions, dict):
        pieces = [action for action, allowed in raw_actions.items() if allowed]
    else:
        pieces = raw_actions
    legal: list[str] = []
    for item in pieces:
        action = normalize_action(item)
        if action in CANONICAL_ACTIONS and action not in legal:
            legal.append(action)
    return tuple(legal)


def _physically_possible_actions(
    *,
    to_call: float,
    stack: float,
    min_raise_to: float,
    max_raise_to: float,
) -> tuple[str, ...]:
    if stack <= 0:
        return ("check",) if to_call <= 0 else ("fold",)
    if to_call > 0:
        actions = ["fold", "call"]
        if max_raise_to >= min_raise_to and max_raise_to > to_call:
            actions.append("raise")
        actions.append("all_in")
        return tuple(actions)
    actions = ["check"]
    if max_raise_to > 0:
        actions.append("bet")
        actions.append("all_in")
    return tuple(actions)


def _first_nonnegative(raw: dict[str, Any], keys: tuple[str, ...], *, default: float) -> float:
    for key in keys:
        if key in raw and raw[key] not in {None, ""}:
            return _nonnegative_float(raw[key])
    return _nonnegative_float(default)


def _nonnegative_float(value: Any) -> float:
    try:
        number = float(value or 0.0)
    except (TypeError, ValueError):
        number = 0.0
    return max(0.0, number)
