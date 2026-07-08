from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


FORBIDDEN_STACK_EVENT_FEATURE_NAMES = (
    "stack_delta",
    "stack_after_event",
    "target_stack_delta",
    "target_action_stack_delta",
    "ending_stack",
)

REQUIRED_STACK_CONTEXT_FEATURE_NAMES = (
    "stack_event_context_reconstructed",
    "stack_event_target_bet_size_used_as_feature",
    "reconstructed_pot",
    "reconstructed_effective_stack",
    "reconstructed_effective_stack_to_pot",
    "reconstructed_spr_after_call",
    "reconstructed_current_street_bet_size",
    "reconstructed_current_street_bet_to_pot",
    "reconstructed_pot_pressure",
    "reconstructed_call_pressure",
    "reconstructed_raise_pressure",
)


def detect_stack_event_feature_leakage(feature_names: Iterable[str]) -> list[str]:
    return sorted(
        {
            str(name)
            for name in feature_names
            for forbidden in FORBIDDEN_STACK_EVENT_FEATURE_NAMES
            if forbidden in str(name)
        }
    )


def assert_stack_decision_context_feature_contract(
    features: Mapping[str, Any],
    *,
    context: str,
) -> None:
    leaked = detect_stack_event_feature_leakage(features.keys())
    if leaked:
        joined = ", ".join(leaked)
        raise ValueError(
            f"Raw stack-event leakage detected in {context}: {joined}. "
            "Use only reconstructed decision-time stack context features."
        )

    missing = [name for name in REQUIRED_STACK_CONTEXT_FEATURE_NAMES if name not in features]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing reconstructed stack context features in {context}: {joined}.")

    sentinel = float(features.get("stack_event_target_bet_size_used_as_feature", 1.0))
    if sentinel != 0.0:
        raise ValueError(
            f"Target action stack delta leakage guard is disabled in {context}. "
            "The sentinel stack_event_target_bet_size_used_as_feature must be 0.0."
        )


@dataclass(frozen=True)
class StackDecisionContext:
    pot_base: float
    effective_stack: float
    stack_to_pot: float
    spr_after_call: float
    current_street_bet_size: float
    current_street_bet_to_pot: float
    pot_pressure: float
    call_pressure: float
    raise_pressure: float
    call_price_ratio: float
    raise_to_stack: float
    hero_commitment_ratio: float
    table_commitment_pressure: float

    def as_feature_dict(self) -> dict[str, float]:
        return {
            "stack_event_context_reconstructed": 1.0,
            "stack_event_target_bet_size_used_as_feature": 0.0,
            "reconstructed_pot": self.pot_base,
            "reconstructed_effective_stack": self.effective_stack,
            "reconstructed_effective_stack_to_pot": self.stack_to_pot,
            "reconstructed_spr_after_call": self.spr_after_call,
            "reconstructed_current_street_bet_size": self.current_street_bet_size,
            "reconstructed_current_street_bet_to_pot": self.current_street_bet_to_pot,
            "reconstructed_pot_pressure": self.pot_pressure,
            "reconstructed_call_pressure": self.call_pressure,
            "reconstructed_raise_pressure": self.raise_pressure,
        }


def build_stack_decision_context(
    *,
    running_pot: float,
    highest_commit: float,
    hero_commit: float,
    decision_stack: float,
    to_call: float,
    min_raise: float,
) -> StackDecisionContext:
    """Convert raw stack-event deltas into decision-time betting context."""
    pot_base = max(float(running_pot or 0.0), float(highest_commit or 0.0), 1.0)
    effective_stack = max(float(decision_stack or 0.0), 0.0)
    hero_stack_base = max(effective_stack + max(float(hero_commit or 0.0), 0.0), 1.0)
    call_price = max(float(to_call or 0.0), 0.0)
    raise_price = max(float(min_raise or 0.0), 0.0)
    street_bet = max(float(highest_commit or 0.0), 0.0)
    pot_after_call = max(pot_base + call_price, 1.0)

    call_to_stack = min(call_price / max(effective_stack, 1.0), 1.0) if effective_stack > 0 else 0.0
    raise_to_stack = min(raise_price / max(effective_stack, 1.0), 1.0) if effective_stack > 0 else 0.0

    return StackDecisionContext(
        pot_base=pot_base,
        effective_stack=effective_stack,
        stack_to_pot=min(effective_stack / pot_base, 100.0),
        spr_after_call=min(max(effective_stack - call_price, 0.0) / pot_after_call, 100.0),
        current_street_bet_size=street_bet,
        current_street_bet_to_pot=min(street_bet / pot_base, 1.0),
        pot_pressure=min(pot_base / max(pot_base + effective_stack, 1.0), 1.0),
        call_pressure=min(call_price / pot_base, 1.0),
        raise_pressure=raise_to_stack,
        call_price_ratio=call_to_stack,
        raise_to_stack=raise_to_stack,
        hero_commitment_ratio=min(max(float(hero_commit or 0.0), 0.0) / hero_stack_base, 1.0),
        table_commitment_pressure=min(street_bet / pot_base, 1.0),
    )


def derive_stack_decision_context_from_events(
    stack_events: Iterable[Mapping[str, Any]],
    *,
    target_frame_id: Any,
    hero_position: str,
    starting_stack: float,
    to_call: float,
    min_raise: float,
    street_start_frame_id: Any | None = None,
) -> StackDecisionContext:
    """Derive policy-safe stack context from raw stack events before the target action.

    Only events with frame_id strictly below target_frame_id are used. This keeps the
    target action's own stack delta, and all future stack changes, out of the feature
    vector used to predict that target action.
    """
    target_frame = _coerce_float(target_frame_id, default=float("inf"))
    street_start = (
        None
        if street_start_frame_id is None
        else _coerce_float(street_start_frame_id, default=float("-inf"))
    )
    hero_key = _normalize_position(hero_position)
    commits_by_position: dict[str, float] = {}
    running_pot = 0.0

    for event in stack_events:
        frame_id = _coerce_float(event.get("frame_id"), default=float("inf"))
        if frame_id >= target_frame:
            continue
        if street_start is not None and frame_id < street_start:
            continue

        diff = _coerce_float(event.get("diff"), default=0.0)
        if diff >= 0.0:
            continue

        contribution = abs(diff)
        running_pot += contribution
        position = _normalize_position(event.get("player_position"))
        if position:
            commits_by_position[position] = commits_by_position.get(position, 0.0) + contribution

    hero_commit = commits_by_position.get(hero_key, 0.0)
    highest_commit = max(commits_by_position.values(), default=0.0)
    decision_stack = max(float(starting_stack or 0.0) - hero_commit, 0.0)

    return build_stack_decision_context(
        running_pot=running_pot,
        highest_commit=highest_commit,
        hero_commit=hero_commit,
        decision_stack=decision_stack,
        to_call=to_call,
        min_raise=min_raise,
    )


def _coerce_float(raw: Any, *, default: float = 0.0) -> float:
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _normalize_position(raw: Any) -> str:
    return str(raw or "").strip().upper()
