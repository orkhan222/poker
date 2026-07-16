from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from poker_agent.action_space import ActionSpace, CANONICAL_ACTIONS
from poker_agent.api_contract import PREDICT_RESPONSE_SCHEMA_VERSION
from poker_agent.game_scope import GameScope

VALID_ACTIONS = CANONICAL_ACTIONS


def _default_action_space() -> ActionSpace:
    return ActionSpace.from_state({}, to_call=0.0, stack=0.0, min_raise=0.0)


def _as_float(raw: Any, default: float = 0.0) -> float:
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _first_float(raw: dict[str, Any], keys: tuple[str, ...], *, default: float = 0.0) -> float:
    for key in keys:
        if key in raw and raw[key] is not None and raw[key] != "":
            return _as_float(raw[key], default)
    return default


def _first_text(raw: dict[str, Any], keys: tuple[str, ...], *, default: str = "") -> str:
    for key in keys:
        if key in raw and raw[key] is not None and raw[key] != "":
            return str(raw[key])
    return default


def _as_text_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [part for part in raw.replace(",", " ").split() if part]
    return [str(item) for item in raw if str(item)]


@dataclass
class PredictionRequest:
    position: str
    street: str = "preflop"
    hole_cards: list[str] = field(default_factory=list)
    board_cards: list[str] = field(default_factory=list)
    pot: float = 0.0
    to_call: float = 0.0
    current_bet: float = 0.0
    amount_to_call: float = 0.0
    stack: float = 0.0
    effective_stack: float = 0.0
    min_raise: float = 0.0
    max_raise: float = 0.0
    small_blind: float = 0.0
    big_blind: float = 0.0
    ante: float = 0.0
    button_position: str = ""
    dealer_position: str = ""
    action_order: list[str] = field(default_factory=list)
    player_count: int = 6
    betting_history: list[dict[str, Any]] = field(default_factory=list)
    game_scope: GameScope = field(default_factory=GameScope.default)
    action_space: ActionSpace = field(default_factory=_default_action_space)

    def __post_init__(self) -> None:
        self.street = str(self.street or "preflop").lower()
        self.amount_to_call = self.amount_to_call if self.amount_to_call > 0 else self.to_call
        self.to_call = self.to_call if self.to_call > 0 else self.amount_to_call
        self.current_bet = self.current_bet if self.current_bet > 0 else self.to_call
        self.effective_stack = self.effective_stack if self.effective_stack > 0 else self.stack
        if not self.button_position and self.dealer_position:
            self.button_position = self.dealer_position
        if not self.dealer_position and self.button_position:
            self.dealer_position = self.button_position
        if self.action_space == _default_action_space() and (
            self.to_call > 0 or self.effective_stack > 0 or self.min_raise > 0 or self.max_raise > 0
        ):
            action_state = {"effective_stack": self.effective_stack}
            if self.max_raise > 0:
                action_state["max_raise"] = self.max_raise
            self.action_space = ActionSpace.from_state(
                action_state,
                to_call=self.to_call,
                stack=self.effective_stack,
                min_raise=self.min_raise,
            )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PredictionRequest":
        parsed_game_scope = GameScope.from_payload(raw)
        game_scope = raw.get("game_scope") if isinstance(raw.get("game_scope"), dict) else {}
        state = {**game_scope, **raw, **parsed_game_scope.to_state_fields()}
        amount_to_call = _first_float(state, ("amount_to_call", "to_call", "call_amount"), default=0.0)
        to_call = _first_float(state, ("to_call", "amount_to_call", "call_amount"), default=amount_to_call)
        stack = _first_float(state, ("stack", "hero_stack", "remaining_stack"), default=0.0)
        effective_stack = _first_float(state, ("effective_stack", "effective_stack_size"), default=stack)
        current_bet = _first_float(
            state,
            ("current_bet", "highest_bet", "street_current_bet", "bet_to_call"),
            default=amount_to_call,
        )
        min_raise = _first_float(state, ("min_raise", "min_raise_by"), default=0.0)
        max_raise = _first_float(state, ("max_raise", "max_raise_to", "all_in_amount"), default=effective_stack or stack)
        action_space = ActionSpace.from_state(state, to_call=to_call, stack=effective_stack or stack, min_raise=min_raise)
        return cls(
            position=str(state.get("position") or state.get("player_position") or "UNK"),
            street=str(state.get("street") or "preflop").lower(),
            hole_cards=[str(card) for card in state.get("hole_cards", [])],
            board_cards=[str(card) for card in state.get("board_cards", [])],
            pot=_first_float(state, ("pot", "pot_size"), default=0.0),
            to_call=to_call,
            current_bet=current_bet,
            amount_to_call=amount_to_call,
            stack=stack,
            effective_stack=effective_stack,
            min_raise=min_raise,
            max_raise=max_raise,
            small_blind=_first_float(state, ("small_blind", "sb"), default=0.0),
            big_blind=_first_float(state, ("big_blind", "bb"), default=0.0),
            ante=_first_float(state, ("ante",), default=0.0),
            button_position=_first_text(state, ("button_position", "button", "dealer_position", "dealer"), default=""),
            dealer_position=_first_text(state, ("dealer_position", "dealer", "button_position", "button"), default=""),
            action_order=_as_text_list(
                state.get("action_order")
                or state.get("acting_order")
                or state.get("positions_in_order")
                or state.get("turn_order")
            ),
            player_count=int(state.get("player_count") or parsed_game_scope.table_size_players),
            betting_history=list(state.get("betting_history") or state.get("action_history") or []),
            game_scope=parsed_game_scope,
            action_space=action_space,
        )

    @property
    def legal_actions(self) -> tuple[str, ...]:
        return self.action_space.legal_actions

    @property
    def min_raise_to(self) -> float:
        return self.action_space.min_raise_to

    @property
    def max_raise_to(self) -> float:
        return self.action_space.max_raise_to

    @property
    def min_raise_by(self) -> float:
        return self.action_space.min_raise_by

    @property
    def max_raise_by(self) -> float:
        return self.action_space.max_raise_by

    @property
    def all_in_amount(self) -> float:
        return self.action_space.all_in_amount

    def action_order_index(self) -> int:
        try:
            return self.action_order.index(self.position)
        except ValueError:
            return -1

    def state_context(self) -> dict[str, Any]:
        denominator = self.pot + self.amount_to_call
        return {
            "pot_size": self.pot,
            "current_bet": self.current_bet,
            "amount_to_call": self.amount_to_call,
            "button_position": self.button_position,
            "dealer_position": self.dealer_position,
            "small_blind": self.small_blind,
            "big_blind": self.big_blind,
            "ante": self.ante,
            "street": self.street,
            "effective_stack": self.effective_stack,
            "spr": self.effective_stack / denominator if denominator > 0 else 0.0,
            "action_order": list(self.action_order),
            "action_order_index": self.action_order_index(),
            "game_scope": self.game_scope.to_dict(),
        }


@dataclass
class PredictionResponse:
    action: str
    probabilities: dict[str, float]
    schema_version: str = PREDICT_RESPONSE_SCHEMA_VERSION
    model_version: str = "unknown"
    confidence: float = 0.0
    model_status: str = "model"
    warnings: list[str] = field(default_factory=list)
    bet_size: float = 0.0
    raise_to: float | None = None
    raise_by: float | None = None
    sizing_method: str = "no_chip_commitment"
    legal_actions: tuple[str, ...] = field(default_factory=tuple)
    action_space: dict[str, Any] = field(default_factory=dict)
    state_context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        confidence = self.confidence or max(self.probabilities.values(), default=0.0)
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "model_version": self.model_version,
            "action": self.action,
            "probabilities": self.probabilities,
            "confidence": confidence,
            "model_status": self.model_status,
            "bet_size": self.bet_size,
            "raise_to": self.raise_to,
            "raise_by": self.raise_by,
            "sizing_method": self.sizing_method,
            "legal_actions": list(self.legal_actions),
            "action_space": self.action_space,
            "state_context": self.state_context,
        }
        if self.warnings:
            payload["warnings"] = self.warnings
        return payload
