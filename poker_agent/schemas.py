from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


VALID_ACTIONS = ("fold", "call", "check", "bet", "raise", "all_in")


@dataclass
class PredictionRequest:
    position: str
    street: str = "preflop"
    hole_cards: list[str] = field(default_factory=list)
    board_cards: list[str] = field(default_factory=list)
    pot: float = 0.0
    to_call: float = 0.0
    stack: float = 0.0
    min_raise: float = 0.0
    player_count: int = 6
    betting_history: list[dict[str, Any]] = field(default_factory=list)
    opponent_wait_before_turn_ms: float = 0.0
    opponent_wait_after_hero_action_ms: float = 0.0

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PredictionRequest":
        timing_context = raw.get("timing_context")
        if not isinstance(timing_context, dict):
            timing_context = {}
        return cls(
            position=str(raw.get("position") or raw.get("player_position") or "UNK"),
            street=str(raw.get("street") or "preflop").lower(),
            hole_cards=[str(card) for card in raw.get("hole_cards", [])],
            board_cards=[str(card) for card in raw.get("board_cards", [])],
            pot=float(raw.get("pot") or 0.0),
            to_call=float(raw.get("to_call") or 0.0),
            stack=float(raw.get("stack") or 0.0),
            min_raise=float(raw.get("min_raise") or 0.0),
            player_count=int(raw.get("player_count") or 6),
            betting_history=list(raw.get("betting_history") or raw.get("action_history") or []),
            opponent_wait_before_turn_ms=max(
                0.0,
                float(
                    raw.get("opponent_wait_before_turn_ms")
                    or timing_context.get("opponent_wait_before_turn_ms")
                    or 0.0
                ),
            ),
            opponent_wait_after_hero_action_ms=max(
                0.0,
                float(
                    raw.get("opponent_wait_after_hero_action_ms")
                    or timing_context.get("opponent_wait_after_hero_action_ms")
                    or 0.0
                ),
            ),
        )


@dataclass
class PredictionResponse:
    action: str
    probabilities: dict[str, float]
    confidence: float = 0.0
    bet_size: float = 0.0
    wait_time_ms: int = 250
    sizing_method: str = "no_chip_commitment"
    timing_method: str = "complexity_calibrated"
    model_status: str = "model"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        confidence = self.confidence or max(self.probabilities.values(), default=0.0)
        payload: dict[str, Any] = {
            "action": self.action,
            "probabilities": self.probabilities,
            "confidence": confidence,
            "bet_size": self.bet_size,
            "wait_time_ms": self.wait_time_ms,
            "sizing_method": self.sizing_method,
            "timing_method": self.timing_method,
            "model_status": self.model_status,
        }
        if self.warnings:
            payload["warnings"] = self.warnings
        return payload
