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

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PredictionRequest":
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
        )


@dataclass
class PredictionResponse:
    action: str
    probabilities: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "probabilities": self.probabilities}

