from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


VALID_ACTIONS = ("fold", "call", "check", "bet", "raise", "all_in")
VALID_GAME_VARIANTS = ("nl_holdem",)
VALID_GAME_TYPES = ("cash", "tournament")
VALID_TABLE_FORMATS = ("6_max", "9_max")
VALID_STACK_UNITS = ("chips", "big_blinds")


@dataclass(frozen=True)
class GameScope:
    game_variant: str = "nl_holdem"
    game_type: str = "cash"
    table_format: str = "6_max"
    small_blind: float = 0.5
    big_blind: float = 1.0
    ante: float = 0.0
    rake_percentage: float = 0.0
    rake_cap: float = 0.0
    stack_unit: str = "chips"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "GameScope":
        scope = raw.get("game_scope")
        if not isinstance(scope, dict):
            scope = {}
        merged = {**raw, **scope}
        table_format = _normalize_table_format(
            merged.get("table_format") or merged.get("table_size") or merged.get("max_players"),
            player_count=merged.get("player_count"),
        )
        small_blind = _nonnegative_float(merged.get("small_blind"), 0.5)
        big_blind = _nonnegative_float(merged.get("big_blind"), 1.0)
        if small_blind > 0 and big_blind > 0 and big_blind < small_blind:
            raise ValueError("game_scope.big_blind must be greater than or equal to small_blind")
        return cls(
            game_variant=_normalize_choice(
                merged.get("game_variant") or merged.get("variant"),
                aliases={
                    "nl_holdem": "nl_holdem",
                    "no_limit_holdem": "nl_holdem",
                    "no-limit holdem": "nl_holdem",
                    "no limit holdem": "nl_holdem",
                    "no-limit hold'em": "nl_holdem",
                    "no limit hold'em": "nl_holdem",
                    "nl holdem": "nl_holdem",
                    "nlhe": "nl_holdem",
                    "texas holdem": "nl_holdem",
                    "texas hold'em": "nl_holdem",
                },
                valid=VALID_GAME_VARIANTS,
                default="nl_holdem",
                field_name="game_scope.game_variant",
            ),
            game_type=_normalize_choice(
                merged.get("game_type") or merged.get("format"),
                aliases={
                    "cash": "cash",
                    "cash_game": "cash",
                    "ring": "cash",
                    "tournament": "tournament",
                    "mtt": "tournament",
                    "sng": "tournament",
                    "sit_and_go": "tournament",
                },
                valid=VALID_GAME_TYPES,
                default="cash",
                field_name="game_scope.game_type",
            ),
            table_format=table_format,
            small_blind=small_blind,
            big_blind=big_blind,
            ante=_nonnegative_float(merged.get("ante"), 0.0),
            rake_percentage=_nonnegative_float(
                merged.get("rake_percentage") or merged.get("rake_percent"),
                0.0,
            ),
            rake_cap=_nonnegative_float(merged.get("rake_cap"), 0.0),
            stack_unit=_normalize_choice(
                merged.get("stack_unit"),
                aliases={
                    "chips": "chips",
                    "chip": "chips",
                    "bb": "big_blinds",
                    "big_blind": "big_blinds",
                    "big_blinds": "big_blinds",
                },
                valid=VALID_STACK_UNITS,
                default="chips",
                field_name="game_scope.stack_unit",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_variant": self.game_variant,
            "game_type": self.game_type,
            "table_format": self.table_format,
            "small_blind": self.small_blind,
            "big_blind": self.big_blind,
            "ante": self.ante,
            "rake_percentage": self.rake_percentage,
            "rake_cap": self.rake_cap,
            "stack_unit": self.stack_unit,
        }


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
    game_scope: GameScope = field(default_factory=GameScope)

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
            game_scope=GameScope.from_dict(raw),
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
    strategy_guardrails: list[str] = field(default_factory=list)

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
        if self.strategy_guardrails:
            payload["strategy_guardrails"] = self.strategy_guardrails
        return payload


def _nonnegative_float(raw: Any, default: float) -> float:
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected non-negative numeric game scope value, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"Expected non-negative game scope value, got {value}")
    return value


def _normalize_choice(
    raw: Any,
    *,
    aliases: dict[str, str],
    valid: tuple[str, ...],
    default: str,
    field_name: str,
) -> str:
    if raw is None or raw == "":
        return default
    key = str(raw).strip().lower().replace("_", " ").replace("-", " ")
    normalized_key = " ".join(key.split())
    value = aliases.get(normalized_key) or aliases.get(normalized_key.replace(" ", "_"))
    if value is None:
        value = str(raw).strip().lower()
    if value not in valid:
        raise ValueError(f"Unsupported {field_name}: {raw!r}. Expected one of {valid}.")
    return value


def _normalize_table_format(raw: Any, *, player_count: Any) -> str:
    if raw is None or raw == "":
        try:
            count = int(player_count)
        except (TypeError, ValueError):
            return "6_max"
        return "6_max" if count <= 6 else "9_max"
    text = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "6": "6_max",
        "6max": "6_max",
        "6_max": "6_max",
        "six_max": "6_max",
        "9": "9_max",
        "9max": "9_max",
        "9_max": "9_max",
        "nine_max": "9_max",
    }
    value = aliases.get(text)
    if value not in VALID_TABLE_FORMATS:
        raise ValueError(f"Unsupported game_scope.table_format: {raw!r}. Expected one of {VALID_TABLE_FORMATS}.")
    return value
