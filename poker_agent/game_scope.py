from __future__ import annotations

from dataclasses import dataclass
from typing import Any

GAME_SCOPE_CONTRACT_VERSION = "game_scope.v1"

SUPPORTED_GAME_TYPES = ("nl_holdem",)
SUPPORTED_FORMATS = ("cash", "tournament")
SUPPORTED_TABLE_SIZES = ("6_max", "9_max")
SUPPORTED_STACK_UNITS = ("chips", "big_blinds", "chips_or_big_blinds")

GAME_TYPE_ALIASES = {
    "nlh": "nl_holdem",
    "nl_holdem": "nl_holdem",
    "no_limit_holdem": "nl_holdem",
    "no-limit-holdem": "nl_holdem",
    "texas_holdem_no_limit": "nl_holdem",
}
FORMAT_ALIASES = {
    "cash": "cash",
    "cash_game": "cash",
    "ring": "cash",
    "tournament": "tournament",
    "mtt": "tournament",
    "sng": "tournament",
}
TABLE_SIZE_ALIASES = {
    "6": "6_max",
    "6max": "6_max",
    "6_max": "6_max",
    "six_max": "6_max",
    "9": "9_max",
    "9max": "9_max",
    "9_max": "9_max",
    "full_ring": "9_max",
}
STACK_UNIT_ALIASES = {
    "chip": "chips",
    "chips": "chips",
    "bb": "big_blinds",
    "big_blind": "big_blinds",
    "big_blinds": "big_blinds",
    "chips_or_big_blinds": "chips_or_big_blinds",
}


def _as_float(raw: Any, default: float = 0.0) -> float:
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _text(raw: Any, default: str = "") -> str:
    if raw is None or raw == "":
        return default
    return str(raw).strip().lower().replace(" ", "_").replace("-", "_")


def _first(raw: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        if key in raw and raw[key] not in (None, ""):
            return raw[key]
    return default


def _scope_payload(payload: dict[str, Any]) -> dict[str, Any]:
    nested = payload.get("game_scope") if isinstance(payload.get("game_scope"), dict) else {}
    return {**nested, **payload}


def normalize_game_type(raw: Any) -> str:
    return GAME_TYPE_ALIASES.get(_text(raw, "nl_holdem"), _text(raw, "nl_holdem"))


def normalize_format(raw: Any) -> str:
    return FORMAT_ALIASES.get(_text(raw, "cash"), _text(raw, "cash"))


def normalize_table_size(raw: Any) -> str:
    return TABLE_SIZE_ALIASES.get(_text(raw, "6_max"), _text(raw, "6_max"))


def normalize_stack_unit(raw: Any) -> str:
    return STACK_UNIT_ALIASES.get(_text(raw, "chips"), _text(raw, "chips"))


def normalize_rake_percentage(raw: Any) -> float:
    value = _as_float(raw, 0.0)
    if value > 1.0 and value <= 100.0:
        return value / 100.0
    return value


@dataclass(frozen=True)
class GameScope:
    game_type: str = "nl_holdem"
    format: str = "cash"
    table_size: str = "6_max"
    small_blind: float = 0.0
    big_blind: float = 0.0
    ante: float = 0.0
    rake_percentage: float = 0.0
    rake_cap: float = 0.0
    stack_unit: str = "chips"

    @classmethod
    def default(cls) -> "GameScope":
        return cls()

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "GameScope":
        raw = _scope_payload(payload)
        big_blind = _as_float(_first(raw, ("big_blind", "bb"), 0.0))
        small_blind = _as_float(_first(raw, ("small_blind", "sb"), 0.0))
        if small_blind <= 0.0 and big_blind > 0.0:
            small_blind = big_blind / 2.0
        return cls(
            game_type=normalize_game_type(_first(raw, ("game_type", "variant"), "nl_holdem")),
            format=normalize_format(_first(raw, ("format", "game_format", "table_format"), "cash")),
            table_size=normalize_table_size(_first(raw, ("table_size", "max_players", "seats"), "6_max")),
            small_blind=small_blind,
            big_blind=big_blind,
            ante=_as_float(_first(raw, ("ante",), 0.0)),
            rake_percentage=normalize_rake_percentage(_first(raw, ("rake_percentage", "rake_percent"), 0.0)),
            rake_cap=_as_float(_first(raw, ("rake_cap",), 0.0)),
            stack_unit=normalize_stack_unit(_first(raw, ("stack_unit", "unit"), "chips")),
        )

    def validate(self) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        if self.game_type not in SUPPORTED_GAME_TYPES:
            findings.append({"field": "game_type", "code": "unsupported_game_type", "value": self.game_type})
        if self.format not in SUPPORTED_FORMATS:
            findings.append({"field": "format", "code": "unsupported_format", "value": self.format})
        if self.table_size not in SUPPORTED_TABLE_SIZES:
            findings.append({"field": "table_size", "code": "unsupported_table_size", "value": self.table_size})
        if self.stack_unit not in SUPPORTED_STACK_UNITS:
            findings.append({"field": "stack_unit", "code": "unsupported_stack_unit", "value": self.stack_unit})
        if self.small_blind < 0 or self.big_blind < 0 or self.ante < 0:
            findings.append({"field": "blinds", "code": "negative_blind_or_ante"})
        if self.big_blind > 0 and self.small_blind > self.big_blind:
            findings.append({"field": "small_blind", "code": "small_blind_exceeds_big_blind"})
        if self.rake_percentage < 0 or self.rake_percentage > 1:
            findings.append({"field": "rake_percentage", "code": "rake_percentage_out_of_range"})
        if self.rake_cap < 0:
            findings.append({"field": "rake_cap", "code": "negative_rake_cap"})
        return findings

    @property
    def table_size_players(self) -> int:
        return 9 if self.table_size == "9_max" else 6

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": GAME_SCOPE_CONTRACT_VERSION,
            "game_type": self.game_type,
            "format": self.format,
            "table_size": self.table_size,
            "table_size_players": self.table_size_players,
            "small_blind": self.small_blind,
            "big_blind": self.big_blind,
            "ante": self.ante,
            "rake_percentage": self.rake_percentage,
            "rake_cap": self.rake_cap,
            "stack_unit": self.stack_unit,
        }

    def to_state_fields(self) -> dict[str, Any]:
        return {
            "small_blind": self.small_blind,
            "big_blind": self.big_blind,
            "ante": self.ante,
        }

    def feature_values(self) -> dict[str, float]:
        return {
            "scope_game_type_nl_holdem": 1.0 if self.game_type == "nl_holdem" else 0.0,
            "scope_format_cash": 1.0 if self.format == "cash" else 0.0,
            "scope_format_tournament": 1.0 if self.format == "tournament" else 0.0,
            "scope_table_6_max": 1.0 if self.table_size == "6_max" else 0.0,
            "scope_table_9_max": 1.0 if self.table_size == "9_max" else 0.0,
            "scope_table_size_players": float(self.table_size_players),
            "scope_rake_percentage": self.rake_percentage,
            "scope_rake_cap": self.rake_cap,
            "scope_has_rake": 1.0 if self.rake_percentage > 0 or self.rake_cap > 0 else 0.0,
            "scope_stack_unit_chips": 1.0 if self.stack_unit == "chips" else 0.0,
            "scope_stack_unit_big_blinds": 1.0 if self.stack_unit == "big_blinds" else 0.0,
        }


def describe_game_scope_contract() -> dict[str, Any]:
    return {
        "schema_version": GAME_SCOPE_CONTRACT_VERSION,
        "supported_game_types": list(SUPPORTED_GAME_TYPES),
        "supported_formats": list(SUPPORTED_FORMATS),
        "supported_table_sizes": list(SUPPORTED_TABLE_SIZES),
        "supported_stack_units": list(SUPPORTED_STACK_UNITS),
        "required_fields": [
            "game_type",
            "format",
            "table_size",
            "small_blind",
            "big_blind",
            "ante",
            "rake_percentage",
            "rake_cap",
            "stack_unit",
        ],
    }
