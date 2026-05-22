from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Iterable

from poker_agent.schemas import PredictionRequest
from poker_agent.schemas import VALID_ACTIONS


RANK_TO_VALUE = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}

STREET_ORDER = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
ACTION_NORMALIZATION = {
    "all-in": "all_in",
    "all in": "all_in",
    "allin": "all_in",
    "calls": "call",
    "bets": "bet",
    "raises": "raise",
    "folds": "fold",
    "checks": "check",
}


def normalize_action(action: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(action).strip().lower())
    cleaned = ACTION_NORMALIZATION.get(cleaned, cleaned)
    if "fold" in cleaned:
        return "fold"
    if "check" in cleaned:
        return "check"
    if "call" in cleaned:
        return "call"
    if "raise" in cleaned:
        return "raise"
    if "bet" in cleaned:
        return "bet"
    if "all" in cleaned:
        return "all_in"
    return cleaned or "unknown"


def parse_cards(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(card).strip() for card in raw if str(card).strip()]
    return [part for part in str(raw).replace(",", " ").split() if part]


def card_rank(card: str) -> int:
    if not card:
        return 0
    return RANK_TO_VALUE.get(card[0].upper(), 0)


def card_suit(card: str) -> str:
    return card[1:].lower() if len(card) > 1 else ""


def hand_strength_proxy(hole_cards: Iterable[str]) -> float:
    cards = list(hole_cards)[:2]
    if len(cards) < 2:
        return 0.0
    ranks = [card_rank(card) for card in cards]
    suits = [card_suit(card) for card in cards]
    high = max(ranks) / 14.0
    low = min(ranks) / 14.0
    pair_bonus = 0.28 if ranks[0] == ranks[1] else 0.0
    suited_bonus = 0.08 if suits[0] and suits[0] == suits[1] else 0.0
    connector_bonus = 0.05 if abs(ranks[0] - ranks[1]) <= 1 else 0.0
    return min(1.0, 0.55 * high + 0.25 * low + pair_bonus + suited_bonus + connector_bonus)


def request_to_features(request: PredictionRequest) -> dict[str, float]:
    pot_odds = request.to_call / (request.pot + request.to_call) if request.pot + request.to_call > 0 else 0.0
    stack_to_pot = request.stack / request.pot if request.pot > 0 else 0.0
    return {
        "bias": 1.0,
        "street_index": float(STREET_ORDER.get(request.street, 0)),
        "board_count": float(len(request.board_cards)),
        "hole_count": float(len(request.hole_cards)),
        "strength_proxy": hand_strength_proxy(request.hole_cards),
        "pot": request.pot,
        "to_call": request.to_call,
        "stack": request.stack,
        "min_raise": request.min_raise,
        "pot_odds": pot_odds,
        "stack_to_pot": min(stack_to_pot, 100.0),
        "player_count": float(request.player_count),
        f"position={request.position}": 1.0,
        f"street={request.street}": 1.0,
    }


def load_training_examples(
    dataset_dir: Path,
    max_examples: int = 0,
) -> list[tuple[dict[str, float], str]]:
    actions_path = dataset_dir / "actions.csv"
    players_path = dataset_dir / "players.csv"
    hands_path = dataset_dir / "hands.csv"
    if not actions_path.exists():
        raise FileNotFoundError(f"Missing actions.csv: {actions_path}")

    players_by_hand_pos: dict[tuple[str, str], dict[str, str]] = {}
    if players_path.exists():
        with players_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                players_by_hand_pos[(row.get("hand_id", ""), row.get("position", ""))] = row

    hands_by_id: dict[str, dict[str, str]] = {}
    if hands_path.exists():
        with hands_path.open("r", newline="", encoding="utf-8") as handle:
            hands_by_id = {row.get("hand_id", ""): row for row in csv.DictReader(handle)}

    examples: list[tuple[dict[str, float], str]] = []
    with actions_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            action = normalize_action(row.get("action", ""))
            if action not in VALID_ACTIONS:
                continue
            hand = hands_by_id.get(row.get("hand_id", ""), {})
            player = players_by_hand_pos.get((row.get("hand_id", ""), row.get("player_position", "")), {})
            request = PredictionRequest(
                position=row.get("player_position", "UNK"),
                street=row.get("street", "preflop"),
                hole_cards=parse_cards(player.get("cards")),
                board_cards=parse_cards(hand.get("board_cards")),
                pot=float(hand.get("pot_from_recognition") or hand.get("pot_from_stacks") or 0.0),
                stack=float(player.get("starting_stack") or 0.0),
            )
            examples.append((request_to_features(request), action))
            if max_examples and len(examples) >= max_examples:
                break
    return examples
