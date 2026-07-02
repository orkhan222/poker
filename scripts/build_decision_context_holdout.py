from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.features import (
    NON_DECISION_ACTIONS,
    amount_near_frame,
    estimate_big_blind,
    normalize_action,
    parse_cards,
    safe_float,
    safe_int,
    visible_board_cards,
)
from poker_agent.llm_decision_context import legal_actions_for_request
from poker_agent.schemas import PredictionRequest, VALID_ACTIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a grouped human-action holdout for LLM context evaluation")
    parser.add_argument("--data-dir", default=ROOT / "data", type=Path)
    parser.add_argument("--out", default=ROOT / "evaluation" / "decision_context_human_holdout.jsonl", type=Path)
    parser.add_argument("--report-out", default=ROOT / "reports" / "decision_context_holdout.json", type=Path)
    parser.add_argument("--hands", default=800, type=int)
    parser.add_argument("--examples-per-action", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def stable_score(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def select_hand_ids(players_path: Path, limit: int, seed: int) -> set[str]:
    heap: list[tuple[int, str]] = []
    observed: set[str] = set()
    with players_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            hand_id = str(row.get("hand_id") or "")
            if not hand_id or hand_id in observed or len(parse_cards(row.get("cards"))) < 2:
                continue
            observed.add(hand_id)
            score = stable_score(seed, hand_id)
            item = (-score, hand_id)
            if len(heap) < limit:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
    return {hand_id for _, hand_id in heap}


def load_selected_rows(path: Path, selected: set[str], key: str = "hand_id") -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get(key) or "") in selected:
                rows.append(row)
    return rows


def build_holdout(
    data_dir: Path,
    *,
    hand_limit: int,
    examples_per_action: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = select_hand_ids(data_dir / "players.csv", hand_limit, seed)
    player_rows = load_selected_rows(data_dir / "players.csv", selected)
    hand_rows = load_selected_rows(data_dir / "hands.csv", selected)
    action_rows = load_selected_rows(data_dir / "actions.csv", selected)
    stack_rows = load_selected_rows(data_dir / "stack_events.csv", selected)

    players: dict[tuple[str, str], dict[str, str]] = {
        (str(row.get("hand_id") or ""), str(row.get("position") or "")): row
        for row in player_rows
    }
    player_counts = Counter(str(row.get("hand_id") or "") for row in player_rows)
    hands = {str(row.get("hand_id") or ""): row for row in hand_rows}
    actions_by_hand: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in action_rows:
        actions_by_hand[str(row.get("hand_id") or "")].append(row)

    contributions: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for row in stack_rows:
        diff = safe_float(row.get("diff"))
        if diff < 0:
            contributions[
                (str(row.get("hand_id") or ""), str(row.get("player_position") or ""))
            ].append((safe_int(row.get("frame_id")), abs(diff)))
    for events in contributions.values():
        events.sort()

    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for hand_id in sorted(actions_by_hand, key=lambda value: stable_score(seed, value)):
        rows = sorted(actions_by_hand[hand_id], key=lambda row: safe_int(row.get("frame_id")))
        final_board = parse_cards((hands.get(hand_id) or {}).get("board_cards"))
        used_events: set[tuple[str, str, int, float]] = set()
        committed: dict[str, float] = defaultdict(float)
        current_street = ""
        big_blind = estimate_big_blind(rows, contributions)
        last_raise_size = big_blind
        running_pot = 0.0
        history: list[dict[str, Any]] = []
        previous_frame: int | None = None
        for row in rows:
            action = normalize_action(str(row.get("action") or ""))
            if action == "all_in":
                action = "raise"
            position = str(row.get("player_position") or "")
            street = str(row.get("street") or "preflop").lower()
            frame_id = safe_int(row.get("frame_id"))
            if street != current_street:
                committed = defaultdict(float)
                current_street = street
                last_raise_size = big_blind
            amount = amount_near_frame(
                contributions,
                used_events,
                hand_id,
                position,
                frame_id,
            )
            highest_commit = max(committed.values(), default=0.0)
            player_commit = committed[position]
            to_call = max(0.0, highest_commit - player_commit)
            if action in VALID_ACTIONS:
                player = players.get((hand_id, position), {})
                cards = parse_cards(player.get("cards"))
                stack = safe_float(player.get("starting_stack"))
                request = PredictionRequest(
                    position=position or "UNK",
                    street=street,
                    hole_cards=cards,
                    board_cards=visible_board_cards(final_board, street),
                    pot=max(running_pot, sum(committed.values()), big_blind, 0.0),
                    to_call=to_call,
                    stack=max(stack - player_commit, 0.0) if stack > 0 else 0.0,
                    min_raise=max(last_raise_size, big_blind, 0.0),
                    player_count=max(player_counts.get(hand_id, 0), 2),
                    betting_history=list(history[-12:]),
                )
                if len(cards) >= 2 and action in legal_actions_for_request(request):
                    record = {
                        "id": f"{hand_id}:{frame_id}:{position}",
                        "source": {
                            "hand_id": hand_id,
                            "frame_id": frame_id,
                            "source_file": row.get("source_file"),
                        },
                        "game_state": {
                            "position": request.position,
                            "street": request.street,
                            "hole_cards": request.hole_cards,
                            "board_cards": request.board_cards,
                            "pot": round(request.pot, 4),
                            "to_call": round(request.to_call, 4),
                            "stack": round(request.stack, 4),
                            "min_raise": round(request.min_raise, 4),
                            "player_count": request.player_count,
                            "betting_history": request.betting_history,
                        },
                        "expected_action": action,
                    }
                    candidates[action].append(record)

            if amount > 0:
                before_highest = max(committed.values(), default=0.0)
                committed[position] += amount
                running_pot += amount
                if action not in NON_DECISION_ACTIONS and committed[position] > before_highest:
                    last_raise_size = max(committed[position] - before_highest, big_blind)
            if action in VALID_ACTIONS:
                history.append(
                    {
                        "player_position": position,
                        "action": action,
                        "amount": round(amount, 4),
                        "street": street,
                        "frame_delta": max(frame_id - previous_frame, 0) if previous_frame is not None else 0,
                    }
                )
                previous_frame = frame_id

    selected_records: list[dict[str, Any]] = []
    for action in ("fold", "check", "call", "bet", "raise"):
        ranked = sorted(
            candidates.get(action, []),
            key=lambda row: stable_score(seed, str(row["id"])),
        )
        selected_records.extend(ranked[:examples_per_action])
    selected_records.sort(key=lambda row: str(row["id"]))
    distribution = Counter(str(row["expected_action"]) for row in selected_records)
    report = {
        "status": "PASS" if all(distribution.get(action, 0) >= examples_per_action for action in ("fold", "check", "call", "bet", "raise")) else "PARTIAL",
        "dataset_kind": "reconstructed_human_holdout",
        "source": str(data_dir),
        "seed": seed,
        "selected_hand_groups": len({row["source"]["hand_id"] for row in selected_records}),
        "examples": len(selected_records),
        "class_distribution": dict(sorted(distribution.items())),
        "leakage_controls": [
            "hand-group deterministic selection",
            "only actions before the target decision are included in betting_history",
            "betting history is capped to the 12 most recent observable events",
            "board cards are truncated to the decision street",
            "final pot and winner fields are excluded",
        ],
    }
    return selected_records, report


def main() -> None:
    args = parse_args()
    rows, report = build_holdout(
        args.data_dir,
        hand_limit=args.hands,
        examples_per_action=args.examples_per_action,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    args.report_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
