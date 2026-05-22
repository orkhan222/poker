"""Build poker ML CSV datasets from HoldemHub-style JSON/JSONL logs.

The script is designed for large folders with thousands of small log files:
it walks the input directory, parses one source at a time, and streams rows to
CSV instead of keeping the whole dataset in memory.

Usage:
    python build_poker_dataset_optimized.py --input C:/path/logs --out-dir out
    python build_poker_dataset_optimized.py --input C:/path/raw --recursive
    python build_poker_dataset_optimized.py --input koV7jxEzrQs.json --out-dir out
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional


DEALER_HAND_PATTERN = re.compile(
    r"hand\s+#(\d+)\s+([^\s]+)\s+wins\s+(?:pot\s+)?([0-9.]+)(?:\s+pot)?",
    re.IGNORECASE,
)

HAND_FIELDS = [
    "hand_id",
    "hand_index",
    "local_hand_index",
    "source_file",
    "start_frame",
    "end_frame",
    "board_cards",
    "total_actions",
    "total_stack_events",
    "winner_positions",
    "pot_from_stacks",
    "pot_from_recognition",
    "dealer_hand_number",
    "dealer_winner",
    "dealer_pot",
]

PLAYER_FIELDS = [
    "hand_id",
    "hand_index",
    "local_hand_index",
    "source_file",
    "position",
    "nickname",
    "cards",
    "starting_stack",
    "ending_stack",
    "stack_delta",
]

ACTION_FIELDS = [
    "hand_id",
    "hand_index",
    "local_hand_index",
    "source_file",
    "frame_id",
    "player_position",
    "player_nickname",
    "action",
    "street",
]

STACK_FIELDS = [
    "hand_id",
    "hand_index",
    "local_hand_index",
    "source_file",
    "frame_id",
    "player_position",
    "event",
    "stack",
    "diff",
    "stack_after_event",
]


@dataclass
class PlayerState:
    position: str
    nickname: Optional[str] = None
    cards: list[str] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    stack_events: list[dict[str, Any]] = field(default_factory=list)
    starting_stack: Optional[float] = None
    ending_stack: Optional[float] = None

    @property
    def stack_delta(self) -> Optional[float]:
        if self.starting_stack is None or self.ending_stack is None:
            return None
        return round(self.ending_stack - self.starting_stack, 6)


@dataclass
class HandState:
    local_index: int
    start_frame: int
    source: str
    players: dict[str, PlayerState] = field(default_factory=dict)
    board_cards: list[str] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    stack_events: list[dict[str, Any]] = field(default_factory=list)
    pot_updates: list[dict[str, Any]] = field(default_factory=list)
    dealer_results: list[dict[str, Any]] = field(default_factory=list)
    end_frame: Optional[int] = None

    def ensure_player(
        self,
        position: str,
        nicknames: dict[str, str],
        stacks: dict[str, float],
    ) -> PlayerState:
        player = self.players.get(position)
        if player is None:
            player = PlayerState(
                position=position,
                nickname=nicknames.get(position),
                starting_stack=stacks.get(position),
            )
            self.players[position] = player
            return player

        if player.nickname is None:
            player.nickname = nicknames.get(position)
        if player.starting_stack is None:
            player.starting_stack = stacks.get(position)
        return player

    def street(self) -> str:
        board_len = len(self.board_cards)
        if board_len >= 5:
            return "river"
        if board_len == 4:
            return "turn"
        if board_len == 3:
            return "flop"
        return "preflop"

    def card_signature(self) -> str:
        parts: list[str] = []
        for position in sorted(self.players):
            cards = self.players[position].cards
            if cards:
                parts.append(f"{position}:{','.join(cards)}")
        if self.board_cards:
            parts.append(f"board:{','.join(self.board_cards)}")
        raw = "|".join(parts) or f"{self.source}:{self.local_index}:{self.start_frame}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="File or folder")
    parser.add_argument("--out-dir", default=Path("poker_dataset"), type=Path)
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search nested folders too. Enabled automatically for directories.",
    )
    parser.add_argument(
        "--extensions",
        default=".json,.jsonl",
        help="Comma separated source extensions. Default: .json,.jsonl",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=0,
        help="Optional debug limit. 0 means no limit.",
    )
    return parser.parse_args()


def parse_float(raw: Any) -> Optional[float]:
    if raw is None or raw == "":
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip().replace(",", ".")
    text = re.sub(r"[^0-9.+-]", "", text)
    if text.count(".") > 1:
        head, *tail = text.split(".")
        text = head + "." + "".join(tail)
    try:
        return float(text)
    except ValueError:
        return None


def unique_cards(cards: Any) -> list[str]:
    if not isinstance(cards, list):
        return []
    seen: set[str] = set()
    result: list[str] = []
    for card in cards:
        text = str(card).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def sanitize_source(source: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "-", source).strip("-") or "game"


def parse_dealer_results(text: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for match in DEALER_HAND_PATTERN.finditer(text or ""):
        results.append(
            {
                "hand_number": int(match.group(1)),
                "winner": match.group(2).strip(" .,:;\n"),
                "pot": parse_float(match.group(3)),
            }
        )
    return results


def unwrap_events(payload: Any) -> Iterator[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return

    if not isinstance(payload, dict):
        return

    for key in ("events", "logs", "frames", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            yield from unwrap_events(value)
            return

    yield payload


def iter_events(path: Path) -> Iterator[dict[str, Any]]:
    """Read newline-delimited JSON first, with fallback for JSON arrays/objects."""
    jsonl_error: Optional[json.JSONDecodeError] = None
    decoded_lines = 0
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    jsonl_error = exc
                    break
                decoded_lines += 1
                yield from unwrap_events(event)
            else:
                return
    except UnicodeDecodeError:
        raise

    if decoded_lines:
        assert jsonl_error is not None
        raise ValueError(f"{path}: invalid JSONL near line {jsonl_error.lineno}") from jsonl_error

    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        yield from unwrap_events(payload)
    except json.JSONDecodeError as exc:
        first_error = jsonl_error or exc
        raise ValueError(f"{path}: invalid JSON/JSONL near line {first_error.lineno}") from exc


def is_hand_activity(event_name: str, object_type: Optional[str], value: dict[str, Any]) -> bool:
    if event_name in {"ocr_action", "ocr_stack", "recognize_pot", "recognize_round_bet"}:
        return True
    return (
        event_name == "recognize_cards"
        and value.get("value") == "update_cards"
        and object_type in {"player", "table", None}
    )


def process_file(path: Path) -> Iterator[HandState]:
    source_label = path.stem
    nicknames: dict[str, str] = {}
    stacks: dict[str, float] = {}
    current: Optional[HandState] = None
    local_index = 0
    awaiting_new_hand = True

    def new_hand(frame_id: int) -> HandState:
        nonlocal current, local_index, awaiting_new_hand
        current = HandState(local_index=local_index, start_frame=frame_id, source=source_label)
        local_index += 1
        awaiting_new_hand = False
        return current

    for event in iter_events(path):
        frame_id = int(event.get("frame_id") or event.get("frame") or 0)
        event_name = str(event.get("event_name") or event.get("name") or "")
        object_type = event.get("object_type")
        value = event.get("event_value")
        if not isinstance(value, dict):
            value = event.get("value") if isinstance(event.get("value"), dict) else {}

        if event_name == "ocr_nickname":
            position = value.get("player_position")
            nickname = value.get("value")
            if position and nickname:
                nicknames[str(position)] = str(nickname)
                if current is not None:
                    current.ensure_player(str(position), nicknames, stacks).nickname = str(nickname)
            continue

        if (
            event_name == "recognize_cards"
            and object_type == "table"
            and value.get("value") == "remove_cards"
        ):
            if current is not None:
                current.end_frame = frame_id
                yield current
                current = None
            awaiting_new_hand = True
            continue

        if current is None:
            if awaiting_new_hand and not is_hand_activity(event_name, object_type, value):
                continue
            current = new_hand(frame_id)

        current.end_frame = frame_id

        if event_name == "recognize_cards":
            cards_value = value.get("value")
            if object_type == "player":
                position = value.get("player_position")
                if position:
                    player = current.ensure_player(str(position), nicknames, stacks)
                    if cards_value == "update_cards":
                        cards = unique_cards(value.get("cards"))
                        if len(cards) >= len(player.cards):
                            player.cards = cards
                    elif cards_value == "remove_cards":
                        player.cards = []
            elif object_type == "table" and cards_value == "update_cards":
                board = unique_cards(value.get("cards"))
                if len(board) >= len(current.board_cards):
                    current.board_cards = board
            continue

        if event_name == "ocr_action":
            position = value.get("player_position")
            action = value.get("value")
            if position and action:
                player = current.ensure_player(str(position), nicknames, stacks)
                row = {
                    "frame_id": frame_id,
                    "player_position": str(position),
                    "player_nickname": player.nickname,
                    "action": str(action).strip().lower(),
                    "street": current.street(),
                }
                player.actions.append(row)
                current.actions.append(row)
            continue

        if event_name == "ocr_stack":
            position = value.get("player_position")
            stack_event = value.get("value")
            if position and stack_event:
                position = str(position)
                player = current.ensure_player(position, nicknames, stacks)
                stack_value = parse_float(value.get("stack"))
                diff_value = parse_float(value.get("diff"))
                row = {
                    "frame_id": frame_id,
                    "player_position": position,
                    "event": str(stack_event),
                    "stack": stack_value,
                    "diff": diff_value,
                }
                if stack_event == "update_stack":
                    previous_stack = stacks.get(position)
                    if player.starting_stack is None:
                        if previous_stack is not None:
                            player.starting_stack = previous_stack
                        elif stack_value is not None and diff_value is not None:
                            player.starting_stack = stack_value - diff_value
                    if stack_value is not None:
                        player.ending_stack = stack_value
                        stacks[position] = stack_value
                elif stack_event == "stack_removed":
                    stacks.pop(position, None)
                row["stack_after_event"] = stacks.get(position)
                player.stack_events.append(row)
                current.stack_events.append(row)
            continue

        if event_name == "recognize_pot":
            current.pot_updates.append({"frame_id": frame_id, "pot": parse_float(value.get("pot"))})
            continue

        if event_name == "dealer_message":
            current.dealer_results.extend(parse_dealer_results(str(value.get("text", ""))))

    if current is not None:
        yield current


def rows_for_hand(hand: HandState, global_index: int) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    source_tag = sanitize_source(hand.source)
    hand_id = f"{source_tag}_H{global_index:06d}_{hand.card_signature()}"
    winners: list[str] = []
    positive_delta = 0.0
    player_rows: list[dict[str, Any]] = []

    for position, player in sorted(hand.players.items()):
        delta = player.stack_delta
        if delta is not None and delta > 0:
            winners.append(position)
            positive_delta += delta
        player_rows.append(
            {
                "hand_id": hand_id,
                "hand_index": global_index,
                "local_hand_index": hand.local_index,
                "source_file": hand.source,
                "position": position,
                "nickname": player.nickname,
                "cards": " ".join(player.cards),
                "starting_stack": player.starting_stack,
                "ending_stack": player.ending_stack,
                "stack_delta": delta,
            }
        )

    dealer = hand.dealer_results[-1] if hand.dealer_results else {}
    hand_row = {
        "hand_id": hand_id,
        "hand_index": global_index,
        "local_hand_index": hand.local_index,
        "source_file": hand.source,
        "start_frame": hand.start_frame,
        "end_frame": hand.end_frame,
        "board_cards": " ".join(hand.board_cards),
        "total_actions": len(hand.actions),
        "total_stack_events": len(hand.stack_events),
        "winner_positions": " ".join(winners),
        "pot_from_stacks": round(positive_delta, 6) if positive_delta > 0 else None,
        "pot_from_recognition": max((p["pot"] for p in hand.pot_updates if p["pot"] is not None), default=None),
        "dealer_hand_number": dealer.get("hand_number"),
        "dealer_winner": dealer.get("winner"),
        "dealer_pot": dealer.get("pot"),
    }

    action_rows = [
        {
            "hand_id": hand_id,
            "hand_index": global_index,
            "local_hand_index": hand.local_index,
            "source_file": hand.source,
            **action,
        }
        for action in hand.actions
    ]

    stack_rows = [
        {
            "hand_id": hand_id,
            "hand_index": global_index,
            "local_hand_index": hand.local_index,
            "source_file": hand.source,
            **event,
        }
        for event in hand.stack_events
    ]

    return hand_row, player_rows, action_rows, stack_rows


def discover_sources(input_path: Path, extensions: set[str], recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    globber = input_path.rglob if recursive else input_path.glob
    return sorted(
        path
        for path in globber("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def open_writer(path: Path, fields: list[str]) -> tuple[Any, csv.DictWriter]:
    handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    return handle, writer


def main() -> None:
    args = parse_args()
    input_path = args.input
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    extensions = {ext.strip().lower() for ext in args.extensions.split(",") if ext.strip()}
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = discover_sources(input_path, extensions, recursive=True if input_path.is_dir() else args.recursive)
    if args.limit_files:
        sources = sources[: args.limit_files]
    if not sources:
        raise FileNotFoundError(f"No source files found in {input_path}")

    hand_handle, hand_writer = open_writer(out_dir / "hands.csv", HAND_FIELDS)
    player_handle, player_writer = open_writer(out_dir / "players.csv", PLAYER_FIELDS)
    action_handle, action_writer = open_writer(out_dir / "actions.csv", ACTION_FIELDS)
    stack_handle, stack_writer = open_writer(out_dir / "stack_events.csv", STACK_FIELDS)

    global_index = 0
    failed_files = 0
    try:
        for source in sources:
            try:
                for hand in process_file(source):
                    hand_row, player_rows, action_rows, stack_rows = rows_for_hand(hand, global_index)
                    hand_writer.writerow(hand_row)
                    player_writer.writerows(player_rows)
                    action_writer.writerows(action_rows)
                    stack_writer.writerows(stack_rows)
                    global_index += 1
            except Exception as exc:
                failed_files += 1
                print(f"WARNING: skipped {source}: {exc}")
    finally:
        for handle in (hand_handle, player_handle, action_handle, stack_handle):
            handle.close()

    print(
        f"Done. Parsed {global_index} hands from {len(sources) - failed_files}/"
        f"{len(sources)} files into {out_dir}"
    )


if __name__ == "__main__":
    main()
