from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.features import normalize_action, parse_cards, safe_float, safe_int


ACTION_PATTERN = re.compile(r"\b(fold|folds|call|calls|check|checks|bet|bets|raise|raises|all[\s_-]?in|allin)\b", re.I)


@dataclass(frozen=True)
class ExtractedEvent:
    source_file: str
    line_number: int
    frame_id: int
    provider: str
    extracted_type: str
    confidence: float
    player_position: str | None = None
    action: str | None = None
    amount: float | None = None
    cards: list[str] | None = None
    raw_event_name: str | None = None
    evidence: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract structured poker events from OCR/dealer logs. The output "
            "schema is stable for later LLM-provider comparisons."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="JSONL file or folder containing JSONL files")
    parser.add_argument("--out", required=True, type=Path, help="Output JSONL with extracted events")
    parser.add_argument("--summary-out", required=True, type=Path, help="Summary JSON path")
    parser.add_argument(
        "--provider",
        choices=("local_rules",),
        default="local_rules",
        help="Extractor provider. local_rules is deterministic and reproducible for CI/smoke runs.",
    )
    parser.add_argument("--max-files", type=int, default=0, help="Maximum JSONL files to scan. 0 means all files.")
    parser.add_argument("--max-records", type=int, default=0, help="Maximum records to read. 0 means all records.")
    parser.add_argument("--min-confidence", type=float, default=0.20)
    parser.add_argument("--include-unmatched", action="store_true")
    return parser.parse_args()


def iter_jsonl_files(path: Path, max_files: int) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    files = sorted(item for item in path.glob("*.jsonl") if item.is_file())
    for index, file_path in enumerate(files):
        if max_files and index >= max_files:
            break
        yield file_path


def compact_evidence(record: dict[str, Any]) -> str:
    event_value = record.get("event_value")
    payload = event_value if isinstance(event_value, dict) else {}
    parts = [
        f"event_name={record.get('event_name', '')}",
        f"object_type={record.get('object_type', '')}",
        f"value={payload.get('value', '')}",
        f"position={payload.get('player_position', '')}",
        f"cards={payload.get('cards', '')}",
        f"stack={payload.get('stack', '')}",
        f"diff={payload.get('diff', '')}",
    ]
    return " ".join(part for part in parts if not part.endswith("="))


def extract_amount(payload: dict[str, Any]) -> float | None:
    for key in ("amount", "bet", "raise_to", "to_call", "diff", "stack"):
        if key in payload:
            value = safe_float(payload.get(key), default=0.0)
            if value != 0.0:
                return value
    return None


class LocalRuleExtractor:
    provider = "local_rules"

    def extract(self, record: dict[str, Any], *, source_file: str, line_number: int) -> ExtractedEvent:
        event_name = str(record.get("event_name") or "")
        event_value = record.get("event_value")
        payload = event_value if isinstance(event_value, dict) else {}
        evidence = compact_evidence(record)
        player_position = payload.get("player_position")
        frame_id = safe_int(record.get("frame_id"))
        raw_value = str(payload.get("value") or "")

        action_match = ACTION_PATTERN.search(" ".join([event_name, raw_value, evidence]))
        action = normalize_action(action_match.group(1)) if action_match else ""
        if event_name == "ocr_action" and action:
            return ExtractedEvent(
                source_file=source_file,
                line_number=line_number,
                frame_id=frame_id,
                provider=self.provider,
                extracted_type="player_action",
                player_position=str(player_position) if player_position is not None else None,
                action=action,
                amount=extract_amount(payload),
                raw_event_name=event_name,
                confidence=0.92,
                evidence=evidence,
            )

        cards = parse_cards(payload.get("cards"))
        if event_name == "recognize_cards" and cards:
            return ExtractedEvent(
                source_file=source_file,
                line_number=line_number,
                frame_id=frame_id,
                provider=self.provider,
                extracted_type="card_update",
                player_position=str(player_position) if player_position is not None else None,
                cards=cards,
                raw_event_name=event_name,
                confidence=0.88,
                evidence=evidence,
            )

        if event_name == "ocr_stack" and ("stack" in payload or "diff" in payload):
            return ExtractedEvent(
                source_file=source_file,
                line_number=line_number,
                frame_id=frame_id,
                provider=self.provider,
                extracted_type="stack_update",
                player_position=str(player_position) if player_position is not None else None,
                amount=extract_amount(payload),
                raw_event_name=event_name,
                confidence=0.84,
                evidence=evidence,
            )

        return ExtractedEvent(
            source_file=source_file,
            line_number=line_number,
            frame_id=frame_id,
            provider=self.provider,
            extracted_type="unmatched",
            raw_event_name=event_name,
            confidence=0.0,
            evidence=evidence,
        )


def process_file(
    extractor: LocalRuleExtractor,
    file_path: Path,
    *,
    records_remaining: int | None,
    min_confidence: float,
    include_unmatched: bool,
) -> tuple[list[ExtractedEvent], int]:
    extracted: list[ExtractedEvent] = []
    read_count = 0
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if records_remaining is not None and read_count >= records_remaining:
                break
            read_count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                event = ExtractedEvent(
                    source_file=file_path.name,
                    line_number=line_number,
                    frame_id=0,
                    provider=extractor.provider,
                    extracted_type="unmatched",
                    confidence=0.0,
                    evidence="invalid_json",
                )
            else:
                event = extractor.extract(record, source_file=file_path.name, line_number=line_number)

            if event.confidence >= min_confidence or (include_unmatched and event.extracted_type == "unmatched"):
                extracted.append(event)
    return extracted, read_count


def summarize(events: list[ExtractedEvent], *, files_scanned: int, records_read: int) -> dict[str, Any]:
    event_types = Counter(event.extracted_type for event in events)
    actions = Counter(event.action for event in events if event.action)
    confidence_values = [event.confidence for event in events if event.confidence > 0]
    return {
        "files_scanned": files_scanned,
        "records_read": records_read,
        "records_written": len(events),
        "event_type_counts": dict(event_types.most_common()),
        "action_counts": dict(actions.most_common()),
        "mean_positive_confidence": statistics.fmean(confidence_values) if confidence_values else 0.0,
    }


def main() -> None:
    args = parse_args()
    extractor = LocalRuleExtractor()
    all_events: list[ExtractedEvent] = []
    records_read = 0
    files_scanned = 0
    max_records = args.max_records if args.max_records > 0 else None

    for file_path in iter_jsonl_files(args.input, args.max_files):
        remaining = None if max_records is None else max(max_records - records_read, 0)
        if remaining == 0:
            break
        events, read_count = process_file(
            extractor,
            file_path,
            records_remaining=remaining,
            min_confidence=args.min_confidence,
            include_unmatched=args.include_unmatched,
        )
        files_scanned += 1
        records_read += read_count
        all_events.extend(events)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for event in all_events:
            handle.write(json.dumps(asdict(event), sort_keys=True) + "\n")

    summary = summarize(all_events, files_scanned=files_scanned, records_read=records_read)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
