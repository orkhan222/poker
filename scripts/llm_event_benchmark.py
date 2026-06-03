from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.features import normalize_action, parse_cards, safe_float, safe_int
from poker_agent.schemas import VALID_ACTIONS
from scripts.llm_event_extraction import ExtractedEvent, LocalRuleExtractor, iter_jsonl_files


LABELS = ("player_action", "card_update", "stack_update", "unmatched")


@dataclass(frozen=True)
class LabeledRecord:
    source_file: str
    line_number: int
    record: dict[str, Any]
    event_type: str
    action: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark OCR/dealer-log event extraction")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--methodology-out", required=True, type=Path)
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--provider", choices=("local_rules",), default="local_rules")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--min-confidence", type=float, default=0.20)
    return parser.parse_args()


def weak_label(record: dict[str, Any]) -> tuple[str, str | None]:
    event_name = str(record.get("event_name") or "")
    event_value = record.get("event_value")
    payload = event_value if isinstance(event_value, dict) else {}
    value = str(payload.get("value") or "")
    action = normalize_action(value)

    if event_name == "ocr_action" and action in VALID_ACTIONS:
        return "player_action", action
    if event_name == "recognize_cards" and parse_cards(payload.get("cards")):
        return "card_update", None
    if event_name == "ocr_stack" and ("stack" in payload or "diff" in payload):
        return "stack_update", None
    return "unmatched", None


def iter_labeled_records(input_path: Path, max_files: int, max_records: int) -> Iterable[LabeledRecord]:
    records_read = 0
    for file_path in iter_jsonl_files(input_path, max_files):
        with file_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if max_records and records_read >= max_records:
                    return
                records_read += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    event_type, action = "unmatched", None
                    record = {}
                else:
                    event_type, action = weak_label(record)
                yield LabeledRecord(
                    source_file=file_path.name,
                    line_number=line_number,
                    record=record,
                    event_type=event_type,
                    action=action,
                )


def value_only_baseline(item: LabeledRecord) -> ExtractedEvent:
    event_value = item.record.get("event_value")
    payload = event_value if isinstance(event_value, dict) else {}
    action = normalize_action(str(payload.get("value") or ""))
    if action in VALID_ACTIONS:
        return ExtractedEvent(
            source_file=item.source_file,
            line_number=item.line_number,
            frame_id=safe_int(item.record.get("frame_id")),
            provider="value_only_baseline",
            extracted_type="player_action",
            player_position=payload.get("player_position"),
            action=action,
            amount=safe_float(payload.get("amount"), default=0.0) or None,
            raw_event_name=str(item.record.get("event_name") or ""),
            confidence=0.65,
            evidence=str(payload.get("value") or ""),
        )
    return ExtractedEvent(
        source_file=item.source_file,
        line_number=item.line_number,
        frame_id=safe_int(item.record.get("frame_id")),
        provider="value_only_baseline",
        extracted_type="unmatched",
        raw_event_name=str(item.record.get("event_name") or ""),
        confidence=0.0,
    )


def score_event_types(true_labels: list[str], predicted_labels: list[str]) -> dict[str, Any]:
    total = len(true_labels)
    correct = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
    per_label: dict[str, dict[str, float]] = {}
    for label in LABELS:
        tp = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == label and pred == label)
        fp = sum(1 for true, pred in zip(true_labels, predicted_labels) if true != label and pred == label)
        fn = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == label and pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for true in true_labels if true == label),
        }
    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(metric["f1"] for metric in per_label.values()) / len(LABELS),
        "class_counts": dict(Counter(true_labels).most_common()),
        "predicted_counts": dict(Counter(predicted_labels).most_common()),
        "per_label": per_label,
    }


def score_actions(items: list[LabeledRecord], predictions: list[ExtractedEvent]) -> dict[str, Any]:
    action_pairs = [
        (item.action, prediction.action)
        for item, prediction in zip(items, predictions)
        if item.event_type == "player_action"
    ]
    total = len(action_pairs)
    correct = sum(1 for true, pred in action_pairs if true == pred)
    true_counts = Counter(true for true, _ in action_pairs if true)
    pred_counts = Counter(pred for _, pred in action_pairs if pred)
    return {
        "support": total,
        "exact_match_accuracy": correct / total if total else 0.0,
        "true_action_counts": dict(true_counts.most_common()),
        "predicted_action_counts": dict(pred_counts.most_common()),
    }


def evaluate(items: list[LabeledRecord], predictions: list[ExtractedEvent]) -> dict[str, Any]:
    true_labels = [item.event_type for item in items]
    predicted_labels = [event.extracted_type for event in predictions]
    return {
        "event_type": score_event_types(true_labels, predicted_labels),
        "action": score_actions(items, predictions),
    }


def methodology_markdown(
    *,
    prompt_text: str,
    report: dict[str, Any],
    methodology_out: Path,
) -> None:
    baseline = report["systems"]["value_only_baseline"]["event_type"]
    candidate = report["systems"]["local_rules"]["event_type"]
    action = report["systems"]["local_rules"]["action"]
    text = f"""# Event Extraction Benchmark

## Objective

Convert OCR/dealer log records into structured poker events that can improve
betting-history reconstruction and downstream action prediction features.

## Methodology

The benchmark uses deterministic weak labels derived from existing structured
log fields:

- `ocr_action` with a normalized action becomes `player_action`.
- `recognize_cards` with parseable cards becomes `card_update`.
- `ocr_stack` with stack or diff values becomes `stack_update`.
- Other records are treated as `unmatched`.

This is not a replacement for hand-reviewed labels. It is a reproducible
screening benchmark for extractor regressions and prompt experiments.

## Prompt Template

```text
{prompt_text.strip()}
```

## Results

| System | Event accuracy | Event macro F1 | Action exact match |
| --- | ---: | ---: | ---: |
| value_only_baseline | {baseline['accuracy']:.4f} | {baseline['macro_f1']:.4f} | {report['systems']['value_only_baseline']['action']['exact_match_accuracy']:.4f} |
| local_rules | {candidate['accuracy']:.4f} | {candidate['macro_f1']:.4f} | {action['exact_match_accuracy']:.4f} |

## Observed Limitations

- Weak labels come from structured fields and can inherit OCR/logging errors.
- The local extractor is deterministic and does not resolve ambiguous natural-language dealer messages.
- True LLM-provider runs should be compared against a manually reviewed stratified sample before training labels are trusted.

Full machine-readable results are saved next to this file.
"""
    methodology_out.parent.mkdir(parents=True, exist_ok=True)
    methodology_out.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    items = list(iter_labeled_records(args.input, args.max_files, args.max_records))
    extractor = LocalRuleExtractor()
    baseline_predictions = [value_only_baseline(item) for item in items]
    local_predictions = [
        extractor.extract(item.record, source_file=item.source_file, line_number=item.line_number)
        for item in items
    ]

    report = {
        "input": str(args.input),
        "provider": args.provider,
        "max_files": args.max_files,
        "max_records": args.max_records,
        "records_evaluated": len(items),
        "weak_label_counts": dict(Counter(item.event_type for item in items).most_common()),
        "systems": {
            "value_only_baseline": evaluate(items, baseline_predictions),
            "local_rules": evaluate(items, local_predictions),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    prompt_text = args.prompt.read_text(encoding="utf-8")
    methodology_markdown(prompt_text=prompt_text, report=report, methodology_out=args.methodology_out)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
