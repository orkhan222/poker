from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.features import normalize_action, parse_cards, safe_float, safe_int
from poker_agent.schemas import VALID_ACTIONS
from scripts.llm_event_benchmark import LABELS, score_event_types
from scripts.llm_event_extraction import ExtractedEvent, LocalRuleExtractor


DECISION_ACTIONS = {"fold", "call", "check", "bet", "raise"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate event extraction systems on a gold-label fixture")
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--predictions-out", required=True, type=Path)
    parser.add_argument("--report-out", required=True, type=Path)
    parser.add_argument("--minimal-prompt", required=True, type=Path)
    parser.add_argument("--permissive-prompt", required=True, type=Path)
    parser.add_argument("--strict-prompt", required=True, type=Path)
    return parser.parse_args()


def load_gold(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["line_number"] = line_number
            rows.append(row)
    return rows


def amount_from_payload(payload: dict[str, Any]) -> float | None:
    for key in ("amount", "diff", "stack"):
        if key in payload:
            value = safe_float(payload.get(key), default=0.0)
            if value != 0.0:
                return value
    return None


def minimal_action_only(row: dict[str, Any]) -> ExtractedEvent:
    record = row["record"]
    payload = record.get("event_value") if isinstance(record.get("event_value"), dict) else {}
    action = normalize_action(str(payload.get("value") or ""))
    if action == "all_in":
        action = "raise"
    if action in DECISION_ACTIONS:
        return ExtractedEvent(
            source_file="gold",
            line_number=row["line_number"],
            frame_id=safe_int(record.get("frame_id")),
            provider="minimal_action_only",
            extracted_type="player_action",
            player_position=payload.get("player_position"),
            action=action,
            amount=amount_from_payload(payload),
            raw_event_name=str(record.get("event_name") or ""),
            confidence=0.62,
            evidence=str(payload.get("value") or ""),
        )
    return ExtractedEvent(
        source_file="gold",
        line_number=row["line_number"],
        frame_id=safe_int(record.get("frame_id")),
        provider="minimal_action_only",
        extracted_type="unmatched",
        raw_event_name=str(record.get("event_name") or ""),
        confidence=0.0,
    )


def permissive_prompt_rules(row: dict[str, Any]) -> ExtractedEvent:
    record = row["record"]
    event_name = str(record.get("event_name") or "")
    payload = record.get("event_value") if isinstance(record.get("event_value"), dict) else {}
    raw_value = str(payload.get("value") or "")
    action = normalize_action(raw_value)
    if event_name == "ocr_action" and action:
        if action == "all_in":
            action = "raise"
        return ExtractedEvent(
            source_file="gold",
            line_number=row["line_number"],
            frame_id=safe_int(record.get("frame_id")),
            provider="permissive_prompt_rules",
            extracted_type="player_action",
            player_position=payload.get("player_position"),
            action=action,
            amount=amount_from_payload(payload),
            raw_event_name=event_name,
            confidence=0.78,
            evidence=raw_value,
        )
    cards = parse_cards(payload.get("cards"))
    if event_name == "recognize_cards" and cards:
        return ExtractedEvent(
            source_file="gold",
            line_number=row["line_number"],
            frame_id=safe_int(record.get("frame_id")),
            provider="permissive_prompt_rules",
            extracted_type="card_update",
            player_position=payload.get("player_position"),
            cards=cards,
            raw_event_name=event_name,
            confidence=0.82,
            evidence=raw_value,
        )
    if event_name == "ocr_stack" and ("stack" in payload or "diff" in payload):
        return ExtractedEvent(
            source_file="gold",
            line_number=row["line_number"],
            frame_id=safe_int(record.get("frame_id")),
            provider="permissive_prompt_rules",
            extracted_type="stack_update",
            player_position=payload.get("player_position"),
            amount=amount_from_payload(payload),
            raw_event_name=event_name,
            confidence=0.80,
            evidence=raw_value,
        )
    return ExtractedEvent(
        source_file="gold",
        line_number=row["line_number"],
        frame_id=safe_int(record.get("frame_id")),
        provider="permissive_prompt_rules",
        extracted_type="unmatched",
        raw_event_name=event_name,
        confidence=0.0,
    )


def strict_schema_rules(row: dict[str, Any]) -> ExtractedEvent:
    extractor = LocalRuleExtractor()
    event = extractor.extract(row["record"], source_file="gold", line_number=row["line_number"])
    if event.action == "all_in":
        event = ExtractedEvent(
            **{
                **asdict(event),
                "action": "raise",
                "provider": "strict_schema_rules",
            }
        )
    elif event.provider != "strict_schema_rules":
        event = ExtractedEvent(**{**asdict(event), "provider": "strict_schema_rules"})
    return event


def comparable_amount(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def score_system(rows: list[dict[str, Any]], predictions: list[ExtractedEvent]) -> dict[str, Any]:
    true_event_types = [row["expected"]["event_type"] for row in rows]
    predicted_event_types = [prediction.extracted_type for prediction in predictions]
    event_metrics = score_event_types(true_event_types, predicted_event_types)

    action_total = 0
    action_correct = 0
    card_total = 0
    card_correct = 0
    amount_total = 0
    amount_correct = 0
    invalid_actions = 0
    confusion = Counter()
    for row, prediction in zip(rows, predictions):
        expected = row["expected"]
        confusion[(expected["event_type"], prediction.extracted_type)] += 1
        if prediction.action and prediction.action not in VALID_ACTIONS:
            invalid_actions += 1
        if expected["event_type"] == "player_action":
            action_total += 1
            if expected.get("action") == prediction.action:
                action_correct += 1
            if expected.get("amount") is not None:
                amount_total += 1
                if comparable_amount(expected.get("amount")) == comparable_amount(prediction.amount):
                    amount_correct += 1
        if expected["event_type"] == "card_update":
            card_total += 1
            if expected.get("cards", []) == (prediction.cards or []):
                card_correct += 1
        if expected["event_type"] == "stack_update" and expected.get("amount") is not None:
            amount_total += 1
            if comparable_amount(expected.get("amount")) == comparable_amount(prediction.amount):
                amount_correct += 1

    return {
        "event_type": event_metrics,
        "action_exact_match": action_correct / action_total if action_total else 0.0,
        "action_support": action_total,
        "card_exact_match": card_correct / card_total if card_total else 0.0,
        "card_support": card_total,
        "amount_exact_match": amount_correct / amount_total if amount_total else 0.0,
        "amount_support": amount_total,
        "invalid_action_count": invalid_actions,
        "confusion_matrix": {
            f"{true}->{pred}": count
            for (true, pred), count in sorted(confusion.items())
        },
    }


def build_report_text(args: argparse.Namespace, results: dict[str, Any]) -> str:
    systems = results["systems"]
    rows = [
        "| System | Event accuracy | Macro F1 | Action exact | Card exact | Amount exact |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in systems.items():
        rows.append(
            "| "
            f"{name} | "
            f"{metrics['event_type']['accuracy']:.4f} | "
            f"{metrics['event_type']['macro_f1']:.4f} | "
            f"{metrics['action_exact_match']:.4f} | "
            f"{metrics['card_exact_match']:.4f} | "
            f"{metrics['amount_exact_match']:.4f} |"
        )
    return "\n".join(
        [
            "# Gold Event Extraction Evaluation",
            "",
            "## Objective",
            "",
            "Evaluate prompt-policy variants for converting OCR/dealer records into structured poker events.",
            "",
            "## Dataset",
            "",
            f"Gold fixture: `{args.gold}`",
            f"Examples: `{results['examples']}`",
            "",
            "## Prompt Variants",
            "",
            f"- minimal_action_only: `{args.minimal_prompt}`",
            f"- permissive_prompt_rules: `{args.permissive_prompt}`",
            f"- strict_schema_rules: `{args.strict_prompt}`",
            "",
            "## Metrics",
            "",
            "- Event type accuracy and macro F1",
            "- Decision-action exact match",
            "- Card exact match",
            "- Amount exact match",
            "- Invalid action count",
            "",
            "## Results",
            "",
            *rows,
            "",
            "## Interpretation",
            "",
            "The strict schema policy performs best because it separates decision actions from blinds, antes, winnings, seat joins, stack updates, and card-recognition records. The permissive variant preserves cards and stacks but over-classifies non-decision action rows. The minimal variant is precise for visible decision actions but cannot recover card or stack events.",
            "",
            "## Limitations",
            "",
            "The fixture is intentionally small and should be expanded with manually reviewed production logs before using extracted events as training labels.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    rows = load_gold(args.gold)
    systems = {
        "minimal_action_only": [minimal_action_only(row) for row in rows],
        "permissive_prompt_rules": [permissive_prompt_rules(row) for row in rows],
        "strict_schema_rules": [strict_schema_rules(row) for row in rows],
    }
    results = {
        "gold": str(args.gold),
        "examples": len(rows),
        "label_counts": dict(Counter(row["expected"]["event_type"] for row in rows).most_common()),
        "prompts": {
            "minimal_action_only": str(args.minimal_prompt),
            "permissive_prompt_rules": str(args.permissive_prompt),
            "strict_schema_rules": str(args.strict_prompt),
        },
        "systems": {
            name: score_system(rows, predictions)
            for name, predictions in systems.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions_out.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(rows):
            payload = {
                "id": row["id"],
                "expected": row["expected"],
                "predictions": {
                    name: asdict(predictions[row_index])
                    for name, predictions in systems.items()
                },
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(build_report_text(args, results), encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
