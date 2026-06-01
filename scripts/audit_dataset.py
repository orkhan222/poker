from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.features import load_training_examples, normalize_action, parse_cards, safe_float, safe_int
from poker_agent.schemas import VALID_ACTIONS


CRITICAL_FEATURES = (
    "pot",
    "to_call",
    "min_raise",
    "stack",
    "pot_odds",
    "spr",
    "strength_proxy",
    "street_action_count",
    "street_aggression_ratio",
    "hero_commitment_ratio",
    "table_commitment_pressure",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit poker dataset quality and leakage risks")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", default=Path("reports/dataset_audit.json"), type=Path)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional row limit per raw CSV file. 0 scans all rows.",
    )
    parser.add_argument(
        "--max-feature-examples",
        type=int,
        default=50000,
        help="Number of extracted feature rows to audit. Use 0 for all.",
    )
    parser.add_argument(
        "--missing-hole-cards",
        choices=("drop", "flag", "keep"),
        default="flag",
        help="Use flag to audit the magnitude of OCR-missing card rows.",
    )
    return parser.parse_args()


def limited_rows(path: Path, max_rows: int) -> Any:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            if max_rows and index > max_rows:
                break
            yield row


def audit_actions(path: Path, max_rows: int) -> dict[str, Any]:
    raw_actions: Counter[str] = Counter()
    normalized_actions: Counter[str] = Counter()
    street_counts: Counter[str] = Counter()
    decision_rows = 0
    non_decision_rows = 0
    missing_position = 0
    missing_street = 0
    frame_regressions = 0
    last_frame_by_hand: dict[str, int] = {}
    total_rows = 0

    for row in limited_rows(path, max_rows):
        total_rows += 1
        raw_action = str(row.get("action", "")).strip().lower()
        action = normalize_action(raw_action)
        hand_id = str(row.get("hand_id", ""))
        frame_id = safe_int(row.get("frame_id"))
        raw_actions[raw_action or "missing"] += 1
        normalized_actions[action] += 1
        street = str(row.get("street") or "missing").lower()
        street_counts[street] += 1
        if action in VALID_ACTIONS:
            decision_rows += 1
        else:
            non_decision_rows += 1
        if not row.get("player_position"):
            missing_position += 1
        if not row.get("street"):
            missing_street += 1
        if hand_id:
            previous = last_frame_by_hand.get(hand_id)
            if previous is not None and frame_id < previous:
                frame_regressions += 1
            last_frame_by_hand[hand_id] = frame_id

    return {
        "rows": total_rows,
        "hands_seen": len(last_frame_by_hand),
        "decision_rows": decision_rows,
        "non_decision_rows": non_decision_rows,
        "raw_action_counts": dict(raw_actions.most_common()),
        "normalized_action_counts": dict(normalized_actions.most_common()),
        "street_counts": dict(street_counts.most_common()),
        "missing_position_rate": missing_position / total_rows if total_rows else 0.0,
        "missing_street_rate": missing_street / total_rows if total_rows else 0.0,
        "frame_regressions": frame_regressions,
    }


def audit_players(path: Path, max_rows: int) -> dict[str, Any]:
    position_counts: Counter[str] = Counter()
    card_count_distribution: Counter[int] = Counter()
    missing_cards = 0
    partial_cards = 0
    complete_cards = 0
    zero_starting_stack = 0
    total_rows = 0

    for row in limited_rows(path, max_rows):
        total_rows += 1
        position_counts[str(row.get("position") or "missing")] += 1
        card_count = min(len(parse_cards(row.get("cards"))), 2)
        card_count_distribution[card_count] += 1
        if card_count == 0:
            missing_cards += 1
        elif card_count == 1:
            partial_cards += 1
        else:
            complete_cards += 1
        if safe_float(row.get("starting_stack")) <= 0:
            zero_starting_stack += 1

    return {
        "rows": total_rows,
        "position_cardinality": len(position_counts),
        "top_positions": dict(position_counts.most_common(20)),
        "card_count_distribution": {str(key): value for key, value in sorted(card_count_distribution.items())},
        "missing_hole_card_rate": missing_cards / total_rows if total_rows else 0.0,
        "partial_hole_card_rate": partial_cards / total_rows if total_rows else 0.0,
        "complete_hole_card_rate": complete_cards / total_rows if total_rows else 0.0,
        "zero_starting_stack_rate": zero_starting_stack / total_rows if total_rows else 0.0,
    }


def audit_hands(path: Path, max_rows: int) -> dict[str, Any]:
    board_count_distribution: Counter[int] = Counter()
    total_rows = 0
    for row in limited_rows(path, max_rows):
        total_rows += 1
        board_count_distribution[len(parse_cards(row.get("board_cards")))] += 1
    return {
        "rows": total_rows,
        "board_count_distribution": {str(key): value for key, value in sorted(board_count_distribution.items())},
    }


def audit_stack_events(path: Path, max_rows: int) -> dict[str, Any]:
    total_rows = 0
    negative = 0
    positive = 0
    zero = 0
    missing_join_keys = 0
    abs_diff_bins: Counter[str] = Counter()
    for row in limited_rows(path, max_rows):
        total_rows += 1
        diff = safe_float(row.get("diff"))
        if diff < 0:
            negative += 1
        elif diff > 0:
            positive += 1
        else:
            zero += 1
        if not row.get("hand_id") or not row.get("player_position"):
            missing_join_keys += 1
        amount = abs(diff)
        if amount == 0:
            abs_diff_bins["0"] += 1
        elif amount < 1:
            abs_diff_bins["(0,1)"] += 1
        elif amount < 5:
            abs_diff_bins["[1,5)"] += 1
        elif amount < 20:
            abs_diff_bins["[5,20)"] += 1
        else:
            abs_diff_bins[">=20"] += 1
    return {
        "rows": total_rows,
        "negative_diff_rows": negative,
        "positive_diff_rows": positive,
        "zero_diff_rows": zero,
        "missing_join_key_rate": missing_join_keys / total_rows if total_rows else 0.0,
        "abs_diff_bins": dict(abs_diff_bins),
    }


def audit_extracted_features(
    dataset: Path,
    *,
    max_feature_examples: int,
    missing_hole_cards: str,
) -> dict[str, Any]:
    examples = load_training_examples(
        dataset,
        max_examples=max_feature_examples,
        require_hole_cards=missing_hole_cards == "drop",
        missing_hole_cards=missing_hole_cards,
        include_hand_id=False,
    )
    label_counts = Counter(label for _, label in examples)
    zero_rates: dict[str, float] = {}
    missing_rates: dict[str, float] = {}
    for feature in CRITICAL_FEATURES:
        zero = sum(1 for row, _ in examples if float(row.get(feature, 0.0)) == 0.0)
        missing = sum(1 for row, _ in examples if feature not in row)
        zero_rates[feature] = zero / len(examples) if examples else 0.0
        missing_rates[feature] = missing / len(examples) if examples else 0.0
    return {
        "examples": len(examples),
        "label_counts": dict(label_counts.most_common()),
        "label_distribution": {
            label: count / len(examples) for label, count in sorted(label_counts.items())
        } if examples else {},
        "critical_feature_zero_rates": zero_rates,
        "critical_feature_missing_rates": missing_rates,
    }


def derive_findings(report: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    players = report.get("players", {})
    features = report.get("features", {})
    label_distribution = features.get("label_distribution", {})

    complete_rate = float(players.get("complete_hole_card_rate", 0.0))
    if complete_rate < 0.35:
        findings.append(
            {
                "severity": "blocker",
                "issue": "Hole-card coverage is too low for card-strength modeling.",
                "expected_impact": "High. The model will underuse the most important poker signal.",
                "recommendation": "Improve OCR/card extraction or train separate visible-card and no-card models.",
            }
        )

    max_label_share = max(label_distribution.values(), default=0.0)
    if max_label_share > 0.45:
        findings.append(
            {
                "severity": "high",
                "issue": "Target distribution is dominated by one action class.",
                "expected_impact": "High. Accuracy can improve while minority action recall remains poor.",
                "recommendation": "Select on macro F1/balanced accuracy and use weighted/focal loss or train-split resampling.",
            }
        )

    zero_rates = features.get("critical_feature_zero_rates", {})
    for feature in ("to_call", "min_raise", "pot_odds", "strength_proxy"):
        if float(zero_rates.get(feature, 0.0)) > 0.80:
            findings.append(
                {
                    "severity": "high",
                    "issue": f"Critical feature `{feature}` is zero for most audited examples.",
                    "expected_impact": "Medium to high depending on street/action mix.",
                    "recommendation": "Validate extraction against raw hand histories and add missingness-specific model paths.",
                }
            )

    actions = report.get("actions", {})
    if int(actions.get("frame_regressions", 0)) > 0:
        findings.append(
            {
                "severity": "medium",
                "issue": "Frame ordering regressions were detected inside hands.",
                "expected_impact": "Medium. Temporal features can be corrupted if action ordering is unstable.",
                "recommendation": "Sort by a canonical event id or normalize frame ordering during dataset construction.",
            }
        )

    return findings


def main() -> None:
    args = parse_args()
    required = {
        "actions": args.dataset / "actions.csv",
        "players": args.dataset / "players.csv",
        "hands": args.dataset / "hands.csv",
        "stack_events": args.dataset / "stack_events.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise SystemExit(f"Missing dataset files: {missing}")

    report: dict[str, Any] = {
        "dataset": str(args.dataset),
        "files": {
            name: {"path": str(path), "bytes": path.stat().st_size}
            for name, path in required.items()
        },
        "actions": audit_actions(required["actions"], args.max_rows),
        "players": audit_players(required["players"], args.max_rows),
        "hands": audit_hands(required["hands"], args.max_rows),
        "stack_events": audit_stack_events(required["stack_events"], args.max_rows),
        "features": audit_extracted_features(
            args.dataset,
            max_feature_examples=args.max_feature_examples,
            missing_hole_cards=args.missing_hole_cards,
        ),
    }
    report["findings"] = derive_findings(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"report={args.out}")
    print(json.dumps({"findings": report["findings"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
