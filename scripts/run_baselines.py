from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.baselines import BASELINE_SPECS, baseline_names, build_baseline_policy, transform_examples_for_baseline
from poker_agent.evaluator import evaluate_policy
from poker_agent.features import load_training_examples
from poker_agent.slices import evaluate_policy_slices
from poker_agent.validation import stratified_group_holdout_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate concrete poker policy baselines")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", default=Path("reports/baseline_report.json"), type=Path)
    parser.add_argument(
        "--baselines",
        default=",".join(baseline_names()),
        help=f"Comma-separated baselines. Available: {', '.join(baseline_names())}",
    )
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--class-weighting", choices=("none", "sqrt_balanced", "balanced"), default="sqrt_balanced")
    parser.add_argument("--max-class-weight", type=float, default=6.0)
    parser.add_argument(
        "--missing-hole-cards",
        choices=("drop", "flag", "keep"),
        default="flag",
        help="Feature-loader policy for missing hole cards.",
    )
    return parser.parse_args()


def parse_baselines(raw: str) -> list[str]:
    names = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(names) - set(BASELINE_SPECS))
    if unknown:
        raise SystemExit(f"Unknown baselines: {unknown}. Available: {list(baseline_names())}")
    return names


def evaluate_baseline(
    name: str,
    train_examples: list[tuple[dict[str, float], str]],
    valid_examples: list[tuple[dict[str, float], str]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    spec = BASELINE_SPECS[name]
    started = time.perf_counter()
    transformed_train = transform_examples_for_baseline(name, train_examples)
    transformed_valid = transform_examples_for_baseline(name, valid_examples)
    model = build_baseline_policy(
        name,
        train_examples if spec.trains_on_dataset else None,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        class_weighting=args.class_weighting,
        max_class_weight=args.max_class_weight,
    )
    train_metrics = evaluate_policy(model, transformed_train) if spec.trains_on_dataset else {}
    valid_metrics = evaluate_policy(model, transformed_valid)
    valid_slice_metrics = evaluate_policy_slices(model, transformed_valid, min_examples=100)
    elapsed = time.perf_counter() - started
    return {
        "name": spec.name,
        "family": spec.family,
        "description": spec.description,
        "trains_on_dataset": spec.trains_on_dataset,
        "uses_private_cards": spec.uses_private_cards,
        "train_examples": len(transformed_train),
        "valid_examples": len(transformed_valid),
        "elapsed_seconds": elapsed,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "valid_slice_metrics": valid_slice_metrics,
    }


def main() -> None:
    args = parse_args()
    records = load_training_examples(
        args.dataset,
        max_examples=args.max_examples,
        require_hole_cards=args.missing_hole_cards == "drop",
        missing_hole_cards=args.missing_hole_cards,
        merge_all_in=True,
        include_hand_id=True,
    )
    if not records:
        raise SystemExit(f"No examples found in {args.dataset}")

    train_examples, valid_examples, split_info = stratified_group_holdout_split(
        records,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )
    results = [
        evaluate_baseline(name, train_examples, valid_examples, args)
        for name in parse_baselines(args.baselines)
    ]
    ranked = sorted(
        results,
        key=lambda row: (
            float(row["valid_metrics"].get("macro_f1", 0.0)),
            float(row["valid_metrics"].get("accuracy", 0.0)),
        ),
        reverse=True,
    )
    report = {
        "dataset": str(args.dataset),
        "split": split_info,
        "settings": {
            "baselines": parse_baselines(args.baselines),
            "missing_hole_cards": args.missing_hole_cards,
            "class_weighting": args.class_weighting,
            "max_class_weight": args.max_class_weight,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
        },
        "baselines": results,
        "best_baseline": ranked[0]["name"] if ranked else None,
        "ranking": [
            {
                "name": row["name"],
                "valid_macro_f1": row["valid_metrics"].get("macro_f1"),
                "valid_accuracy": row["valid_metrics"].get("accuracy"),
            }
            for row in ranked
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"report={args.out}")
    for row in ranked:
        metrics = row["valid_metrics"]
        print(
            f"{row['name']}: "
            f"accuracy={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"ce={metrics['cross_entropy']:.4f}"
        )


if __name__ == "__main__":
    main()
