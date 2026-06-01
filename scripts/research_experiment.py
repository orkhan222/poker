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

from poker_agent.evaluator import evaluate_policy
from poker_agent.features import load_training_examples
from poker_agent.model import SklearnPolicy, SoftmaxPolicy
from poker_agent.validation import stratified_group_holdout_split


DEFAULT_POLICIES = (
    "hist_gradient_boosting",
    "extra_trees",
    "random_forest",
    "mlp",
    "xgboost",
    "lightgbm",
    "catboost",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a leakage-aware poker action prediction experiment across "
            "multiple model families."
        )
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out-dir", default=Path("research_runs"), type=Path)
    parser.add_argument(
        "--policies",
        default=",".join(DEFAULT_POLICIES),
        help="Comma-separated model list. Optional libraries are skipped if unavailable.",
    )
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument(
        "--missing-hole-cards",
        choices=("drop", "flag", "keep"),
        default="drop",
        help="drop is the default research-safe option for current OCR data.",
    )
    parser.add_argument(
        "--class-weighting",
        choices=("none", "sqrt_balanced", "balanced"),
        default="sqrt_balanced",
    )
    parser.add_argument("--max-class-weight", type=float, default=6.0)
    parser.add_argument("--max-iter", type=int, default=90)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.02)
    parser.add_argument("--n-estimators", type=int, default=350)
    parser.add_argument(
        "--selection-metric",
        choices=("macro_f1", "weighted_f1", "accuracy"),
        default="macro_f1",
        help="Metric used to select the best model in the report.",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Save each fitted model artifact into --out-dir/models.",
    )
    return parser.parse_args()


def parse_policies(raw: str) -> list[str]:
    policies = [item.strip() for item in raw.split(",") if item.strip()]
    if not policies:
        raise ValueError("At least one policy must be provided")
    return policies


def build_model(policy_name: str) -> SoftmaxPolicy | SklearnPolicy:
    if policy_name == "softmax":
        return SoftmaxPolicy()
    return SklearnPolicy()


def fit_model(
    model: SoftmaxPolicy | SklearnPolicy,
    policy_name: str,
    train_examples: list[tuple[dict[str, float], str]],
    args: argparse.Namespace,
) -> None:
    if policy_name == "softmax":
        model.fit(
            train_examples,
            epochs=args.max_iter,
            learning_rate=args.learning_rate,
            class_weighting=args.class_weighting,
            max_class_weight=args.max_class_weight,
        )
        return

    model.fit(
        train_examples,
        model_kind=policy_name,
        class_weighting=args.class_weighting,
        max_class_weight=args.max_class_weight,
        random_state=args.seed,
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        l2_regularization=args.l2_regularization,
        n_estimators=args.n_estimators,
    )


def model_path_for(out_dir: Path, policy_name: str) -> Path:
    suffix = ".json" if policy_name == "softmax" else ".joblib"
    return out_dir / "models" / f"{policy_name}{suffix}"


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_models:
        (args.out_dir / "models").mkdir(parents=True, exist_ok=True)

    records = load_training_examples(
        args.dataset,
        max_examples=args.max_examples,
        missing_hole_cards=args.missing_hole_cards,
        require_hole_cards=args.missing_hole_cards == "drop",
        merge_all_in=True,
        include_hand_id=True,
    )
    if not records:
        raise SystemExit(f"No training examples found in {args.dataset}")

    train_examples, valid_examples, split_info = stratified_group_holdout_split(
        records,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )

    results: list[dict[str, Any]] = []
    for policy_name in parse_policies(args.policies):
        started = time.perf_counter()
        result: dict[str, Any] = {
            "policy": policy_name,
            "status": "started",
        }
        try:
            model = build_model(policy_name)
            fit_model(model, policy_name, train_examples, args)
            train_metrics = evaluate_policy(model, train_examples)
            valid_metrics = evaluate_policy(model, valid_examples)
            model.metadata = {
                "dataset": str(args.dataset),
                "policy": policy_name,
                "split": split_info,
                "settings": {
                    "missing_hole_cards": args.missing_hole_cards,
                    "class_weighting": args.class_weighting,
                    "max_class_weight": args.max_class_weight,
                },
                "train_metrics": train_metrics,
                "valid_metrics": valid_metrics,
            }
            if args.save_models:
                model.save(model_path_for(args.out_dir, policy_name))
            result.update(
                {
                    "status": "ok",
                    "seconds": round(time.perf_counter() - started, 3),
                    "class_weights": getattr(model, "class_weights", {}),
                    "train": train_metrics,
                    "valid": valid_metrics,
                }
            )
        except RuntimeError as exc:
            result.update(
                {
                    "status": "skipped",
                    "reason": str(exc),
                    "seconds": round(time.perf_counter() - started, 3),
                }
            )
        except Exception as exc:
            result.update(
                {
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "seconds": round(time.perf_counter() - started, 3),
                }
            )
        results.append(result)
        print(json.dumps(result, sort_keys=True))

    successful = [item for item in results if item.get("status") == "ok"]
    best = None
    if successful:
        best = max(successful, key=lambda item: float(item["valid"][args.selection_metric]))

    report = {
        "dataset": str(args.dataset),
        "settings": {
            "missing_hole_cards": args.missing_hole_cards,
            "class_weighting": args.class_weighting,
            "max_class_weight": args.max_class_weight,
            "selection_metric": args.selection_metric,
            "seed": args.seed,
        },
        "split": split_info,
        "best_policy": best["policy"] if best else None,
        "best_valid_metric": float(best["valid"][args.selection_metric]) if best else None,
        "results": results,
        "transformer_note": (
            "Transformer sequence modeling is scaffolded in poker_agent.sequence_models. "
            "It should be trained only after preserving full ordered action histories, "
            "not from flattened single-row features."
        ),
    }
    report_path = args.out_dir / "model_comparison.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"report={report_path}")
    if best:
        print(f"best_policy={best['policy']} best_{args.selection_metric}={best['valid'][args.selection_metric]:.4f}")


if __name__ == "__main__":
    main()
