from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.evaluator import evaluate_policy
from poker_agent.features import load_training_examples, public_context_features
from poker_agent.model import RoutedPolicyBundle, SklearnPolicy
from poker_agent.slices import evaluate_policy_slices
from poker_agent.validation import stratified_group_holdout_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train routed observed-card/context poker policy bundle")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model-out", required=True, type=Path)
    parser.add_argument("--observed-policy", default="hist_gradient_boosting")
    parser.add_argument("--context-policy", default="hist_gradient_boosting")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-weighting", choices=("none", "sqrt_balanced", "balanced"), default="sqrt_balanced")
    parser.add_argument("--max-class-weight", type=float, default=6.0)
    parser.add_argument("--max-iter", type=int, default=90)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.02)
    parser.add_argument("--n-estimators", type=int, default=350)
    return parser.parse_args()


def fit_sklearn_policy(
    examples: list[tuple[dict[str, float], str]],
    *,
    policy_name: str,
    args: argparse.Namespace,
) -> SklearnPolicy:
    model = SklearnPolicy()
    model.fit(
        examples,
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
    return model


def observed_only(examples: list[tuple[dict[str, float], str]]) -> list[tuple[dict[str, float], str]]:
    return [
        (features, label)
        for features, label in examples
        if features.get("hole_card_observed_ratio", 0.0) >= 1.0
    ]


def context_only(examples: list[tuple[dict[str, float], str]]) -> list[tuple[dict[str, float], str]]:
    return [(public_context_features(features), label) for features, label in examples]


def main() -> None:
    args = parse_args()
    records = load_training_examples(
        args.dataset,
        max_examples=args.max_examples,
        require_hole_cards=False,
        missing_hole_cards="flag",
        merge_all_in=True,
        include_hand_id=True,
    )
    if not records:
        raise SystemExit(f"No examples found in {args.dataset}")

    train_all, valid_all, split_info = stratified_group_holdout_split(
        records,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )
    train_observed = observed_only(train_all)
    valid_observed = observed_only(valid_all)
    train_context = context_only(train_all)
    valid_context = context_only(valid_all)
    if not train_observed:
        raise SystemExit("No observed-card training examples found")
    if not train_context:
        raise SystemExit("No context training examples found")

    observed_model = fit_sklearn_policy(train_observed, policy_name=args.observed_policy, args=args)
    context_model = fit_sklearn_policy(train_context, policy_name=args.context_policy, args=args)
    labels = sorted({label for _, label in train_all})
    bundle = RoutedPolicyBundle(
        observed_policy=observed_model,
        missing_policy=context_model,
        labels=labels,
    )

    train_metrics = evaluate_policy(bundle, train_all)
    valid_metrics = evaluate_policy(bundle, valid_all)
    valid_slice_metrics = evaluate_policy_slices(bundle, valid_all, min_examples=100)
    observed_valid_metrics = evaluate_policy(observed_model, valid_observed) if valid_observed else {}
    context_valid_metrics = evaluate_policy(context_model, valid_context) if valid_context else {}

    bundle.metadata = {
        "policy": "routed_policy_bundle",
        "observed_policy": args.observed_policy,
        "context_policy": args.context_policy,
        "dataset": str(args.dataset),
        "split": split_info,
        "missing_hole_cards": "routed_observed_plus_context",
        "class_weighting": args.class_weighting,
        "max_class_weight": args.max_class_weight,
        "train_examples": len(train_all),
        "valid_examples": len(valid_all),
        "observed_train_examples": len(train_observed),
        "context_train_examples": len(train_context),
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "valid_slice_metrics": valid_slice_metrics,
        "observed_model_valid_metrics": observed_valid_metrics,
        "context_model_valid_metrics": context_valid_metrics,
    }
    bundle.save(args.model_out)

    print(f"saved_model={args.model_out}")
    print("policy=routed_policy_bundle")
    print(f"examples={len(records)} train_examples={len(train_all)} valid_examples={len(valid_all)}")
    print(f"observed_train_examples={len(train_observed)} context_train_examples={len(train_context)}")
    print(f"split={json.dumps(split_info, sort_keys=True)}")
    print(f"valid_accuracy={valid_metrics['accuracy']:.4f} valid_ce={valid_metrics['cross_entropy']:.4f}")
    print(f"valid_balanced_accuracy={valid_metrics['balanced_accuracy']:.4f}")
    print(f"valid_macro_f1={valid_metrics['macro_f1']:.4f}")
    print(f"valid_weighted_f1={valid_metrics['weighted_f1']:.4f}")
    print(f"valid_majority_baseline={valid_metrics['majority_baseline_accuracy']:.4f} valid_lift_vs_majority={valid_metrics['lift_vs_majority']:.4f}")
    print(f"valid_predicted_class_counts={json.dumps(valid_metrics['predicted_class_counts'], sort_keys=True)}")
    print(f"valid_per_class={json.dumps(valid_metrics['per_class'], sort_keys=True)}")
    print(f"valid_slice_metrics={json.dumps(valid_slice_metrics, sort_keys=True)}")


if __name__ == "__main__":
    main()
