from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.evaluator import evaluate_policy
from poker_agent.features import load_training_examples
from poker_agent.model import SklearnPolicy, SoftmaxPolicy
from poker_agent.slices import evaluate_policy_slices
from poker_agent.validation import random_action_split, stratified_group_holdout_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train supervised poker policy")
    parser.add_argument("--dataset", required=True, type=Path, help="Folder with hands/actions/players CSV files")
    parser.add_argument("--model-out", required=True, type=Path, help="Output JSON model path")
    parser.add_argument(
        "--policy",
        choices=(
            "hist_gradient_boosting",
            "xgboost",
            "lightgbm",
            "catboost",
            "extra_trees",
            "random_forest",
            "mlp",
            "softmax",
        ),
        default="hist_gradient_boosting",
        help="Model family to train. Boosting policies are the preferred research-grade tabular options.",
    )
    parser.add_argument("--epochs", type=int, default=12, help="Softmax-only epoch count")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=90, help="HistGradientBoosting iteration count")
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.02)
    parser.add_argument("--n-estimators", type=int, default=350, help="Tree count for forest policies")
    parser.add_argument(
        "--class-weighting",
        choices=("none", "sqrt_balanced", "balanced"),
        default="sqrt_balanced",
        help=(
            "Loss/sample weighting mode. Use sqrt_balanced/balanced for imbalance "
            "experiments, or none for pure accuracy optimization."
        ),
    )
    parser.add_argument("--max-class-weight", type=float, default=6.0)
    parser.add_argument(
        "--allow-missing-hole-cards",
        action="store_true",
        help="Keep rows where OCR did not capture two hole cards. Default filters them out.",
    )
    parser.add_argument(
        "--missing-hole-cards",
        choices=("drop", "flag", "keep"),
        default="drop",
        help=(
            "Missing-card policy. drop removes rows, flag keeps rows with missingness "
            "features, keep keeps rows without adding special handling beyond features."
        ),
    )
    parser.add_argument(
        "--keep-all-in-class",
        action="store_true",
        help="Keep all_in as a separate class. Default merges it into raise because it is too rare in OCR logs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument(
        "--split-strategy",
        choices=("stratified_hand_group", "random_action"),
        default="stratified_hand_group",
        help=(
            "Validation split strategy. stratified_hand_group is the research-safe "
            "default; random_action is only for smoke tests."
        ),
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Optional limit for quick test training. 0 means use all actions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_training_examples(
        args.dataset,
        max_examples=args.max_examples,
        require_hole_cards=not args.allow_missing_hole_cards,
        missing_hole_cards="flag" if args.allow_missing_hole_cards and args.missing_hole_cards == "drop" else args.missing_hole_cards,
        merge_all_in=not args.keep_all_in_class,
        include_hand_id=args.split_strategy == "stratified_hand_group",
    )
    if not records:
        raise SystemExit(f"No training examples found in {args.dataset}")
    if args.split_strategy == "stratified_hand_group":
        train_examples, valid_examples, split_info = stratified_group_holdout_split(
            records,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
        )
        examples_count = len(records)
    else:
        train_examples, valid_examples, split_info = random_action_split(
            records,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
        )
        examples_count = len(records)

    if args.policy == "softmax":
        model = SoftmaxPolicy()
        model.fit(
            train_examples,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            class_weighting=args.class_weighting,
            max_class_weight=args.max_class_weight,
        )
    else:
        model = SklearnPolicy()
        model.fit(
            train_examples,
            model_kind=args.policy,
            class_weighting=args.class_weighting,
            max_class_weight=args.max_class_weight,
            random_state=args.seed,
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            max_leaf_nodes=args.max_leaf_nodes,
            l2_regularization=args.l2_regularization,
            n_estimators=args.n_estimators,
        )
    train_metrics = evaluate_policy(model, train_examples)
    valid_metrics = evaluate_policy(model, valid_examples)
    valid_slice_metrics = evaluate_policy_slices(model, valid_examples, min_examples=100)
    model.metadata = {
        "dataset": str(args.dataset),
        "policy": args.policy,
        "split": split_info,
        "missing_hole_cards": args.missing_hole_cards,
        "merge_all_in": not args.keep_all_in_class,
        "class_weighting": args.class_weighting,
        "max_class_weight": args.max_class_weight,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "valid_slice_metrics": valid_slice_metrics,
    }
    model.save(args.model_out)

    print(f"saved_model={args.model_out}")
    print(f"policy={args.policy}")
    print(f"examples={examples_count} train_examples={len(train_examples)} valid_examples={len(valid_examples)}")
    print(f"split={json.dumps(split_info, sort_keys=True)}")
    print(f"class_weighting={args.class_weighting} class_weights={json.dumps(model.class_weights, sort_keys=True)}")
    print(f"train_class_counts={json.dumps(train_metrics['class_counts'], sort_keys=True)}")
    print(f"valid_class_counts={json.dumps(valid_metrics['class_counts'], sort_keys=True)}")
    print(f"train_accuracy={train_metrics['accuracy']:.4f} train_ce={train_metrics['cross_entropy']:.4f}")
    print(f"train_balanced_accuracy={train_metrics['balanced_accuracy']:.4f}")
    print(f"train_macro_f1={train_metrics['macro_f1']:.4f}")
    print(f"train_weighted_f1={train_metrics['weighted_f1']:.4f}")
    print(f"train_brier_loss={train_metrics['brier_loss']:.4f} train_ece_10={train_metrics['ece_10']:.4f}")
    print(
        "train_majority_baseline="
        f"{train_metrics['majority_baseline_accuracy']:.4f} "
        f"train_lift_vs_majority={train_metrics['lift_vs_majority']:.4f}"
    )
    print(f"valid_accuracy={valid_metrics['accuracy']:.4f} valid_ce={valid_metrics['cross_entropy']:.4f}")
    print(f"valid_balanced_accuracy={valid_metrics['balanced_accuracy']:.4f}")
    print(f"valid_macro_f1={valid_metrics['macro_f1']:.4f}")
    print(f"valid_weighted_f1={valid_metrics['weighted_f1']:.4f}")
    print(f"valid_brier_loss={valid_metrics['brier_loss']:.4f} valid_ece_10={valid_metrics['ece_10']:.4f}")
    print(
        "valid_majority_baseline="
        f"{valid_metrics['majority_baseline_accuracy']:.4f} "
        f"valid_lift_vs_majority={valid_metrics['lift_vs_majority']:.4f}"
    )
    print(f"valid_predicted_class_counts={json.dumps(valid_metrics['predicted_class_counts'], sort_keys=True)}")
    print(f"valid_per_class={json.dumps(valid_metrics['per_class'], sort_keys=True)}")
    print(f"valid_confusion_matrix={json.dumps(valid_metrics['confusion_matrix'], sort_keys=True)}")
    print(f"valid_slice_metrics={json.dumps(valid_slice_metrics, sort_keys=True)}")


if __name__ == "__main__":
    main()
