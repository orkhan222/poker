from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.evaluator import evaluate_policy
from poker_agent.features import load_training_examples
from poker_agent.model import SklearnPolicy, SoftmaxPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train supervised poker policy")
    parser.add_argument("--dataset", required=True, type=Path, help="Folder with hands/actions/players CSV files")
    parser.add_argument("--model-out", required=True, type=Path, help="Output JSON model path")
    parser.add_argument(
        "--policy",
        choices=("hist_gradient_boosting", "extra_trees", "random_forest", "softmax"),
        default="hist_gradient_boosting",
        help="Model family to train. Tree/boosting policies are the professional non-linear options.",
    )
    parser.add_argument("--epochs", type=int, default=12, help="Softmax-only epoch count")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=220, help="HistGradientBoosting iteration count")
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.01)
    parser.add_argument("--n-estimators", type=int, default=350, help="Tree count for forest policies")
    parser.add_argument(
        "--class-weighting",
        choices=("none", "sqrt_balanced", "balanced"),
        default="none",
        help=(
            "Loss/sample weighting mode. Use sqrt_balanced/balanced for imbalance "
            "experiments, or none for pure accuracy optimization."
        ),
    )
    parser.add_argument("--max-class-weight", type=float, default=12.0)
    parser.add_argument(
        "--allow-missing-hole-cards",
        action="store_true",
        help="Keep rows where OCR did not capture two hole cards. Default filters them out.",
    )
    parser.add_argument(
        "--keep-all-in-class",
        action="store_true",
        help="Keep all_in as a separate class. Default merges it into raise because it is too rare in OCR logs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Optional limit for quick test training. 0 means use all actions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_training_examples(
        args.dataset,
        max_examples=args.max_examples,
        require_hole_cards=not args.allow_missing_hole_cards,
        merge_all_in=not args.keep_all_in_class,
    )
    if not examples:
        raise SystemExit(f"No training examples found in {args.dataset}")
    random.Random(args.seed).shuffle(examples)
    split = max(1, int(len(examples) * 0.85))
    train_examples = examples[:split]
    valid_examples = examples[split:] or examples[:]

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
    model.save(args.model_out)

    train_metrics = evaluate_policy(model, train_examples)
    valid_metrics = evaluate_policy(model, valid_examples)
    print(f"saved_model={args.model_out}")
    print(f"policy={args.policy}")
    print(f"examples={len(examples)} train_examples={len(train_examples)} valid_examples={len(valid_examples)}")
    print(f"class_weighting={args.class_weighting} class_weights={json.dumps(model.class_weights, sort_keys=True)}")
    print(f"train_class_counts={json.dumps(train_metrics['class_counts'], sort_keys=True)}")
    print(f"valid_class_counts={json.dumps(valid_metrics['class_counts'], sort_keys=True)}")
    print(f"train_accuracy={train_metrics['accuracy']:.4f} train_ce={train_metrics['cross_entropy']:.4f}")
    print(f"train_macro_f1={train_metrics['macro_f1']:.4f}")
    print(
        "train_majority_baseline="
        f"{train_metrics['majority_baseline_accuracy']:.4f} "
        f"train_lift_vs_majority={train_metrics['lift_vs_majority']:.4f}"
    )
    print(f"valid_accuracy={valid_metrics['accuracy']:.4f} valid_ce={valid_metrics['cross_entropy']:.4f}")
    print(f"valid_macro_f1={valid_metrics['macro_f1']:.4f}")
    print(
        "valid_majority_baseline="
        f"{valid_metrics['majority_baseline_accuracy']:.4f} "
        f"valid_lift_vs_majority={valid_metrics['lift_vs_majority']:.4f}"
    )
    print(f"valid_predicted_class_counts={json.dumps(valid_metrics['predicted_class_counts'], sort_keys=True)}")
    print(f"valid_per_class={json.dumps(valid_metrics['per_class'], sort_keys=True)}")


if __name__ == "__main__":
    main()
