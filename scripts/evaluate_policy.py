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
from poker_agent.model import load_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate supervised poker policy")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Optional limit for quick evaluation. 0 means use all actions.",
    )
    parser.add_argument(
        "--allow-missing-hole-cards",
        action="store_true",
        help="Keep rows where OCR did not capture two hole cards. Default filters them out.",
    )
    parser.add_argument(
        "--missing-hole-cards",
        choices=("drop", "flag", "keep"),
        default="drop",
        help="Missing-card policy used by the feature loader.",
    )
    parser.add_argument(
        "--keep-all-in-class",
        action="store_true",
        help="Keep all_in as a separate class. Default merges it into raise because it is too rare in OCR logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise SystemExit(
            "Model file not found: "
            f"{args.model}\n\n"
            "First train it, for example:\n"
            f"python scripts\\train_policy.py --dataset \"{args.dataset}\" "
            f"--model-out \"{args.model}\""
        )
    model = load_policy(args.model)
    examples = load_training_examples(
        args.dataset,
        max_examples=args.max_examples,
        require_hole_cards=not args.allow_missing_hole_cards,
        missing_hole_cards="flag" if args.allow_missing_hole_cards and args.missing_hole_cards == "drop" else args.missing_hole_cards,
        merge_all_in=not args.keep_all_in_class,
    )
    if not examples:
        raise SystemExit(f"No evaluation examples found in {args.dataset}")
    metrics = evaluate_policy(model, examples)
    print(f"examples={int(metrics['examples'])}")
    print(f"accuracy={metrics['accuracy']:.4f}")
    print(f"cross_entropy={metrics['cross_entropy']:.4f}")
    print(f"macro_f1={metrics['macro_f1']:.4f}")
    print(f"weighted_f1={metrics['weighted_f1']:.4f}")
    print(f"majority_baseline_accuracy={metrics['majority_baseline_accuracy']:.4f}")
    print(f"lift_vs_majority={metrics['lift_vs_majority']:.4f}")
    print(f"class_counts={json.dumps(metrics['class_counts'], sort_keys=True)}")
    print(f"predicted_class_counts={json.dumps(metrics['predicted_class_counts'], sort_keys=True)}")
    print(f"per_class={json.dumps(metrics['per_class'], sort_keys=True)}")
    print(f"confusion_matrix={json.dumps(metrics['confusion_matrix'], sort_keys=True)}")


if __name__ == "__main__":
    main()
