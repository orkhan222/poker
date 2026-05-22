from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.evaluator import evaluate_policy
from poker_agent.features import load_training_examples
from poker_agent.model import SoftmaxPolicy


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
    model = SoftmaxPolicy.load(args.model)
    examples = load_training_examples(args.dataset, max_examples=args.max_examples)
    if not examples:
        raise SystemExit(f"No evaluation examples found in {args.dataset}")
    metrics = evaluate_policy(model, examples)
    print(f"examples={int(metrics['examples'])}")
    print(f"accuracy={metrics['accuracy']:.4f}")
    print(f"cross_entropy={metrics['cross_entropy']:.4f}")


if __name__ == "__main__":
    main()
