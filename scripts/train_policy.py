from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.evaluator import evaluate_policy
from poker_agent.features import load_training_examples
from poker_agent.model import SoftmaxPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train supervised poker policy")
    parser.add_argument("--dataset", required=True, type=Path, help="Folder with hands/actions/players CSV files")
    parser.add_argument("--model-out", required=True, type=Path, help="Output JSON model path")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.08)
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
    examples = load_training_examples(args.dataset, max_examples=args.max_examples)
    if not examples:
        raise SystemExit(f"No training examples found in {args.dataset}")
    random.Random(args.seed).shuffle(examples)
    split = max(1, int(len(examples) * 0.85))
    train_examples = examples[:split]
    valid_examples = examples[split:] or examples[:]

    model = SoftmaxPolicy()
    model.fit(train_examples, epochs=args.epochs, learning_rate=args.learning_rate)
    model.save(args.model_out)

    train_metrics = evaluate_policy(model, train_examples)
    valid_metrics = evaluate_policy(model, valid_examples)
    print(f"saved_model={args.model_out}")
    print(f"train_accuracy={train_metrics['accuracy']:.4f} train_ce={train_metrics['cross_entropy']:.4f}")
    print(f"valid_accuracy={valid_metrics['accuracy']:.4f} valid_ce={valid_metrics['cross_entropy']:.4f}")


if __name__ == "__main__":
    main()
