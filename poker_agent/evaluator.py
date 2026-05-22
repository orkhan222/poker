from __future__ import annotations

import math

from poker_agent.model import SoftmaxPolicy


def evaluate_policy(
    model: SoftmaxPolicy,
    examples: list[tuple[dict[str, float], str]],
) -> dict[str, float]:
    if not examples:
        raise ValueError("No examples to evaluate")

    correct = 0
    total_loss = 0.0
    for features, label in examples:
        prediction, probabilities = model.predict_from_features(features)
        if prediction == label:
            correct += 1
        total_loss += -math.log(max(probabilities.get(label, 0.0), 1e-12))

    return {
        "examples": float(len(examples)),
        "accuracy": correct / len(examples),
        "cross_entropy": total_loss / len(examples),
    }

