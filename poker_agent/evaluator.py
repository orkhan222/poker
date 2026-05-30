from __future__ import annotations

import math
from collections import Counter
from typing import Any

from poker_agent.model import SoftmaxPolicy


def evaluate_policy(
    model: SoftmaxPolicy,
    examples: list[tuple[dict[str, float], str]],
) -> dict[str, Any]:
    if not examples:
        raise ValueError("No examples to evaluate")

    correct = 0
    total_loss = 0.0
    counts = Counter(label for _, label in examples)
    predicted_counts: Counter[str] = Counter()
    true_positive: Counter[str] = Counter()
    majority_count = max(counts.values()) if counts else 0
    for features, label in examples:
        prediction, probabilities = model.predict_from_features(features)
        predicted_counts[prediction] += 1
        if prediction == label:
            correct += 1
            true_positive[label] += 1
        total_loss += -math.log(max(probabilities.get(label, 0.0), 1e-12))

    accuracy = correct / len(examples)
    majority_baseline = majority_count / len(examples)
    per_class: dict[str, dict[str, float]] = {}
    f1_scores: list[float] = []
    for label in sorted(counts):
        precision = true_positive[label] / predicted_counts[label] if predicted_counts[label] else 0.0
        recall = true_positive[label] / counts[label] if counts[label] else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(counts[label]),
        }
        f1_scores.append(f1)
    return {
        "examples": float(len(examples)),
        "accuracy": accuracy,
        "cross_entropy": total_loss / len(examples),
        "macro_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "majority_baseline_accuracy": majority_baseline,
        "lift_vs_majority": accuracy - majority_baseline,
        "class_counts": dict(sorted(counts.items())),
        "predicted_class_counts": dict(sorted(predicted_counts.items())),
        "per_class": per_class,
    }
