from __future__ import annotations

import math
from collections import Counter
from typing import Any

def evaluate_policy(
    model: Any,
    examples: list[tuple[dict[str, float], str]],
) -> dict[str, Any]:
    if not examples:
        raise ValueError("No examples to evaluate")

    correct = 0
    total_loss = 0.0
    brier_loss = 0.0
    counts = Counter(label for _, label in examples)
    predicted_counts: Counter[str] = Counter()
    true_positive: Counter[str] = Counter()
    majority_count = max(counts.values()) if counts else 0
    feature_rows = [features for features, _ in examples]
    if hasattr(model, "predict_batch_from_features"):
        model_predictions = model.predict_batch_from_features(feature_rows)
    else:
        model_predictions = [model.predict_from_features(features) for features in feature_rows]

    for (_, label), (prediction, probabilities) in zip(examples, model_predictions):
        predicted_counts[prediction] += 1
        if prediction == label:
            correct += 1
            true_positive[label] += 1
        total_loss += -math.log(max(probabilities.get(label, 0.0), 1e-12))
        for candidate in set(counts) | set(probabilities):
            target = 1.0 if candidate == label else 0.0
            brier_loss += (float(probabilities.get(candidate, 0.0)) - target) ** 2

    accuracy = correct / len(examples)
    majority_baseline = majority_count / len(examples)
    per_class: dict[str, dict[str, float]] = {}
    f1_scores: list[float] = []
    labels = sorted(set(counts) | set(predicted_counts))
    weighted_f1_total = 0.0
    recall_total = 0.0
    for label in labels:
        precision = true_positive[label] / predicted_counts[label] if predicted_counts[label] else 0.0
        recall = true_positive[label] / counts[label] if counts[label] else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = float(counts[label])
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1_scores.append(f1)
        weighted_f1_total += f1 * support
        recall_total += recall

    label_to_index = {label: index for index, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for (_, label), (prediction, _) in zip(examples, model_predictions):
        matrix[label_to_index[label]][label_to_index[prediction]] += 1

    return {
        "examples": float(len(examples)),
        "accuracy": accuracy,
        "cross_entropy": total_loss / len(examples),
        "brier_loss": brier_loss / len(examples),
        "macro_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "weighted_f1": weighted_f1_total / len(examples),
        "balanced_accuracy": recall_total / len(labels) if labels else 0.0,
        "ece_10": expected_calibration_error(model_predictions, examples, bins=10),
        "majority_baseline_accuracy": majority_baseline,
        "lift_vs_majority": accuracy - majority_baseline,
        "labels": labels,
        "class_counts": dict(sorted(counts.items())),
        "predicted_class_counts": dict(sorted(predicted_counts.items())),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": labels,
            "matrix": matrix,
        },
    }


def expected_calibration_error(
    model_predictions: list[tuple[str, dict[str, float]]],
    examples: list[tuple[dict[str, float], str]],
    *,
    bins: int = 10,
) -> float:
    bucket_total = [0 for _ in range(bins)]
    bucket_confidence = [0.0 for _ in range(bins)]
    bucket_accuracy = [0.0 for _ in range(bins)]

    for (_, label), (prediction, probabilities) in zip(examples, model_predictions):
        confidence = max(probabilities.values()) if probabilities else 0.0
        bucket = min(bins - 1, int(confidence * bins))
        bucket_total[bucket] += 1
        bucket_confidence[bucket] += confidence
        bucket_accuracy[bucket] += 1.0 if prediction == label else 0.0

    total = len(examples)
    ece = 0.0
    for index, count in enumerate(bucket_total):
        if not count:
            continue
        avg_confidence = bucket_confidence[index] / count
        avg_accuracy = bucket_accuracy[index] / count
        ece += (count / total) * abs(avg_confidence - avg_accuracy)
    return ece
