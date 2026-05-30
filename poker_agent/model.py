from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


def softmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    exp_scores = {key: math.exp(value - max_score) for key, value in scores.items()}
    total = sum(exp_scores.values()) or 1.0
    return {key: value / total for key, value in exp_scores.items()}


@dataclass
class SoftmaxPolicy:
    labels: list[str] = field(default_factory=list)
    weights: dict[str, dict[str, float]] = field(default_factory=dict)
    feature_means: dict[str, float] = field(default_factory=dict)
    feature_scales: dict[str, float] = field(default_factory=dict)
    class_weights: dict[str, float] = field(default_factory=dict)

    def fit(
        self,
        examples: list[tuple[dict[str, float], str]],
        epochs: int = 8,
        learning_rate: float = 0.08,
        l2: float = 0.0005,
        class_weighting: str = "balanced",
        max_class_weight: float = 12.0,
    ) -> None:
        self.labels = sorted({label for _, label in examples})
        if not self.labels:
            raise ValueError("No labels found for training")
        self._fit_scaler([features for features, _ in examples])
        self.weights = {label: {} for label in self.labels}
        self.class_weights = self._class_weights(
            [label for _, label in examples],
            mode=class_weighting,
            max_weight=max_class_weight,
        )

        for _ in range(epochs):
            for raw_features, label in examples:
                features = self._scale(raw_features)
                probabilities = self.predict_proba_from_features(raw_features)
                sample_weight = self.class_weights.get(label, 1.0)
                for candidate in self.labels:
                    error = sample_weight * (
                        (1.0 if candidate == label else 0.0) - probabilities.get(candidate, 0.0)
                    )
                    class_weights = self.weights[candidate]
                    for name, value in features.items():
                        old = class_weights.get(name, 0.0)
                        class_weights[name] = old + learning_rate * (error * value - l2 * old)

    def predict_proba_from_features(self, raw_features: dict[str, float]) -> dict[str, float]:
        features = self._scale(raw_features)
        scores: dict[str, float] = {}
        for label in self.labels:
            label_weights = self.weights.get(label, {})
            scores[label] = sum(label_weights.get(name, 0.0) * value for name, value in features.items())
        return softmax(scores)

    def predict_from_features(self, raw_features: dict[str, float]) -> tuple[str, dict[str, float]]:
        probabilities = self.predict_proba_from_features(raw_features)
        action = max(probabilities, key=probabilities.get)
        return action, probabilities

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "labels": self.labels,
                    "weights": self.weights,
                    "feature_means": self.feature_means,
                    "feature_scales": self.feature_scales,
                    "class_weights": self.class_weights,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "SoftmaxPolicy":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            labels=list(payload["labels"]),
            weights={label: dict(values) for label, values in payload["weights"].items()},
            feature_means=dict(payload.get("feature_means", {})),
            feature_scales=dict(payload.get("feature_scales", {})),
            class_weights=dict(payload.get("class_weights", {})),
        )

    def _class_weights(
        self,
        labels: list[str],
        mode: str,
        max_weight: float,
    ) -> dict[str, float]:
        if mode == "none":
            return {label: 1.0 for label in self.labels}
        if mode not in {"balanced", "sqrt_balanced"}:
            raise ValueError(f"Unsupported class weighting mode: {mode}")
        counts = Counter(labels)
        total = sum(counts.values())
        class_count = len(self.labels) or 1
        weights: dict[str, float] = {}
        for label in self.labels:
            balanced_weight = total / (class_count * max(1, counts.get(label, 0)))
            if mode == "sqrt_balanced":
                balanced_weight = math.sqrt(balanced_weight)
            weights[label] = min(max_weight, balanced_weight)
        return weights

    def _fit_scaler(self, feature_rows: list[dict[str, float]]) -> None:
        numeric_names = sorted({name for row in feature_rows for name in row})
        for name in numeric_names:
            values = [row.get(name, 0.0) for row in feature_rows]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            scale = math.sqrt(variance) or 1.0
            self.feature_means[name] = mean
            self.feature_scales[name] = scale

    def _scale(self, features: dict[str, float]) -> dict[str, float]:
        scaled: dict[str, float] = {}
        for name, value in features.items():
            if (
                name == "bias"
                or name.startswith("position_group=")
                or name.startswith("street=")
            ):
                scaled[name] = value
            else:
                scaled[name] = (value - self.feature_means.get(name, 0.0)) / self.feature_scales.get(name, 1.0)
        scaled.setdefault("bias", 1.0)
        return scaled
