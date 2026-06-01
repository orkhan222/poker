from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def softmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    exp_scores = {key: math.exp(value - max_score) for key, value in scores.items()}
    total = sum(exp_scores.values()) or 1.0
    return {key: value / total for key, value in exp_scores.items()}


def balanced_class_weights(
    labels: list[str],
    label_order: list[str],
    mode: str,
    max_weight: float,
) -> dict[str, float]:
    if mode == "none":
        return {label: 1.0 for label in label_order}
    if mode not in {"balanced", "sqrt_balanced"}:
        raise ValueError(f"Unsupported class weighting mode: {mode}")

    counts = Counter(labels)
    total = sum(counts.values())
    class_count = len(label_order) or 1
    weights: dict[str, float] = {}
    for label in label_order:
        balanced_weight = total / (class_count * max(1, counts.get(label, 0)))
        if mode == "sqrt_balanced":
            balanced_weight = math.sqrt(balanced_weight)
        weights[label] = min(max_weight, balanced_weight)
    return weights


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
        self.class_weights = balanced_class_weights(
            [label for _, label in examples],
            mode=class_weighting,
            label_order=self.labels,
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
        return balanced_class_weights(labels, self.labels, mode, max_weight)

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


@dataclass
class SklearnPolicy:
    """Non-linear sklearn policy for production-style supervised baselines."""

    labels: list[str] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    estimator: Any = None
    model_kind: str = "hist_gradient_boosting"
    class_weights: dict[str, float] = field(default_factory=dict)
    encoded_labels: bool = False
    id_to_label: dict[int, str] = field(default_factory=dict)

    def fit(
        self,
        examples: list[tuple[dict[str, float], str]],
        model_kind: str = "hist_gradient_boosting",
        class_weighting: str = "sqrt_balanced",
        max_class_weight: float = 8.0,
        random_state: int = 42,
        max_iter: int = 220,
        learning_rate: float = 0.05,
        max_leaf_nodes: int = 31,
        l2_regularization: float = 0.01,
        n_estimators: int = 350,
    ) -> None:
        try:
            import numpy as np
            from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
            from sklearn.neural_network import MLPClassifier
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:
            raise RuntimeError(
                "The sklearn policy requires scikit-learn, numpy, and joblib. "
                "Install requirements.txt before training this model."
            ) from exc

        self.labels = sorted({label for _, label in examples})
        if not self.labels:
            raise ValueError("No labels found for training")

        self.feature_names = sorted({name for features, _ in examples for name in features})
        self.model_kind = model_kind
        y = [label for _, label in examples]
        y_fit: Any = y
        self.encoded_labels = False
        self.id_to_label = {}
        self.class_weights = balanced_class_weights(
            y,
            self.labels,
            mode=class_weighting,
            max_weight=max_class_weight,
        )
        sample_weight = np.array([self.class_weights.get(label, 1.0) for label in y], dtype=float)
        x_train = self._matrix([features for features, _ in examples], np=np)

        if model_kind == "hist_gradient_boosting":
            estimator = HistGradientBoostingClassifier(
                learning_rate=learning_rate,
                max_iter=max_iter,
                max_leaf_nodes=max_leaf_nodes,
                l2_regularization=l2_regularization,
                random_state=random_state,
            )
        elif model_kind == "extra_trees":
            estimator = ExtraTreesClassifier(
                n_estimators=n_estimators,
                min_samples_leaf=8,
                max_features="sqrt",
                n_jobs=-1,
                random_state=random_state,
            )
        elif model_kind == "random_forest":
            estimator = RandomForestClassifier(
                n_estimators=n_estimators,
                min_samples_leaf=8,
                max_features="sqrt",
                n_jobs=-1,
                random_state=random_state,
            )
        elif model_kind == "mlp":
            estimator = make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(256, 128, 64),
                    activation="relu",
                    alpha=l2_regularization,
                    batch_size=512,
                    learning_rate_init=learning_rate,
                    max_iter=max_iter,
                    early_stopping=True,
                    n_iter_no_change=10,
                    random_state=random_state,
                ),
            )
        elif model_kind == "xgboost":
            try:
                from xgboost import XGBClassifier
            except ImportError as exc:
                raise RuntimeError("Install xgboost to train --policy xgboost") from exc
            y_fit = self._encode_labels(y)
            estimator = XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                num_class=len(self.labels),
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=6,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=max(l2_regularization, 1e-6),
                tree_method="hist",
                random_state=random_state,
            )
        elif model_kind == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
            except ImportError as exc:
                raise RuntimeError("Install lightgbm to train --policy lightgbm") from exc
            y_fit = self._encode_labels(y)
            estimator = LGBMClassifier(
                objective="multiclass",
                num_class=len(self.labels),
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                num_leaves=max_leaf_nodes,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=max(l2_regularization, 1e-6),
                random_state=random_state,
                n_jobs=-1,
            )
        elif model_kind == "catboost":
            try:
                from catboost import CatBoostClassifier
            except ImportError as exc:
                raise RuntimeError("Install catboost to train --policy catboost") from exc
            y_fit = self._encode_labels(y)
            estimator = CatBoostClassifier(
                loss_function="MultiClass",
                iterations=n_estimators,
                learning_rate=learning_rate,
                depth=6,
                l2_leaf_reg=max(l2_regularization * 100.0, 1.0),
                random_seed=random_state,
                verbose=False,
            )
        else:
            raise ValueError(f"Unsupported sklearn model kind: {model_kind}")

        try:
            estimator.fit(x_train, y_fit, sample_weight=sample_weight)
        except TypeError:
            estimator.fit(x_train, y_fit)
        self.estimator = estimator

    def predict_proba_from_features(self, raw_features: dict[str, float]) -> dict[str, float]:
        if self.estimator is None:
            raise ValueError("SklearnPolicy estimator is not fitted")
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("numpy is required to run the sklearn policy") from exc

        probabilities = self.estimator.predict_proba(self._matrix([raw_features], np=np))[0]
        classes = self._class_labels()
        by_label = {str(label): float(probability) for label, probability in zip(classes, probabilities)}
        total = sum(by_label.values()) or 1.0
        return {label: by_label.get(label, 0.0) / total for label in self.labels}

    def predict_from_features(self, raw_features: dict[str, float]) -> tuple[str, dict[str, float]]:
        probabilities = self.predict_proba_from_features(raw_features)
        action = max(probabilities, key=probabilities.get)
        return action, probabilities

    def predict_batch_from_features(
        self,
        feature_rows: list[dict[str, float]],
    ) -> list[tuple[str, dict[str, float]]]:
        if self.estimator is None:
            raise ValueError("SklearnPolicy estimator is not fitted")
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("numpy is required to run the sklearn policy") from exc

        probability_rows = self.estimator.predict_proba(self._matrix(feature_rows, np=np))
        classes = self._class_labels()
        predictions: list[tuple[str, dict[str, float]]] = []
        for probabilities in probability_rows:
            by_label = {str(label): float(probability) for label, probability in zip(classes, probabilities)}
            total = sum(by_label.values()) or 1.0
            normalized = {label: by_label.get(label, 0.0) / total for label in self.labels}
            predictions.append((max(normalized, key=normalized.get), normalized))
        return predictions

    def save(self, path: Path) -> None:
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError("joblib is required to save the sklearn policy") from exc

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "policy_type": "sklearn_policy",
                "labels": self.labels,
                "feature_names": self.feature_names,
                "estimator": self.estimator,
                "model_kind": self.model_kind,
                "class_weights": self.class_weights,
                "encoded_labels": self.encoded_labels,
                "id_to_label": self.id_to_label,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "SklearnPolicy":
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError("joblib is required to load the sklearn policy") from exc

        payload = joblib.load(path)
        if payload.get("policy_type") != "sklearn_policy":
            raise ValueError(f"Unsupported sklearn policy payload: {path}")
        return cls(
            labels=list(payload["labels"]),
            feature_names=list(payload["feature_names"]),
            estimator=payload["estimator"],
            model_kind=str(payload.get("model_kind", "hist_gradient_boosting")),
            class_weights=dict(payload.get("class_weights", {})),
            encoded_labels=bool(payload.get("encoded_labels", False)),
            id_to_label={int(key): str(value) for key, value in dict(payload.get("id_to_label", {})).items()},
        )

    def _matrix(self, feature_rows: list[dict[str, float]], np: Any) -> Any:
        return np.array(
            [[float(row.get(name, 0.0)) for name in self.feature_names] for row in feature_rows],
            dtype=float,
        )

    def _encode_labels(self, labels: list[str]) -> list[int]:
        label_to_id = {label: index for index, label in enumerate(self.labels)}
        self.id_to_label = {index: label for label, index in label_to_id.items()}
        self.encoded_labels = True
        return [label_to_id[label] for label in labels]

    def _class_labels(self) -> list[str]:
        raw_classes = list(getattr(self.estimator, "classes_", self.labels))
        if not self.encoded_labels:
            return [str(label) for label in raw_classes]
        return [self.id_to_label.get(int(label), str(label)) for label in raw_classes]


def load_policy(path: Path) -> SoftmaxPolicy | SklearnPolicy:
    if path.suffix.lower() in {".joblib", ".pkl"}:
        return SklearnPolicy.load(path)
    return SoftmaxPolicy.load(path)
