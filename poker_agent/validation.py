from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any

FeatureExample = tuple[dict[str, float], str]
GroupedFeatureExample = tuple[dict[str, float], str, str]


def random_action_split(
    examples: list[FeatureExample],
    *,
    valid_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[FeatureExample], list[FeatureExample], dict[str, Any]]:
    shuffled = examples[:]
    random.Random(seed).shuffle(shuffled)
    split = max(1, int(len(shuffled) * (1.0 - valid_ratio)))
    train_examples = shuffled[:split]
    valid_examples = shuffled[split:] or shuffled[:]
    return train_examples, valid_examples, {
        "split_type": "random_action",
        "valid_ratio": valid_ratio,
        "train_examples": len(train_examples),
        "valid_examples": len(valid_examples),
        "warning": (
            "Random action split is for smoke tests only. It can leak hand-level "
            "context between train and validation."
        ),
    }


def stratified_group_holdout_split(
    records: list[GroupedFeatureExample],
    *,
    valid_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[FeatureExample], list[FeatureExample], dict[str, Any]]:
    """Split examples by hand_id while roughly preserving label distribution.

    A pure random action split is optimistic for poker logs because actions from
    the same hand share cards, pot trajectory, positions, and OCR artifacts.
    This greedy splitter keeps every hand in exactly one side and chooses
    validation groups to approximate the global label distribution.
    """

    if not records:
        raise ValueError("No records to split")

    grouped: dict[str, list[FeatureExample]] = defaultdict(list)
    group_counts: dict[str, Counter[str]] = defaultdict(Counter)
    total_counts: Counter[str] = Counter()
    for features, label, group_id in records:
        group_key = str(group_id or "unknown")
        grouped[group_key].append((features, label))
        group_counts[group_key][label] += 1
        total_counts[label] += 1

    target_valid = max(1, int(len(records) * valid_ratio))
    target_counts = {label: count * valid_ratio for label, count in total_counts.items()}
    rng = random.Random(seed)
    group_ids = list(grouped)
    rng.shuffle(group_ids)
    group_ids.sort(key=lambda key: len(grouped[key]), reverse=True)

    valid_groups: set[str] = set()
    valid_counts: Counter[str] = Counter()
    valid_size = 0

    for group_id in group_ids:
        if valid_size >= target_valid:
            break
        candidate_counts = valid_counts + group_counts[group_id]
        current_error = _distribution_error(valid_counts, target_counts)
        candidate_error = _distribution_error(candidate_counts, target_counts)
        under_target = valid_size < target_valid
        improves_distribution = candidate_error <= current_error
        if under_target or improves_distribution:
            valid_groups.add(group_id)
            valid_counts = candidate_counts
            valid_size += len(grouped[group_id])

    if not valid_groups:
        valid_groups.add(group_ids[0])

    train_examples: list[FeatureExample] = []
    valid_examples: list[FeatureExample] = []
    for group_id, examples in grouped.items():
        if group_id in valid_groups:
            valid_examples.extend(examples)
        else:
            train_examples.extend(examples)

    if not train_examples:
        fallback = [(features, label) for features, label, _ in records]
        return random_action_split(fallback, valid_ratio=valid_ratio, seed=seed)

    split_info = {
        "split_type": "stratified_hand_group_holdout",
        "valid_ratio": valid_ratio,
        "train_examples": len(train_examples),
        "valid_examples": len(valid_examples),
        "train_groups": len(grouped) - len(valid_groups),
        "valid_groups": len(valid_groups),
        "train_class_counts": dict(sorted(Counter(label for _, label in train_examples).items())),
        "valid_class_counts": dict(sorted(Counter(label for _, label in valid_examples).items())),
    }
    return train_examples, valid_examples, split_info


def _distribution_error(counts: Counter[str], target_counts: dict[str, float]) -> float:
    error = 0.0
    for label, target in target_counts.items():
        error += abs(counts.get(label, 0) - target) / max(target, 1.0)
    return error
