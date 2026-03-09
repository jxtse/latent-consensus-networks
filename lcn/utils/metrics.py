"""Evaluation metrics used by LCN experiments."""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Mapping, Sequence


def accuracy(prediction: str, target: str) -> float:
    """Binary accuracy for one decision."""
    return float(prediction == target)


def majority_vote(decisions: Mapping[int, str] | Sequence[str]) -> str:
    """Return the most common decision label."""
    values = list(decisions.values()) if isinstance(decisions, Mapping) else list(decisions)
    if not values:
        raise ValueError("decisions must not be empty")
    return Counter(values).most_common(1)[0][0]


def conformity_rate(decisions: Mapping[int, str], pressure_option: str, *, target_agent_id: int = 0) -> float:
    """Return whether the target agent yielded to group pressure."""
    return float(decisions[target_agent_id] == pressure_option)


def mean_absolute_error(predictions: Iterable[float], targets: Iterable[float]) -> float:
    """Compute mean absolute error."""
    pairs = list(zip(predictions, targets))
    if not pairs:
        raise ValueError("predictions and targets must contain at least one item")
    return sum(abs(pred - target) for pred, target in pairs) / len(pairs)


def decision_entropy(decisions: Mapping[int, str] | Sequence[str]) -> float:
    """Entropy of categorical decisions, useful for diversity analysis."""
    values = list(decisions.values()) if isinstance(decisions, Mapping) else list(decisions)
    if not values:
        raise ValueError("decisions must not be empty")
    counts = Counter(values)
    total = len(values)
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log(probability + 1e-12, 2)
    return entropy
