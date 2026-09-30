from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Prediction:
    confidence: float
    correct: bool


def binary_nll(predictions: Sequence[Prediction]) -> float:
    if not predictions:
        raise ValueError("predictions must not be empty")

    total = 0.0
    for item in predictions:
        p = min(max(item.confidence, 1e-12), 1.0 - 1e-12)
        y = 1.0 if item.correct else 0.0
        total += -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))
    return total / len(predictions)


def brier_score(predictions: Sequence[Prediction]) -> float:
    if not predictions:
        raise ValueError("predictions must not be empty")

    return sum(
        (item.confidence - (1.0 if item.correct else 0.0)) ** 2
        for item in predictions
    ) / len(predictions)


def expected_calibration_error(
    predictions: Sequence[Prediction],
    *,
    bins: int = 10,
) -> float:
    if not predictions:
        raise ValueError("predictions must not be empty")
    if bins <= 0:
        raise ValueError("bins must be > 0")

    total = len(predictions)
    ece = 0.0

    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins

        bucket = [
            item
            for item in predictions
            if lower <= item.confidence <= upper
            and (index == bins - 1 or item.confidence < upper)
        ]
        if not bucket:
            continue

        accuracy = sum(1 for item in bucket if item.correct) / len(bucket)
        mean_confidence = sum(item.confidence for item in bucket) / len(bucket)
        ece += (len(bucket) / total) * abs(accuracy - mean_confidence)

    return ece


def multiclass_nll(gold_probabilities: Sequence[float]) -> float:
    if not gold_probabilities:
        raise ValueError("gold_probabilities must not be empty")

    return sum(
        -math.log(min(max(probability, 1e-12), 1.0))
        for probability in gold_probabilities
    ) / len(gold_probabilities)


def macro_f1(
    gold: Sequence[str],
    predicted: Sequence[str],
) -> float:
    if not gold:
        raise ValueError("gold must not be empty")
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted must have equal length")

    labels = sorted(set(gold) | set(predicted))
    scores: list[float] = []
    for label in labels:
        true_positive = sum(
            1
            for actual, guess in zip(gold, predicted)
            if actual == label and guess == label
        )
        false_positive = sum(
            1
            for actual, guess in zip(gold, predicted)
            if actual != label and guess == label
        )
        false_negative = sum(
            1
            for actual, guess in zip(gold, predicted)
            if actual == label and guess != label
        )

        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = (
            true_positive / precision_denominator
            if precision_denominator
            else 0.0
        )
        recall = (
            true_positive / recall_denominator
            if recall_denominator
            else 0.0
        )
        scores.append(
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

    return sum(scores) / len(scores)
