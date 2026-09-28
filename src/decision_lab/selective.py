from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .metrics import Prediction


@dataclass(frozen=True)
class CoveragePoint:
    threshold: float
    coverage: float
    risk: float
    false_automation_rate: float


def risk_coverage(
    predictions: Sequence[Prediction],
    *,
    thresholds: Sequence[float],
) -> list[CoveragePoint]:
    if not predictions:
        raise ValueError("predictions must not be empty")

    points: list[CoveragePoint] = []
    total = len(predictions)

    for threshold in thresholds:
        automated = [p for p in predictions if p.confidence >= threshold]
        coverage = len(automated) / total

        if automated:
            errors = sum(1 for p in automated if not p.correct)
            risk = errors / len(automated)
            false_automation_rate = errors / total
        else:
            risk = 0.0
            false_automation_rate = 0.0

        points.append(
            CoveragePoint(
                threshold=threshold,
                coverage=coverage,
                risk=risk,
                false_automation_rate=false_automation_rate,
            )
        )

    return points
