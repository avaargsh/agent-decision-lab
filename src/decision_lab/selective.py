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
    fallback_rate: float


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
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("thresholds must be between 0 and 1")

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
                fallback_rate=1.0 - coverage,
            )
        )

    return points


def select_operating_point(
    points: Sequence[CoveragePoint],
    *,
    max_risk: float,
    min_coverage: float = 0.0,
) -> CoveragePoint | None:
    """Choose maximum automation coverage that stays inside a risk budget.

    Empty-automation points are intentionally excluded: zero coverage trivially has
    zero observed risk but is not a useful operating point.
    """
    if not 0.0 <= max_risk <= 1.0:
        raise ValueError("max_risk must be between 0 and 1")
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError("min_coverage must be between 0 and 1")

    eligible = [
        point
        for point in points
        if point.coverage > 0.0
        and point.coverage >= min_coverage
        and point.risk <= max_risk
    ]
    if not eligible:
        return None

    return max(
        eligible,
        key=lambda point: (
            point.coverage,
            -point.risk,
            point.threshold,
        ),
    )
