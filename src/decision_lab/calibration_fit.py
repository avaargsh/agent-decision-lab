from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .calibration import TemperatureCalibrator
from .models import CandidateScore


@dataclass(frozen=True)
class CalibrationExample:
    scores: Sequence[CandidateScore]
    gold_candidate: str


@dataclass(frozen=True)
class TemperatureFit:
    temperature: float
    nll_before: float
    nll_after: float


def multiclass_nll(
    examples: Sequence[CalibrationExample],
    *,
    temperature: float = 1.0,
) -> float:
    if not examples:
        raise ValueError("examples must not be empty")

    calibrator = TemperatureCalibrator(temperature)
    total = 0.0

    for example in examples:
        calibrated = calibrator.calibrate(example.scores)
        by_candidate = {
            item.candidate: item.probability
            for item in calibrated
        }
        if example.gold_candidate not in by_candidate:
            raise ValueError(
                f"gold candidate missing from score set: {example.gold_candidate}"
            )
        probability = max(
            by_candidate[example.gold_candidate],
            1e-12,
        )
        total += -math.log(probability)

    return total / len(examples)


def fit_temperature(
    examples: Sequence[CalibrationExample],
    *,
    min_temperature: float = 0.05,
    max_temperature: float = 10.0,
    steps: int = 200,
) -> TemperatureFit:
    if min_temperature <= 0:
        raise ValueError("min_temperature must be > 0")
    if max_temperature <= min_temperature:
        raise ValueError(
            "max_temperature must be greater than min_temperature"
        )
    if steps < 2:
        raise ValueError("steps must be >= 2")

    # Log-spaced search is stable and dependency-free for the reference
    # implementation. More advanced optimizers can be plugged in later.
    log_min = math.log(min_temperature)
    log_max = math.log(max_temperature)

    candidates = [
        math.exp(
            log_min
            + (log_max - log_min) * index / (steps - 1)
        )
        for index in range(steps)
    ]

    best_temperature = 1.0
    best_nll = float("inf")

    for temperature in candidates:
        nll = multiclass_nll(
            examples,
            temperature=temperature,
        )
        if nll < best_nll:
            best_temperature = temperature
            best_nll = nll

    return TemperatureFit(
        temperature=best_temperature,
        nll_before=multiclass_nll(
            examples,
            temperature=1.0,
        ),
        nll_after=best_nll,
    )
