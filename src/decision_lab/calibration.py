from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

from .models import CandidateScore


class Calibrator(Protocol):
    def calibrate(self, scores: Sequence[CandidateScore]) -> list[CandidateScore]:
        ...


@dataclass(frozen=True)
class IdentityCalibrator:
    def calibrate(self, scores: Sequence[CandidateScore]) -> list[CandidateScore]:
        return list(scores)


@dataclass(frozen=True)
class TemperatureCalibrator:
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")

    def calibrate(self, scores: Sequence[CandidateScore]) -> list[CandidateScore]:
        if not scores:
            return []

        # Convert probabilities to stabilized log-probabilities, then rescale.
        logits = [math.log(max(item.probability, 1e-12)) for item in scores]
        scaled = [logit / self.temperature for logit in logits]
        max_logit = max(scaled)
        exps = [math.exp(value - max_logit) for value in scaled]
        normalizer = sum(exps)

        return [
            CandidateScore(item.candidate, value / normalizer)
            for item, value in zip(scores, exps, strict=True)
        ]
