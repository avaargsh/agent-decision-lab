from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .calibration import Calibrator
from .models import CandidateScore, DecisionRequest
from .runner import DecisionAdapter


@dataclass
class CalibratedAdapter:
    base: DecisionAdapter
    calibrator: Calibrator

    @property
    def name(self) -> str:
        return f"{self.base.name}+calibrated"

    @property
    def last_tokens_processed(self) -> int | None:
        value = getattr(
            self.base,
            "last_tokens_processed",
            None,
        )
        return (
            int(value)
            if value is not None
            else None
        )

    @property
    def last_latency_ms(self) -> float | None:
        value = getattr(
            self.base,
            "last_latency_ms",
            None,
        )
        return (
            float(value)
            if value is not None
            else None
        )

    def score(
        self,
        request: DecisionRequest,
    ) -> Sequence[CandidateScore]:
        return self.calibrator.calibrate(
            self.base.score(request)
        )
