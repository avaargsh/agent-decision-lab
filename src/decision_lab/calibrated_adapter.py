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

    def score(
        self,
        request: DecisionRequest,
    ) -> Sequence[CandidateScore]:
        return self.calibrator.calibrate(
            self.base.score(request)
        )
