from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThresholdPolicy:
    execute_threshold: float = 0.85
    margin_threshold: float = 0.20

    def should_execute(
        self,
        top_probability: float,
        runner_up_probability: float,
    ) -> tuple[bool, str]:
        if top_probability < self.execute_threshold:
            return False, "LOW_CONFIDENCE"

        if top_probability - runner_up_probability < self.margin_threshold:
            return False, "LOW_MARGIN"

        return True, "CONFIDENT"
