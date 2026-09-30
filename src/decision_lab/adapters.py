from __future__ import annotations

import json
from dataclasses import dataclass
import math
from typing import Callable, Mapping, Sequence

from .models import CandidateScore, DecisionRequest


def structured_choice_scores(
    *,
    candidate: str,
    confidence: float,
    candidates: Sequence[str],
) -> list[CandidateScore]:
    """Convert an explicit structured choice into a probability distribution.

    Generated confidence is treated as self-reported probability only when it
    is above the uniform prior. Lower values mean nearly uniform, but the
    explicitly generated candidate must remain top-1 so evaluation does not
    silently replace the model choice with candidate-list order.
    """
    if not candidates:
        raise ValueError("candidates must not be empty")
    if candidate not in candidates:
        raise ValueError(f"generated unknown candidate: {candidate}")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("generated confidence must be between 0 and 1")

    if len(candidates) == 1:
        return [CandidateScore(candidate, 1.0)]

    uniform = 1.0 / len(candidates)
    selected_probability = max(confidence, uniform)
    if selected_probability == uniform:
        selected_probability = math.nextafter(uniform, 1.0)

    remaining_probability = 1.0 - selected_probability
    share = remaining_probability / (len(candidates) - 1)

    return [
        CandidateScore(
            item,
            selected_probability if item == candidate else share,
        )
        for item in candidates
    ]


@dataclass
class MappingScoreAdapter:
    """Simple adapter useful for fixtures and deterministic baselines."""

    mapping_fn: Callable[[DecisionRequest], Mapping[str, float]]
    name: str = "mapping-score"

    def score(self, request: DecisionRequest) -> Sequence[CandidateScore]:
        mapping = dict(self.mapping_fn(request))
        return [CandidateScore(candidate, mapping[candidate]) for candidate in request.candidates]


@dataclass
class StructuredOutputAdapter:
    """Wrap an autoregressive structured-output model behind the same candidate API.

    The callable must return JSON containing:
      {"candidate": "<one of candidates>", "confidence": 0.0-1.0}

    Non-selected candidates share the remaining probability mass equally. This is a
    comparison adapter, not a claim that generated confidence is calibrated.
    """

    generate_fn: Callable[[DecisionRequest], str]
    name: str = "structured-output"

    def score(self, request: DecisionRequest) -> Sequence[CandidateScore]:
        raw = self.generate_fn(request)
        obj = json.loads(raw)

        candidate = obj["candidate"]
        confidence = float(obj.get("confidence", 1.0))

        if candidate not in request.candidates:
            raise ValueError(f"generated unknown candidate: {candidate}")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("generated confidence must be between 0 and 1")

        return structured_choice_scores(
            candidate=candidate,
            confidence=confidence,
            candidates=request.candidates,
        )
