from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from .models import CandidateScore, DecisionRequest


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

        remaining = [item for item in request.candidates if item != candidate]
        remainder = 1.0 - confidence
        share = remainder / len(remaining) if remaining else 0.0

        return [
            CandidateScore(
                item,
                confidence if item == candidate else share,
            )
            for item in request.candidates
        ]
