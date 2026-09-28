from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class CandidateScore:
    candidate: str
    probability: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")


@dataclass(frozen=True)
class DecisionRequest:
    decision_type: str
    candidates: Sequence[str]
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.decision_type:
            raise ValueError("decision_type must not be empty")
        if not self.candidates:
            raise ValueError("candidates must not be empty")
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError("candidates must be unique")


@dataclass(frozen=True)
class Decision:
    candidate: str
    confidence: float
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionResult:
    request: DecisionRequest
    decision: Decision | None
    action: str
    reason_code: str
    scores: Sequence[CandidateScore]
