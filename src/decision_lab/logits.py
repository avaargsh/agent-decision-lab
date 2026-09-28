from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

from .models import CandidateScore, DecisionRequest


class CandidateLogprobBackend(Protocol):
    """Backend that scores candidate continuations without free-form generation."""

    name: str

    def logprob(
        self,
        *,
        prompt: str,
        candidate: str,
    ) -> float:
        ...


@dataclass
class FrozenLogitAdapter:
    """Normalize candidate continuation log-probabilities into a bounded distribution."""

    backend: CandidateLogprobBackend
    prompt_builder: callable
    name: str = "frozen-logits"

    def score(self, request: DecisionRequest) -> Sequence[CandidateScore]:
        prompt = self.prompt_builder(request)

        raw = [
            self.backend.logprob(prompt=prompt, candidate=candidate)
            for candidate in request.candidates
        ]

        max_value = max(raw)
        exp_values = [math.exp(value - max_value) for value in raw]
        normalizer = sum(exp_values)

        return [
            CandidateScore(candidate, value / normalizer)
            for candidate, value in zip(
                request.candidates,
                exp_values,
                strict=True,
            )
        ]


def default_candidate_prompt(request: DecisionRequest) -> str:
    context_lines = [
        f"{key}: {value}"
        for key, value in sorted(request.context.items())
    ]
    candidates = ", ".join(request.candidates)

    return (
        f"Decision type: {request.decision_type}\n"
        f"Candidates: {candidates}\n"
        + ("Context:\n" + "\n".join(context_lines) + "\n" if context_lines else "")
        + "Choose exactly one candidate:\n"
    )
