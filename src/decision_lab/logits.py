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

    def __post_init__(self) -> None:
        self.last_tokens_processed: int | None = None

    def score(self, request: DecisionRequest) -> Sequence[CandidateScore]:
        prompt = self.prompt_builder(request)

        batch_fn = getattr(
            self.backend,
            "logprobs",
            None,
        )

        if callable(batch_fn):
            raw = list(
                batch_fn(
                    prompt=prompt,
                    candidates=request.candidates,
                )
            )
            usage = getattr(
                self.backend,
                "last_tokens_processed",
                None,
            )
            self.last_tokens_processed = (
                int(usage)
                if usage is not None
                else None
            )
        else:
            raw = []
            total_tokens = 0
            has_usage = False

            for candidate in request.candidates:
                raw.append(
                    self.backend.logprob(
                        prompt=prompt,
                        candidate=candidate,
                    )
                )

                usage = getattr(
                    self.backend,
                    "last_tokens_processed",
                    None,
                )
                if usage is not None:
                    total_tokens += int(usage)
                    has_usage = True

            self.last_tokens_processed = (
                total_tokens
                if has_usage
                else None
            )

        if len(raw) != len(
            request.candidates
        ):
            raise ValueError(
                "backend returned wrong number "
                "of candidate scores"
            )

        max_value = max(raw)
        exp_values = [
            math.exp(value - max_value)
            for value in raw
        ]
        normalizer = sum(exp_values)

        return [
            CandidateScore(
                candidate,
                value / normalizer,
            )
            for candidate, value in zip(
                request.candidates,
                exp_values,
                strict=True,
            )
        ]


def default_candidate_prompt(request: DecisionRequest) -> str:
    context_lines = [
        f"{key}: {value}"
        for key, value in sorted(
            request.context.items()
        )
    ]
    candidates = ", ".join(
        request.candidates
    )

    return (
        f"Decision type: {request.decision_type}\n"
        f"Candidates: {candidates}\n"
        + (
            "Context:\n"
            + "\n".join(context_lines)
            + "\n"
            if context_lines
            else ""
        )
        + "Choose exactly one candidate:\n"
    )


def compact_candidate_prompt(request: DecisionRequest) -> str:
    """Prompt for scalable candidate continuation scoring.

    The bounded candidate set is enforced by the continuations being scored, so
    the full list does not need to be duplicated inside every candidate sequence.
    This keeps prompt tokens roughly constant as candidate count grows.
    """
    context_lines = [
        f"{key}: {value}"
        for key, value in sorted(
            request.context.items()
        )
    ]

    return (
        f"Decision type: {request.decision_type}\n"
        + (
            "Context:\n"
            + "\n".join(context_lines)
            + "\n"
            if context_lines
            else ""
        )
        + "Best matching candidate: "
    )
