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


def backend_candidate_logprobs(
    backend: CandidateLogprobBackend,
    *,
    prompt: str,
    candidates: Sequence[str],
) -> tuple[list[float], int | None]:
    """Score explicit candidates and return model-token usage when exposed."""
    batch_fn = getattr(
        backend,
        "logprobs",
        None,
    )

    if callable(batch_fn):
        raw = list(
            batch_fn(
                prompt=prompt,
                candidates=candidates,
            )
        )
        usage = getattr(
            backend,
            "last_tokens_processed",
            None,
        )
        tokens = (
            int(usage)
            if usage is not None
            else None
        )
    else:
        raw = []
        total_tokens = 0
        has_usage = False

        for candidate in candidates:
            raw.append(
                backend.logprob(
                    prompt=prompt,
                    candidate=candidate,
                )
            )
            usage = getattr(
                backend,
                "last_tokens_processed",
                None,
            )
            if usage is not None:
                total_tokens += int(usage)
                has_usage = True

        tokens = (
            total_tokens
            if has_usage
            else None
        )

    if len(raw) != len(candidates):
        raise ValueError(
            "backend returned wrong number of candidate scores"
        )

    return raw, tokens


def normalized_candidate_scores(
    candidates: Sequence[str],
    logits: Sequence[float],
) -> list[CandidateScore]:
    if not candidates:
        raise ValueError("candidates must not be empty")
    if len(candidates) != len(logits):
        raise ValueError(
            "candidates and logits must have equal length"
        )

    max_value = max(logits)
    exp_values = [
        math.exp(value - max_value)
        for value in logits
    ]
    normalizer = sum(exp_values)

    return [
        CandidateScore(
            candidate,
            value / normalizer,
        )
        for candidate, value in zip(
            candidates,
            exp_values,
            strict=True,
        )
    ]


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
        raw, tokens = backend_candidate_logprobs(
            self.backend,
            prompt=prompt,
            candidates=request.candidates,
        )
        self.last_tokens_processed = tokens

        return normalized_candidate_scores(
            request.candidates,
            raw,
        )


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
