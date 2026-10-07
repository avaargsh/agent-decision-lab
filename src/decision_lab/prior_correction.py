from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .logits import (
    CandidateLogprobBackend,
    backend_candidate_logprobs,
    normalized_candidate_scores,
)
from .models import CandidateScore, DecisionRequest


def neutral_candidate_prompt(request: DecisionRequest) -> str:
    """Context-free prompt used to estimate candidate continuation priors."""
    return (
        f"Decision type: {request.decision_type}\n"
        "Context:\n"
        "task_evidence: unavailable\n"
        "Best matching candidate: "
    )


@dataclass
class PriorCorrectedFrozenLogitAdapter:
    """Subtract context-free candidate priors before restricted softmax.

    This is a scalable PMI-style candidate-prior correction for continuation
    scoring. It is inspired by bias-correction work such as AnyJev, but it is
    not an implementation of AnyJev L0: there is no option-label rotation or
    next-token label readout here.
    """

    backend: CandidateLogprobBackend
    prompt_builder: callable
    prior_prompt_builder: callable = neutral_candidate_prompt
    prior_strength: float = 1.0
    name: str = "prior-corrected-frozen-logits"
    _prior_cache: dict[tuple[str, str], float] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.prior_strength < 0.0:
            raise ValueError("prior_strength must be >= 0")
        self.last_tokens_processed: int | None = None

    def score(
        self,
        request: DecisionRequest,
    ) -> Sequence[CandidateScore]:
        task_prompt = self.prompt_builder(request)
        task_logits, task_tokens = backend_candidate_logprobs(
            self.backend,
            prompt=task_prompt,
            candidates=request.candidates,
        )

        prior_prompt = self.prior_prompt_builder(request)
        missing = [
            candidate
            for candidate in request.candidates
            if (prior_prompt, candidate)
            not in self._prior_cache
        ]

        prior_tokens: int | None = 0
        if missing:
            prior_logits, prior_tokens = backend_candidate_logprobs(
                self.backend,
                prompt=prior_prompt,
                candidates=missing,
            )
            for candidate, value in zip(
                missing,
                prior_logits,
                strict=True,
            ):
                self._prior_cache[
                    (prior_prompt, candidate)
                ] = value

        priors = [
            self._prior_cache[
                (prior_prompt, candidate)
            ]
            for candidate in request.candidates
        ]
        corrected = [
            task - self.prior_strength * prior
            for task, prior in zip(
                task_logits,
                priors,
                strict=True,
            )
        ]

        self.last_tokens_processed = _sum_usage(
            task_tokens,
            prior_tokens,
        )

        return normalized_candidate_scores(
            request.candidates,
            corrected,
        )

    def clear_prior_cache(self) -> None:
        self._prior_cache.clear()


def _sum_usage(
    first: int | None,
    second: int | None,
) -> int | None:
    if first is None and second is None:
        return None
    return int(first or 0) + int(second or 0)
