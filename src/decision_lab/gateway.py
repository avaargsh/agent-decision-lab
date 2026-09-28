from __future__ import annotations

from collections.abc import Callable, Sequence

from .models import CandidateScore, Decision, DecisionRequest, DecisionResult
from .policy import ThresholdPolicy

Scorer = Callable[[DecisionRequest], Sequence[CandidateScore]]


class DecisionGateway:
    def __init__(
        self,
        scorer: Scorer,
        policy: ThresholdPolicy | None = None,
    ) -> None:
        self._scorer = scorer
        self._policy = policy or ThresholdPolicy()

    def decide(self, request: DecisionRequest) -> DecisionResult:
        scores = list(self._scorer(request))
        self._validate_scores(request, scores)

        ranked = sorted(scores, key=lambda item: item.probability, reverse=True)
        top = ranked[0]
        runner_up = ranked[1].probability if len(ranked) > 1 else 0.0

        should_execute, reason_code = self._policy.should_execute(
            top_probability=top.probability,
            runner_up_probability=runner_up,
        )

        if not should_execute:
            return DecisionResult(
                request=request,
                decision=None,
                action="FALLBACK",
                reason_code=reason_code,
                scores=ranked,
            )

        return DecisionResult(
            request=request,
            decision=Decision(
                candidate=top.candidate,
                confidence=top.probability,
                evidence={
                    "runner_up_probability": runner_up,
                    "margin": top.probability - runner_up,
                },
            ),
            action="EXECUTE",
            reason_code=reason_code,
            scores=ranked,
        )

    @staticmethod
    def _validate_scores(
        request: DecisionRequest,
        scores: Sequence[CandidateScore],
    ) -> None:
        expected = set(request.candidates)
        actual = {item.candidate for item in scores}

        if expected != actual:
            raise ValueError(
                "scorer must return exactly one score for each request candidate"
            )

        total = sum(item.probability for item in scores)
        if abs(total - 1.0) > 1e-6:
            raise ValueError("candidate probabilities must sum to 1.0")
