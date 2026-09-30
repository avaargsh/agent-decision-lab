from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Protocol, Sequence

from .benchmark import BenchmarkCase
from .models import CandidateScore, DecisionRequest
from .runner import BenchmarkReport


class DecisionAdapter(Protocol):
    name: str

    def score(self, request: DecisionRequest) -> Sequence[CandidateScore]:
        ...


@dataclass(frozen=True)
class FallbackCaseResult:
    case_id: str
    fast_confidence: float
    fast_predicted: str
    fallback_predicted: str
    gold: str
    correct: bool
    latency_ms: float
    tokens_processed: int | None


@dataclass(frozen=True)
class FallbackEvaluation:
    measured: bool
    adapter: str
    threshold: float
    eligible_case_count: int
    fallback_case_count: int
    fallback_rate: float
    accuracy: float | None
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    mean_tokens_processed: float | None
    cases: tuple[FallbackCaseResult, ...]

    def artifact_payload(self) -> dict:
        payload = asdict(self)
        payload["cases"] = [asdict(case) for case in self.cases]
        return payload


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def evaluate_system2_fallback(
    fast_report: BenchmarkReport,
    cases: Sequence[BenchmarkCase],
    fallback: DecisionAdapter,
    *,
    threshold: float,
) -> FallbackEvaluation:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    case_by_id = {case.case_id: case for case in cases}
    if len(case_by_id) != len(cases):
        raise ValueError("case ids must be unique")

    results: list[FallbackCaseResult] = []
    for fast in fast_report.cases:
        if fast.confidence >= threshold:
            continue
        case = case_by_id.get(fast.case_id)
        if case is None:
            raise ValueError(f"missing benchmark case: {fast.case_id}")

        request = DecisionRequest(
            decision_type=case.decision_type,
            candidates=case.candidates,
            context=case.context,
        )
        started = time.perf_counter()
        scores = list(fallback.score(request))
        measured_latency_ms = (time.perf_counter() - started) * 1000.0
        latency = getattr(fallback, "last_latency_ms", None)
        latency_ms = float(latency) if latency is not None else measured_latency_ms

        expected = set(case.candidates)
        actual = {score.candidate for score in scores}
        if actual != expected or len(scores) != len(case.candidates):
            raise ValueError(f"fallback returned invalid candidates for {case.case_id}")

        predicted = max(scores, key=lambda score: score.probability).candidate
        tokens = getattr(fallback, "last_tokens_processed", None)
        results.append(
            FallbackCaseResult(
                case_id=case.case_id,
                fast_confidence=fast.confidence,
                fast_predicted=fast.predicted,
                fallback_predicted=predicted,
                gold=case.gold_candidate,
                correct=predicted == case.gold_candidate,
                latency_ms=latency_ms,
                tokens_processed=int(tokens) if tokens is not None else None,
            )
        )

    total = len(fast_report.cases)
    if not results:
        return FallbackEvaluation(
            measured=True,
            adapter=fallback.name,
            threshold=threshold,
            eligible_case_count=total,
            fallback_case_count=0,
            fallback_rate=0.0,
            accuracy=None,
            p50_latency_ms=None,
            p95_latency_ms=None,
            mean_tokens_processed=None,
            cases=(),
        )

    latencies = [result.latency_ms for result in results]
    token_counts = [
        result.tokens_processed
        for result in results
        if result.tokens_processed is not None
    ]
    return FallbackEvaluation(
        measured=True,
        adapter=fallback.name,
        threshold=threshold,
        eligible_case_count=total,
        fallback_case_count=len(results),
        fallback_rate=len(results) / total,
        accuracy=sum(result.correct for result in results) / len(results),
        p50_latency_ms=_percentile(latencies, 0.5),
        p95_latency_ms=_percentile(latencies, 0.95),
        mean_tokens_processed=(
            sum(token_counts) / len(token_counts)
            if token_counts
            else None
        ),
        cases=tuple(results),
    )
