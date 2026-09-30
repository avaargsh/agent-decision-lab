from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Protocol, Sequence

from .benchmark import BenchmarkCase
from .metrics import (
    Prediction,
    brier_score,
    expected_calibration_error,
    macro_f1,
    multiclass_nll,
)
from .models import CandidateScore, DecisionRequest
from .selective import CoveragePoint, risk_coverage
from .telemetry import set_attribute, span


class DecisionAdapter(Protocol):
    name: str

    def score(self, request: DecisionRequest) -> Sequence[CandidateScore]:
        ...


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    predicted: str
    gold: str
    confidence: float
    correct: bool
    latency_ms: float
    tokens_processed: int | None = None
    gold_probability: float | None = None


@dataclass(frozen=True)
class BenchmarkReport:
    adapter: str
    accuracy: float
    brier: float
    ece: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    mean_tokens_processed_per_decision: float | None
    coverage: list[CoveragePoint]
    cases: list[CaseResult]
    macro_f1: float = 0.0
    nll: float = 0.0


def _percentile(
    values: Sequence[float],
    quantile: float,
) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(
            "quantile must be between 0 and 1"
        )

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    weight = position - lower
    return (
        ordered[lower] * (1.0 - weight)
        + ordered[upper] * weight
    )


def _validate_scores(
    *,
    adapter_name: str,
    case_id: str,
    candidates: Sequence[str],
    scores: Sequence[CandidateScore],
) -> None:
    expected = set(candidates)
    actual = {score.candidate for score in scores}

    if expected != actual or len(scores) != len(candidates):
        raise ValueError(
            f"{adapter_name} returned an invalid candidate set for {case_id}"
        )

    total = sum(score.probability for score in scores)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"{adapter_name} probabilities must sum to 1.0 for {case_id}; got {total}"
        )


def run_benchmark(
    adapter: DecisionAdapter,
    cases: Sequence[BenchmarkCase],
    *,
    thresholds: Sequence[float] = (0.5, 0.7, 0.8, 0.9, 0.95),
) -> BenchmarkReport:
    if not cases:
        raise ValueError("cases must not be empty")

    results: list[CaseResult] = []
    predictions: list[Prediction] = []

    with span(
        "decision.benchmark",
        {
            "benchmark.adapter": adapter.name,
            "benchmark.case_count": len(cases),
        },
    ) as benchmark_span:
        for case in cases:
            request = DecisionRequest(
                decision_type=case.decision_type,
                candidates=case.candidates,
                context=case.context,
            )

            with span(
                "decision.benchmark.case",
                {
                    "benchmark.adapter": adapter.name,
                    "benchmark.case_id": case.case_id,
                    "decision.type": case.decision_type,
                    "decision.candidate_count": len(case.candidates),
                },
            ) as case_span:
                started = time.perf_counter()
                scores = list(adapter.score(request))
                measured_latency_ms = (
                    time.perf_counter() - started
                ) * 1000.0

                adapter_latency_ms = getattr(
                    adapter,
                    "last_latency_ms",
                    None,
                )
                latency_ms = (
                    float(adapter_latency_ms)
                    if adapter_latency_ms is not None
                    else measured_latency_ms
                )

                tokens_processed = getattr(
                    adapter,
                    "last_tokens_processed",
                    None,
                )
                if tokens_processed is not None:
                    tokens_processed = int(
                        tokens_processed
                    )

                _validate_scores(
                    adapter_name=adapter.name,
                    case_id=case.case_id,
                    candidates=case.candidates,
                    scores=scores,
                )

                ranked = sorted(
                    scores,
                    key=lambda item: item.probability,
                    reverse=True,
                )
                predicted = ranked[0].candidate
                confidence = ranked[0].probability
                correct = predicted == case.gold_candidate
                gold_probability = next(
                    score.probability
                    for score in scores
                    if score.candidate == case.gold_candidate
                )

                set_attribute(
                    case_span,
                    "decision.confidence",
                    confidence,
                )
                set_attribute(
                    case_span,
                    "decision.correct",
                    correct,
                )
                set_attribute(
                    case_span,
                    "decision.latency_ms",
                    latency_ms,
                )
                set_attribute(
                    case_span,
                    "decision.tokens_processed",
                    tokens_processed,
                )

                results.append(
                    CaseResult(
                        case_id=case.case_id,
                        predicted=predicted,
                        gold=case.gold_candidate,
                        confidence=confidence,
                        correct=correct,
                        latency_ms=latency_ms,
                        tokens_processed=tokens_processed,
                        gold_probability=gold_probability,
                    )
                )
                predictions.append(
                    Prediction(
                        confidence=confidence,
                        correct=correct,
                    )
                )

        accuracy = sum(
            1 for result in results if result.correct
        ) / len(results)
        brier = brier_score(predictions)
        ece = expected_calibration_error(predictions)
        report_macro_f1 = macro_f1(
            [result.gold for result in results],
            [result.predicted for result in results],
        )
        nll = multiclass_nll(
            [
                result.gold_probability
                for result in results
                if result.gold_probability is not None
            ]
        )

        latencies = [
            result.latency_ms
            for result in results
        ]
        mean_latency_ms = (
            sum(latencies) / len(latencies)
        )
        p50_latency_ms = _percentile(
            latencies,
            0.50,
        )
        p95_latency_ms = _percentile(
            latencies,
            0.95,
        )

        token_counts = [
            result.tokens_processed
            for result in results
            if result.tokens_processed is not None
        ]
        mean_tokens_processed = (
            sum(token_counts) / len(token_counts)
            if token_counts
            else None
        )

        set_attribute(
            benchmark_span,
            "benchmark.accuracy",
            accuracy,
        )
        set_attribute(
            benchmark_span,
            "benchmark.brier",
            brier,
        )
        set_attribute(
            benchmark_span,
            "benchmark.ece",
            ece,
        )
        set_attribute(
            benchmark_span,
            "benchmark.macro_f1",
            report_macro_f1,
        )
        set_attribute(
            benchmark_span,
            "benchmark.nll",
            nll,
        )
        set_attribute(
            benchmark_span,
            "benchmark.mean_latency_ms",
            mean_latency_ms,
        )
        set_attribute(
            benchmark_span,
            "benchmark.p50_latency_ms",
            p50_latency_ms,
        )
        set_attribute(
            benchmark_span,
            "benchmark.p95_latency_ms",
            p95_latency_ms,
        )
        set_attribute(
            benchmark_span,
            "benchmark.mean_tokens_processed_per_decision",
            mean_tokens_processed,
        )

        return BenchmarkReport(
            adapter=adapter.name,
            accuracy=accuracy,
            brier=brier,
            ece=ece,
            mean_latency_ms=mean_latency_ms,
            p50_latency_ms=p50_latency_ms,
            p95_latency_ms=p95_latency_ms,
            mean_tokens_processed_per_decision=(
                mean_tokens_processed
            ),
            coverage=risk_coverage(
                predictions,
                thresholds=thresholds,
            ),
            cases=results,
            macro_f1=report_macro_f1,
            nll=nll,
        )
