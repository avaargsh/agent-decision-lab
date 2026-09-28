from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, Sequence

from .benchmark import BenchmarkCase
from .metrics import Prediction, brier_score, expected_calibration_error
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


@dataclass(frozen=True)
class BenchmarkReport:
    adapter: str
    accuracy: float
    brier: float
    ece: float
    mean_latency_ms: float
    coverage: list[CoveragePoint]
    cases: list[CaseResult]


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
                latency_ms = (time.perf_counter() - started) * 1000.0

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

                results.append(
                    CaseResult(
                        case_id=case.case_id,
                        predicted=predicted,
                        gold=case.gold_candidate,
                        confidence=confidence,
                        correct=correct,
                        latency_ms=latency_ms,
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
        mean_latency_ms = (
            sum(result.latency_ms for result in results)
            / len(results)
        )

        set_attribute(benchmark_span, "benchmark.accuracy", accuracy)
        set_attribute(benchmark_span, "benchmark.brier", brier)
        set_attribute(benchmark_span, "benchmark.ece", ece)
        set_attribute(
            benchmark_span,
            "benchmark.mean_latency_ms",
            mean_latency_ms,
        )

        return BenchmarkReport(
            adapter=adapter.name,
            accuracy=accuracy,
            brier=brier,
            ece=ece,
            mean_latency_ms=mean_latency_ms,
            coverage=risk_coverage(
                predictions,
                thresholds=thresholds,
            ),
            cases=results,
        )
