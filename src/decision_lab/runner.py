from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, Sequence

from .benchmark import BenchmarkCase
from .metrics import Prediction, brier_score, expected_calibration_error
from .models import CandidateScore, DecisionRequest
from .selective import CoveragePoint, risk_coverage


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

    for case in cases:
        request = DecisionRequest(
            decision_type=case.decision_type,
            candidates=case.candidates,
            context=case.context,
        )

        started = time.perf_counter()
        scores = list(adapter.score(request))
        latency_ms = (time.perf_counter() - started) * 1000.0

        if {s.candidate for s in scores} != set(case.candidates):
            raise ValueError(
                f"{adapter.name} returned an invalid candidate set for {case.case_id}"
            )

        ranked = sorted(scores, key=lambda item: item.probability, reverse=True)
        predicted = ranked[0].candidate
        confidence = ranked[0].probability
        correct = predicted == case.gold_candidate

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
        predictions.append(Prediction(confidence=confidence, correct=correct))

    accuracy = sum(1 for result in results if result.correct) / len(results)

    return BenchmarkReport(
        adapter=adapter.name,
        accuracy=accuracy,
        brier=brier_score(predictions),
        ece=expected_calibration_error(predictions),
        mean_latency_ms=sum(r.latency_ms for r in results) / len(results),
        coverage=risk_coverage(predictions, thresholds=thresholds),
        cases=results,
    )
