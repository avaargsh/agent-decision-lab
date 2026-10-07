from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Sequence

from .benchmark import BenchmarkCase
from .cache_adapter import CachingDecisionAdapter
from .calibrated_adapter import CalibratedAdapter
from .calibration import TemperatureCalibrator
from .calibration_fit import TemperatureFit, fit_temperature
from .experiment import calibration_examples
from .robustness import (
    AbstentionRobustnessReport,
    PermutationRobustnessReport,
    ThresholdTransferReport,
    evaluate_abstention_robustness,
    evaluate_permutation_robustness,
    evaluate_threshold_transfer,
)
from .runner import BenchmarkReport, DecisionAdapter, run_benchmark
from .stress_suite import group_by_candidate_count


@dataclass(frozen=True)
class StressKResult:
    candidate_count: int
    covered: BenchmarkReport
    abstention: AbstentionRobustnessReport
    threshold_transfer: ThresholdTransferReport
    permutation: PermutationRobustnessReport | None


@dataclass(frozen=True)
class CalibratedStressReport:
    adapter: str
    temperature_fit: TemperatureFit
    transfer_threshold: float
    threshold_selection: str
    risk_budget: float
    source_calibration: BenchmarkReport
    base_test: BenchmarkReport
    by_candidate_count: list[StressKResult]


def run_calibrated_stress_experiment(
    adapter: DecisionAdapter,
    *,
    calibration_cases: Sequence[BenchmarkCase],
    base_test_cases: Sequence[BenchmarkCase],
    covered_stress_cases: Sequence[BenchmarkCase],
    missing_stress_cases: Sequence[BenchmarkCase],
    risk_budget: float = 0.5,
    min_coverage: float = 0.25,
    permutation_cases_per_k: int = 2,
    threshold_grid: Sequence[float] = tuple(
        value / 20.0
        for value in range(21)
    ),
) -> CalibratedStressReport:
    """Fit only on calibration data, then reuse one threshold across stress K.

    The model score cache is shared across raw/calibrated evaluations. Cached
    requests preserve the original model latency/token metadata so repeated
    metric views do not trigger duplicate inference.
    """
    if not 0.0 <= risk_budget <= 1.0:
        raise ValueError("risk_budget must be between 0 and 1")
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError("min_coverage must be between 0 and 1")
    if permutation_cases_per_k < 0:
        raise ValueError("permutation_cases_per_k must be >= 0")

    cached = CachingDecisionAdapter(adapter)
    fit = fit_temperature(
        calibration_examples(
            cached,
            calibration_cases,
        )
    )
    calibrated = CalibratedAdapter(
        base=cached,
        calibrator=TemperatureCalibrator(
            fit.temperature
        ),
    )

    source = run_benchmark(
        calibrated,
        calibration_cases,
        thresholds=threshold_grid,
        risk_budget=risk_budget,
        min_coverage=min_coverage,
    )

    if source.operating_point is not None:
        threshold = source.operating_point.threshold
        threshold_selection = "calibration_risk_budget"
    else:
        threshold = statistics.median(
            case.confidence
            for case in source.cases
        )
        threshold_selection = "calibration_median_confidence_fallback"

    base_test = run_benchmark(
        calibrated,
        base_test_cases,
        thresholds=[threshold],
    )

    covered_by_k = group_by_candidate_count(
        covered_stress_cases
    )
    missing_by_k = group_by_candidate_count(
        missing_stress_cases
    )
    if set(covered_by_k) != set(missing_by_k):
        raise ValueError(
            "covered and missing stress suites must have the same candidate counts"
        )

    results: list[StressKResult] = []
    for candidate_count in sorted(covered_by_k):
        covered_cases = covered_by_k[candidate_count]
        missing_cases = missing_by_k[candidate_count]

        covered_report = run_benchmark(
            calibrated,
            covered_cases,
            thresholds=[threshold],
        )
        abstention_report = evaluate_abstention_robustness(
            calibrated,
            [*covered_cases, *missing_cases],
            threshold=threshold,
        )
        transfer_report = evaluate_threshold_transfer(
            source,
            covered_report,
            threshold=threshold,
        )

        permutation_report = None
        if permutation_cases_per_k:
            permutation_report = evaluate_permutation_robustness(
                calibrated,
                covered_cases[:permutation_cases_per_k],
            )

        results.append(
            StressKResult(
                candidate_count=candidate_count,
                covered=covered_report,
                abstention=abstention_report,
                threshold_transfer=transfer_report,
                permutation=permutation_report,
            )
        )

    return CalibratedStressReport(
        adapter=adapter.name,
        temperature_fit=fit,
        transfer_threshold=threshold,
        threshold_selection=threshold_selection,
        risk_budget=risk_budget,
        source_calibration=source,
        base_test=base_test,
        by_candidate_count=results,
    )
