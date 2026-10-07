from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Sequence

from .abstention_groups import AbstentionGroupedReport, summarize_abstention_groups
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


@dataclass(frozen=True)
class CalibratedQualityReport:
    adapter: str
    temperature_fit: TemperatureFit
    transfer_threshold: float
    threshold_selection: str
    risk_budget: float
    source_calibration: BenchmarkReport
    closed_test: BenchmarkReport
    threshold_transfer: ThresholdTransferReport
    permutation: PermutationRobustnessReport
    abstention: AbstentionRobustnessReport
    abstention_groups: AbstentionGroupedReport


def run_calibrated_quality_experiment(
    adapter: DecisionAdapter,
    *,
    calibration_cases: Sequence[BenchmarkCase],
    test_cases: Sequence[BenchmarkCase],
    abstention_cases: Sequence[BenchmarkCase],
    risk_budget: float = 0.5,
    min_coverage: float = 0.25,
    threshold_grid: Sequence[float] = tuple(
        value / 20.0
        for value in range(21)
    ),
) -> CalibratedQualityReport:
    """Fit only on source-grounded calibration, then freeze policy on tests."""
    if not calibration_cases:
        raise ValueError("calibration_cases must not be empty")
    if not test_cases:
        raise ValueError("test_cases must not be empty")
    if not abstention_cases:
        raise ValueError("abstention_cases must not be empty")
    if not 0.0 <= risk_budget <= 1.0:
        raise ValueError("risk_budget must be between 0 and 1")
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError("min_coverage must be between 0 and 1")

    if any(
        case.gold_candidate is None
        for case in calibration_cases
    ):
        raise ValueError(
            "quality calibration requires single-gold cases"
        )
    if any(
        case.gold_candidate is None
        for case in test_cases
    ):
        raise ValueError(
            "quality closed test requires single-gold cases"
        )
    if any(
        not bool(case.metadata.get("expected_abstain", False))
        for case in abstention_cases
    ):
        raise ValueError(
            "quality abstention set must contain expected-abstain cases only"
        )

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

    closed_test = run_benchmark(
        calibrated,
        test_cases,
        thresholds=[threshold],
    )
    transfer = evaluate_threshold_transfer(
        source,
        closed_test,
        threshold=threshold,
    )
    permutation = evaluate_permutation_robustness(
        calibrated,
        test_cases,
    )
    abstention = evaluate_abstention_robustness(
        calibrated,
        abstention_cases,
        threshold=threshold,
    )
    grouped = summarize_abstention_groups(
        abstention_cases,
        abstention,
    )

    return CalibratedQualityReport(
        adapter=adapter.name,
        temperature_fit=fit,
        transfer_threshold=threshold,
        threshold_selection=threshold_selection,
        risk_budget=risk_budget,
        source_calibration=source,
        closed_test=closed_test,
        threshold_transfer=transfer,
        permutation=permutation,
        abstention=abstention,
        abstention_groups=grouped,
    )
