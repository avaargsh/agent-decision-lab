"""Build sealed profiles from an actually executed M5/M6 calibration report.

This is provenance-bound experiment evidence, not runtime model attestation
or authorization for tool execution.
"""
from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .benchmark import BenchmarkCase, load_jsonl
from .calibration_profile import build_calibration_profile
from .dataset_validation import validate_calibration_test_pair
from .provenance import dataset_provenance
from .quality_experiment import CalibratedQualityReport
from .stress_experiment import CalibratedStressReport


def build_replay_calibration_profile(
    report: CalibratedQualityReport | CalibratedStressReport,
    *,
    calibration_path: str | Path,
    calibration_cases: Sequence[BenchmarkCase],
    test_cases: Sequence[BenchmarkCase],
    model_ref: str,
    inventory_sha256: str,
) -> dict[str, Any]:
    """Freeze calibration-only fit and threshold selection with exact input bytes.

    The provided report must come from evaluating the supplied calibration
    cases. This function enforces reproducible identities and report
    invariants, but cannot independently prove the caller's model weights.
    """
    validate_calibration_test_pair(calibration_cases, test_cases)

    # Bind the file's *actual bytes* and ensure the fitted in-memory cases
    # still match the file. A path or a caller-supplied digest alone is weak.
    reloaded = load_jsonl(calibration_path)
    if list(calibration_cases) != reloaded:
        raise ValueError("calibration input bytes do not match fitted cases")

    source_cases = report.source_calibration.cases
    expected_ids = [case.case_id for case in calibration_cases]
    if [case.case_id for case in source_cases] != expected_ids:
        raise ValueError("report calibration case identities do not match")
    if any(
        result.gold != case.gold_candidate
        for result, case in zip(source_cases, calibration_cases, strict=True)
    ):
        raise ValueError("report calibration gold labels do not match")

    types = {
        case.decision_type for case in [*calibration_cases, *test_cases]
    }
    if len(types) != 1:
        raise ValueError("calibration profile requires one decision type")
    if report.source_calibration.adapter != f"{report.adapter}+calibrated":
        raise ValueError("calibration report adapter mismatch")
    if report.source_calibration.risk_budget != report.risk_budget:
        raise ValueError("calibration report risk budget mismatch")

    operating_point = report.source_calibration.operating_point
    selection = report.threshold_selection
    threshold = report.transfer_threshold
    if selection == "calibration_risk_budget":
        if operating_point is None or operating_point.threshold != threshold:
            raise ValueError("threshold must match calibration operating point")
    elif selection == "calibration_median_confidence_fallback":
        if operating_point is not None or threshold != statistics.median(
            case.confidence for case in source_cases
        ):
            raise ValueError("median fallback threshold mismatch")
    else:
        raise ValueError("unsupported threshold selection")

    provenance = dataset_provenance(calibration_path, calibration_cases)
    return build_calibration_profile(
        adapter=report.adapter,
        model_ref=model_ref,
        decision_type=types.pop(),
        calibration={
            "sha256": provenance["sha256"],
            "case_count": provenance["case_count"],
            "case_ids": provenance["case_ids"],
        },
        temperature=report.temperature_fit.temperature,
        execute_threshold=threshold,
        risk_budget=report.risk_budget,
        threshold_selection=selection,
        inventory_sha256=inventory_sha256,
    )
