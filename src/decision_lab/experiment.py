from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .benchmark import BenchmarkCase
from .calibrated_adapter import CalibratedAdapter
from .calibration import TemperatureCalibrator
from .calibration_fit import CalibrationExample, TemperatureFit, fit_temperature
from .models import DecisionRequest
from .runner import BenchmarkReport, DecisionAdapter, run_benchmark


def calibration_examples(
    adapter: DecisionAdapter,
    cases: Sequence[BenchmarkCase],
) -> list[CalibrationExample]:
    examples: list[CalibrationExample] = []

    for case in cases:
        request = DecisionRequest(
            decision_type=case.decision_type,
            candidates=case.candidates,
            context=case.context,
        )
        examples.append(
            CalibrationExample(
                scores=list(adapter.score(request)),
                gold_candidate=case.gold_candidate,
            )
        )

    return examples


def run_calibrated_experiment(
    adapter: DecisionAdapter,
    *,
    calibration_cases: Sequence[BenchmarkCase],
    test_cases: Sequence[BenchmarkCase],
) -> tuple[TemperatureFit, BenchmarkReport, BenchmarkReport]:
    fit = fit_temperature(
        calibration_examples(
            adapter,
            calibration_cases,
        )
    )

    raw = run_benchmark(adapter, test_cases)
    calibrated = run_benchmark(
        CalibratedAdapter(
            base=adapter,
            calibrator=TemperatureCalibrator(
                fit.temperature
            ),
        ),
        test_cases,
    )

    return fit, raw, calibrated


def write_experiment_report(
    path: str | Path,
    *,
    fit: TemperatureFit,
    raw: BenchmarkReport,
    calibrated: BenchmarkReport,
    metadata: dict | None = None,
) -> None:
    payload = {
        "metadata": metadata or {},
        "temperature_fit": asdict(fit),
        "raw": asdict(raw),
        "calibrated": asdict(calibrated),
    }
    Path(path).write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
