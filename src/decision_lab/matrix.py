from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

from .runner import BenchmarkReport


@dataclass(frozen=True)
class BenchmarkArmSummary:
    arm: str
    adapter: str
    accuracy: float
    macro_f1: float
    nll: float
    brier: float
    ece: float
    p95_latency_ms: float
    mean_tokens_processed_per_decision: float | None
    risk_budget: float | None
    operating_threshold: float | None
    operating_coverage: float | None
    operating_risk: float | None
    false_automation_rate: float | None
    fallback_rate: float | None


def build_benchmark_matrix(
    reports: Mapping[str, BenchmarkReport],
) -> tuple[BenchmarkArmSummary, ...]:
    """Normalize benchmark reports into a comparable, stable experiment matrix."""
    if not reports:
        raise ValueError("reports must not be empty")

    rows: list[BenchmarkArmSummary] = []
    for arm, report in sorted(reports.items()):
        operating = report.operating_point
        rows.append(
            BenchmarkArmSummary(
                arm=arm,
                adapter=report.adapter,
                accuracy=report.accuracy,
                macro_f1=report.macro_f1,
                nll=report.nll,
                brier=report.brier,
                ece=report.ece,
                p95_latency_ms=report.p95_latency_ms,
                mean_tokens_processed_per_decision=(
                    report.mean_tokens_processed_per_decision
                ),
                risk_budget=report.risk_budget,
                operating_threshold=(
                    operating.threshold if operating is not None else None
                ),
                operating_coverage=(
                    operating.coverage if operating is not None else None
                ),
                operating_risk=(
                    operating.risk if operating is not None else None
                ),
                false_automation_rate=(
                    operating.false_automation_rate
                    if operating is not None
                    else None
                ),
                fallback_rate=(
                    operating.fallback_rate if operating is not None else None
                ),
            )
        )
    return tuple(rows)


def benchmark_matrix_dict(
    reports: Mapping[str, BenchmarkReport],
) -> dict[str, object]:
    rows = build_benchmark_matrix(reports)
    return {
        "schema_version": "v1",
        "arms": [asdict(row) for row in rows],
    }
