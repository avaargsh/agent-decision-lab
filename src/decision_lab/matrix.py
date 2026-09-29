from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

from .runner import BenchmarkReport


@dataclass(frozen=True)
class BenchmarkArmSummary:
    arm: str
    adapter: str
    accuracy: float
    brier: float
    ece: float
    p95_latency_ms: float
    mean_tokens_processed_per_decision: float | None


def build_benchmark_matrix(
    reports: Mapping[str, BenchmarkReport],
) -> tuple[BenchmarkArmSummary, ...]:
    """Normalize benchmark reports into a comparable, stable experiment matrix."""
    if not reports:
        raise ValueError("reports must not be empty")

    return tuple(
        BenchmarkArmSummary(
            arm=arm,
            adapter=report.adapter,
            accuracy=report.accuracy,
            brier=report.brier,
            ece=report.ece,
            p95_latency_ms=report.p95_latency_ms,
            mean_tokens_processed_per_decision=(
                report.mean_tokens_processed_per_decision
            ),
        )
        for arm, report in sorted(reports.items())
    )


def benchmark_matrix_dict(
    reports: Mapping[str, BenchmarkReport],
) -> dict[str, object]:
    rows = build_benchmark_matrix(reports)
    return {
        "schema_version": "v1",
        "arms": [asdict(row) for row in rows],
    }
