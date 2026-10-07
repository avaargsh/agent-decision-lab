from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .benchmark import BenchmarkCase
from .robustness import AbstentionRobustnessReport


@dataclass(frozen=True)
class AbstentionGroupMetric:
    group_key: str
    group_value: str
    case_count: int
    accepted_count: int
    false_accept_rate: float


@dataclass(frozen=True)
class AbstentionGroupedReport:
    case_count: int
    groups: list[AbstentionGroupMetric]


def abstention_reason(case: BenchmarkCase) -> str:
    coverage = case.metadata.get("coverage")
    ambiguity = case.metadata.get("ambiguity")

    if coverage == "missing_candidate":
        return "missing_candidate"
    if coverage == "unsupported":
        return "unsupported"
    if ambiguity == "underspecified":
        return "underspecified"
    if ambiguity == "multi_valid":
        return "multi_valid"
    return "other"


def summarize_abstention_groups(
    cases: Sequence[BenchmarkCase],
    report: AbstentionRobustnessReport,
    *,
    group_keys: Sequence[str] = (
        "abstention_reason",
        "coverage",
        "ambiguity",
        "automation_risk",
    ),
) -> AbstentionGroupedReport:
    if not cases:
        raise ValueError("cases must not be empty")
    if len(cases) != report.case_count:
        raise ValueError(
            "case count does not match abstention report"
        )

    by_id = {case.case_id: case for case in cases}
    if len(by_id) != len(cases):
        raise ValueError("case ids must be unique")

    result_by_id = {
        item.case_id: item
        for item in report.cases
    }
    if set(result_by_id) != set(by_id):
        raise ValueError(
            "abstention report cases do not match supplied cases"
        )

    buckets: dict[
        tuple[str, str],
        list[bool],
    ] = {}

    for case in cases:
        if not bool(
            case.metadata.get(
                "expected_abstain",
                False,
            )
        ):
            raise ValueError(
                "grouped abstention FAR requires expected-abstain cases only: "
                f"{case.case_id}"
            )

        result = result_by_id[case.case_id]

        for group_key in group_keys:
            if group_key == "abstention_reason":
                value = abstention_reason(case)
            else:
                raw = case.metadata.get(group_key)
                if raw is None:
                    value = "unknown"
                elif isinstance(raw, str):
                    value = raw
                else:
                    raise ValueError(
                        f"{case.case_id} group field {group_key!r} "
                        "must be a string"
                    )

            buckets.setdefault(
                (group_key, value),
                [],
            ).append(result.accepted)

    groups = [
        AbstentionGroupMetric(
            group_key=key,
            group_value=value,
            case_count=len(accepted),
            accepted_count=sum(
                1 for item in accepted if item
            ),
            false_accept_rate=(
                sum(
                    1
                    for item in accepted
                    if item
                )
                / len(accepted)
            ),
        )
        for (key, value), accepted in sorted(
            buckets.items()
        )
    ]

    return AbstentionGroupedReport(
        case_count=len(cases),
        groups=groups,
    )
