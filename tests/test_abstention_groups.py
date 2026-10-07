from __future__ import annotations

from decision_lab.abstention_groups import (
    abstention_reason,
    summarize_abstention_groups,
)
from decision_lab.adapters import MappingScoreAdapter
from decision_lab.benchmark import BenchmarkCase
from decision_lab.robustness import (
    evaluate_abstention_robustness,
)


def _case(
    case_id: str,
    *,
    reason: str,
    confidence: float,
) -> tuple[BenchmarkCase, float]:
    if reason == "missing_candidate":
        metadata = {
            "expected_abstain": True,
            "coverage": "missing_candidate",
            "ambiguity": "clear",
            "automation_risk": "medium",
        }
        gold = "missing"
    elif reason == "unsupported":
        metadata = {
            "expected_abstain": True,
            "coverage": "unsupported",
            "ambiguity": "clear",
            "automation_risk": "high",
        }
        gold = None
    elif reason == "underspecified":
        metadata = {
            "expected_abstain": True,
            "coverage": "covered",
            "ambiguity": "underspecified",
            "automation_risk": "low",
        }
        gold = None
    else:
        metadata = {
            "expected_abstain": True,
            "coverage": "covered",
            "ambiguity": "multi_valid",
            "automation_risk": "medium",
        }
        gold = None

    return (
        BenchmarkCase(
            case_id=case_id,
            decision_type="router",
            context={"confidence": confidence},
            candidates=["a", "b"],
            gold_candidate=gold,
            metadata=metadata,
        ),
        confidence,
    )


def test_abstention_reason_keeps_failure_semantics_distinct() -> None:
    cases = [
        _case("a", reason=reason, confidence=0.5)[0]
        for reason in (
            "underspecified",
            "multi_valid",
            "missing_candidate",
            "unsupported",
        )
    ]

    assert [abstention_reason(case) for case in cases] == [
        "underspecified",
        "multi_valid",
        "missing_candidate",
        "unsupported",
    ]


def test_grouped_far_is_computed_from_one_abstention_report() -> None:
    pairs = [
        _case("u1", reason="underspecified", confidence=0.95),
        _case("u2", reason="underspecified", confidence=0.60),
        _case("m1", reason="missing_candidate", confidence=0.90),
        _case("x1", reason="unsupported", confidence=0.40),
    ]
    cases = [case for case, _ in pairs]

    adapter = MappingScoreAdapter(
        lambda request: {
            "a": request.context["confidence"],
            "b": 1.0 - request.context["confidence"],
        }
    )
    report = evaluate_abstention_robustness(
        adapter,
        cases,
        threshold=0.8,
    )

    grouped = summarize_abstention_groups(
        cases,
        report,
        group_keys=["abstention_reason"],
    )

    metrics = {
        item.group_value: item
        for item in grouped.groups
    }
    assert metrics["underspecified"].case_count == 2
    assert metrics["underspecified"].false_accept_rate == 0.5
    assert metrics["missing_candidate"].false_accept_rate == 1.0
    assert metrics["unsupported"].false_accept_rate == 0.0
