from __future__ import annotations

from decision_lab.adapters import MappingScoreAdapter
from decision_lab.benchmark import BenchmarkCase
from decision_lab.quality_experiment import (
    run_calibrated_quality_experiment,
)


def _closed(case_id: str, split: str, intent: str, gold: str) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=case_id,
        decision_type="router",
        context={"intent": intent},
        candidates=["a", "b"],
        gold_candidate=gold,
        metadata={"split": split},
    )


def _abstain(
    case_id: str,
    *,
    reason: str,
    confidence_marker: str,
) -> BenchmarkCase:
    if reason == "missing_candidate":
        coverage = "missing_candidate"
        ambiguity = "clear"
        gold = "missing"
    elif reason == "unsupported":
        coverage = "unsupported"
        ambiguity = "clear"
        gold = None
    elif reason == "underspecified":
        coverage = "covered"
        ambiguity = "underspecified"
        gold = None
    else:
        coverage = "covered"
        ambiguity = "multi_valid"
        gold = None

    return BenchmarkCase(
        case_id=case_id,
        decision_type="router",
        context={"intent": confidence_marker},
        candidates=["a", "b"],
        gold_candidate=gold,
        metadata={
            "split": "test",
            "expected_abstain": True,
            "coverage": coverage,
            "ambiguity": ambiguity,
            "automation_risk": "medium",
        },
    )


def test_quality_experiment_freezes_calibration_threshold_for_abstention() -> None:
    def mapping(request):
        intent = request.context["intent"]
        if intent in {"cal-a", "test-a"}:
            return {"a": 0.95, "b": 0.05}
        if intent in {"cal-b", "test-b"}:
            return {"a": 0.05, "b": 0.95}
        if intent == "abstain-high":
            return {"a": 0.9, "b": 0.1}
        return {"a": 0.55, "b": 0.45}

    calibration = [
        _closed("c1", "calibration", "cal-a", "a"),
        _closed("c2", "calibration", "cal-b", "b"),
    ]
    test = [
        _closed("t1", "test", "test-a", "a"),
        _closed("t2", "test", "test-b", "b"),
    ]
    abstention = [
        _abstain(
            "a1",
            reason="underspecified",
            confidence_marker="abstain-high",
        ),
        _abstain(
            "a2",
            reason="unsupported",
            confidence_marker="abstain-low",
        ),
    ]

    report = run_calibrated_quality_experiment(
        MappingScoreAdapter(mapping),
        calibration_cases=calibration,
        test_cases=test,
        abstention_cases=abstention,
        risk_budget=0.0,
        min_coverage=0.5,
        threshold_grid=[0.99],
    )

    assert report.transfer_threshold == 0.99
    assert report.threshold_selection == "calibration_risk_budget"
    assert report.closed_test.accuracy == 1.0
    assert report.abstention.threshold == 0.99
    assert report.abstention.false_accept_rate == 0.5

    by_reason = {
        item.group_value: item.false_accept_rate
        for item in report.abstention_groups.groups
        if item.group_key == "abstention_reason"
    }
    assert by_reason["underspecified"] == 1.0
    assert by_reason["unsupported"] == 0.0


def test_quality_experiment_rejects_abstention_case_in_calibration() -> None:
    calibration = [
        BenchmarkCase(
            case_id="bad",
            decision_type="router",
            context={"intent": "ambiguous"},
            candidates=["a", "b"],
            gold_candidate=None,
            metadata={"split": "calibration"},
        )
    ]

    try:
        run_calibrated_quality_experiment(
            MappingScoreAdapter(
                lambda request: {"a": 0.5, "b": 0.5}
            ),
            calibration_cases=calibration,
            test_cases=[_closed("t", "test", "test-a", "a")],
            abstention_cases=[
                _abstain(
                    "a",
                    reason="unsupported",
                    confidence_marker="low",
                )
            ],
        )
    except ValueError as exc:
        assert "single-gold" in str(exc)
    else:
        raise AssertionError(
            "abstention data must not participate in calibration"
        )
