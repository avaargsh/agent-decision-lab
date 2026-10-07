from __future__ import annotations

from decision_lab.adapters import MappingScoreAdapter
from decision_lab.benchmark import BenchmarkCase
from decision_lab.stress_experiment import (
    run_calibrated_stress_experiment,
)


def _base_case(
    case_id: str,
    *,
    split: str,
    intent: str,
    gold: str,
) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=case_id,
        decision_type="router",
        context={"intent": intent},
        candidates=["a", "b"],
        gold_candidate=gold,
        metadata={"split": split},
    )


def _stress_case(
    case_id: str,
    *,
    intent: str,
    gold: str,
    k: int,
    missing: bool,
) -> BenchmarkCase:
    candidates = ["a", "b", *[
        f"d{index}"
        for index in range(2, k)
    ]]
    if missing:
        candidates = [
            candidate
            for candidate in candidates
            if candidate != gold
        ]
        candidates.append(f"extra-{k}")
    return BenchmarkCase(
        case_id=case_id,
        decision_type="router",
        context={"intent": intent},
        candidates=candidates,
        gold_candidate=gold,
        metadata={
            "split": "test",
            "candidate_count": k,
            "expected_abstain": missing,
            "shift": (
                "missing_candidate"
                if missing
                else "candidate_count"
            ),
        },
    )


def test_calibrated_stress_reuses_source_threshold_across_k() -> None:
    def mapping(request):
        intent = request.context["intent"]
        preferred = "a" if "a-" in intent else "b"
        scores = {
            candidate: 0.01
            for candidate in request.candidates
        }
        if preferred in scores:
            scores[preferred] = 0.9
        else:
            scores[request.candidates[0]] = 0.6
        total = sum(scores.values())
        return {
            candidate: value / total
            for candidate, value in scores.items()
        }

    calibration = [
        _base_case(
            "c1",
            split="calibration",
            intent="a-cal",
            gold="a",
        ),
        _base_case(
            "c2",
            split="calibration",
            intent="b-cal",
            gold="b",
        ),
    ]
    base_test = [
        _base_case(
            "t1",
            split="test",
            intent="a-test",
            gold="a",
        ),
        _base_case(
            "t2",
            split="test",
            intent="b-test",
            gold="b",
        ),
    ]
    covered = [
        _stress_case(
            f"covered-{k}",
            intent="a-stress",
            gold="a",
            k=k,
            missing=False,
        )
        for k in (5, 10)
    ]
    missing = [
        _stress_case(
            f"missing-{k}",
            intent="a-missing",
            gold="a",
            k=k,
            missing=True,
        )
        for k in (5, 10)
    ]

    report = run_calibrated_stress_experiment(
        MappingScoreAdapter(mapping),
        calibration_cases=calibration,
        base_test_cases=base_test,
        covered_stress_cases=covered,
        missing_stress_cases=missing,
        risk_budget=0.0,
        min_coverage=0.5,
        permutation_cases_per_k=1,
    )

    assert report.threshold_selection == "calibration_risk_budget"
    assert [item.candidate_count for item in report.by_candidate_count] == [
        5,
        10,
    ]
    for item in report.by_candidate_count:
        assert (
            item.threshold_transfer.threshold
            == report.transfer_threshold
        )
        assert (
            item.abstention.threshold
            == report.transfer_threshold
        )
        assert item.permutation is not None


def test_stress_experiment_rejects_mismatched_k_sets() -> None:
    adapter = MappingScoreAdapter(
        lambda request: {
            candidate: 1.0 / len(request.candidates)
            for candidate in request.candidates
        }
    )
    calibration = [
        _base_case(
            "c1",
            split="calibration",
            intent="a-cal",
            gold="a",
        )
    ]
    base_test = [
        _base_case(
            "t1",
            split="test",
            intent="a-test",
            gold="a",
        )
    ]
    covered = [
        _stress_case(
            "covered-5",
            intent="a-covered",
            gold="a",
            k=5,
            missing=False,
        )
    ]
    missing = [
        _stress_case(
            "missing-10",
            intent="a-missing",
            gold="a",
            k=10,
            missing=True,
        )
    ]

    try:
        run_calibrated_stress_experiment(
            adapter,
            calibration_cases=calibration,
            base_test_cases=base_test,
            covered_stress_cases=covered,
            missing_stress_cases=missing,
            permutation_cases_per_k=0,
        )
    except ValueError as exc:
        assert "same candidate counts" in str(exc)
    else:
        raise AssertionError("mismatched K suites must fail")
