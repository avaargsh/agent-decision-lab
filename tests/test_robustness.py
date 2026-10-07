from decision_lab.adapters import MappingScoreAdapter
from decision_lab.benchmark import BenchmarkCase
from decision_lab.models import CandidateScore
from decision_lab.robustness import (
    evaluate_abstention_robustness,
    evaluate_permutation_robustness,
    evaluate_threshold_transfer,
)
from decision_lab.runner import run_benchmark


def test_candidate_permutation_flip_rate_detects_order_sensitive_adapter() -> None:
    class FirstCandidateAdapter:
        name = "first-candidate"

        def score(self, request):
            count = len(request.candidates)
            remainder = 0.1 / (count - 1)
            return [
                CandidateScore(
                    candidate,
                    0.9 if index == 0 else remainder,
                )
                for index, candidate in enumerate(request.candidates)
            ]

    cases = [
        BenchmarkCase(
            case_id="p1",
            decision_type="router",
            context={},
            candidates=["a", "b"],
            gold_candidate="a",
            metadata={},
        ),
        BenchmarkCase(
            case_id="p2",
            decision_type="router",
            context={},
            candidates=["b", "a"],
            gold_candidate="b",
            metadata={},
        ),
    ]

    report = evaluate_permutation_robustness(
        FirstCandidateAdapter(),
        cases,
    )

    assert report.baseline_accuracy == 1.0
    assert report.comparison_count == 2
    assert report.flip_rate == 1.0
    assert report.perturbed_accuracy == 0.0


def test_missing_candidate_far_and_covered_frr_are_separate() -> None:
    def mapping(request):
        intent = request.context["intent"]
        if intent == "covered-high":
            return {"a": 0.9, "b": 0.1}
        if intent == "covered-low":
            return {"a": 0.6, "b": 0.4}
        if intent == "missing-high":
            return {"a": 0.95, "b": 0.05}
        return {"a": 0.55, "b": 0.45}

    cases = [
        BenchmarkCase(
            "c1",
            "router",
            {"intent": "covered-high"},
            ["a", "b"],
            "a",
            {"expected_abstain": False},
        ),
        BenchmarkCase(
            "c2",
            "router",
            {"intent": "covered-low"},
            ["a", "b"],
            "a",
            {"expected_abstain": False},
        ),
        BenchmarkCase(
            "m1",
            "router",
            {"intent": "missing-high"},
            ["a", "b"],
            "missing-tool",
            {"expected_abstain": True, "shift": "missing_candidate"},
        ),
        BenchmarkCase(
            "o1",
            "router",
            {"intent": "ood-low"},
            ["a", "b"],
            "unsupported",
            {"expected_abstain": True, "shift": "ood"},
        ),
    ]

    report = evaluate_abstention_robustness(
        MappingScoreAdapter(mapping),
        cases,
        threshold=0.8,
    )

    assert report.false_accept_rate == 0.5
    assert report.false_rejection_rate == 0.5
    assert report.covered_accuracy == 1.0
    assert report.automation_coverage == 0.5
    assert report.unsafe_automation_rate == 0.25


def test_threshold_transfer_reuses_source_operating_point_without_refit() -> None:
    source_cases = [
        BenchmarkCase("s1", "router", {"id": "s1"}, ["a", "b"], "a", {}),
        BenchmarkCase("s2", "router", {"id": "s2"}, ["a", "b"], "b", {}),
    ]
    target_cases = [
        BenchmarkCase("t1", "router", {"id": "t1"}, ["a", "b"], "a", {}),
        BenchmarkCase("t2", "router", {"id": "t2"}, ["a", "b"], "b", {}),
    ]

    source = run_benchmark(
        MappingScoreAdapter(
            lambda request: (
                {"a": 0.9, "b": 0.1}
                if request.context["id"] == "s1"
                else {"a": 0.7, "b": 0.3}
            )
        ),
        source_cases,
        thresholds=[0.8],
        risk_budget=0.0,
    )
    target = run_benchmark(
        MappingScoreAdapter(
            lambda request: (
                {"a": 0.85, "b": 0.15}
                if request.context["id"] == "t1"
                else {"a": 0.9, "b": 0.1}
            )
        ),
        target_cases,
        thresholds=[0.8],
    )

    transfer = evaluate_threshold_transfer(source, target)

    assert transfer.threshold == 0.8
    assert transfer.source.coverage == 0.5
    assert transfer.source.risk == 0.0
    assert transfer.target.coverage == 1.0
    assert transfer.target.risk == 0.5
    assert transfer.coverage_delta == 0.5
    assert transfer.risk_delta == 0.5
    assert transfer.false_automation_rate_delta == 0.5
