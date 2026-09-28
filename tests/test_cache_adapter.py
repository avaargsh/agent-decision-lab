from decision_lab.benchmark import BenchmarkCase
from decision_lab.cache_adapter import CachingDecisionAdapter
from decision_lab.experiment import run_calibrated_experiment
from decision_lab.models import CandidateScore, DecisionRequest


class CountingAdapter:
    name = "counting"

    def __init__(self):
        self.calls = 0
        self.last_tokens_processed = 10

    def score(self, request: DecisionRequest):
        self.calls += 1
        if request.context["kind"] == "a":
            return [
                CandidateScore("a", 0.8),
                CandidateScore("b", 0.2),
            ]
        return [
            CandidateScore("a", 0.2),
            CandidateScore("b", 0.8),
        ]


def cases():
    return [
        BenchmarkCase(
            case_id="1",
            decision_type="router",
            context={"kind": "a"},
            candidates=["a", "b"],
            gold_candidate="a",
            metadata={},
        ),
        BenchmarkCase(
            case_id="2",
            decision_type="router",
            context={"kind": "b"},
            candidates=["a", "b"],
            gold_candidate="b",
            metadata={},
        ),
    ]


def test_cache_adapter_reuses_scores() -> None:
    base = CountingAdapter()
    cache = CachingDecisionAdapter(base)
    request = DecisionRequest(
        decision_type="router",
        candidates=["a", "b"],
        context={"kind": "a"},
    )

    first = list(cache.score(request))
    first_latency = cache.last_latency_ms
    second = list(cache.score(request))

    assert first == second
    assert base.calls == 1
    assert cache.last_tokens_processed == 10
    assert cache.last_latency_ms == first_latency


def test_calibrated_experiment_does_not_repeat_test_inference() -> None:
    base = CountingAdapter()

    run_calibrated_experiment(
        base,
        calibration_cases=cases(),
        test_cases=cases(),
    )

    # Two calibration cases + two test cases.
    # The calibrated report reuses the test scores.
    assert base.calls == 4
