from decision_lab.adapters import MappingScoreAdapter
from decision_lab.benchmark import BenchmarkCase
from decision_lab.fallback_eval import evaluate_system2_fallback
from decision_lab.runner import run_benchmark


CASES = [
    BenchmarkCase(
        case_id="high",
        decision_type="mcp_tool_router",
        context={"intent": "metrics"},
        candidates=["prometheus.query", "logs.search"],
        gold_candidate="prometheus.query",
        metadata={},
    ),
    BenchmarkCase(
        case_id="low",
        decision_type="mcp_tool_router",
        context={"intent": "logs"},
        candidates=["prometheus.query", "logs.search"],
        gold_candidate="logs.search",
        metadata={},
    ),
]


def test_only_low_confidence_cases_execute_system2():
    fast = MappingScoreAdapter(
        lambda request: (
            {"prometheus.query": 0.95, "logs.search": 0.05}
            if request.context["intent"] == "metrics"
            else {"prometheus.query": 0.55, "logs.search": 0.45}
        )
    )
    report = run_benchmark(fast, CASES)

    fallback = MappingScoreAdapter(
        lambda request: (
            {"prometheus.query": 0.1, "logs.search": 0.9}
        )
    )
    fallback.last_tokens_processed = 42

    measured = evaluate_system2_fallback(
        report,
        CASES,
        fallback,
        threshold=0.8,
    )

    assert measured.measured is True
    assert measured.fallback_case_count == 1
    assert measured.fallback_rate == 0.5
    assert measured.accuracy == 1.0
    assert measured.cases[0].case_id == "low"
    assert measured.cases[0].fast_predicted == "prometheus.query"
    assert measured.cases[0].fallback_predicted == "logs.search"


def test_no_low_confidence_cases_is_still_a_measured_run():
    fast = MappingScoreAdapter(
        lambda request: {
            "prometheus.query": 0.99,
            "logs.search": 0.01,
        }
    )
    report = run_benchmark(fast, CASES[:1])
    fallback = MappingScoreAdapter(
        lambda request: {
            "prometheus.query": 0.5,
            "logs.search": 0.5,
        }
    )

    measured = evaluate_system2_fallback(
        report,
        CASES[:1],
        fallback,
        threshold=0.8,
    )

    assert measured.measured is True
    assert measured.fallback_case_count == 0
    assert measured.accuracy is None
