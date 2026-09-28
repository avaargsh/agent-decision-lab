from decision_lab.adapters import MappingScoreAdapter, StructuredOutputAdapter
from decision_lab.benchmark import BenchmarkCase
from decision_lab.runner import run_benchmark


CASES = [
    BenchmarkCase(
        case_id="1",
        decision_type="router",
        context={"intent": "metrics"},
        candidates=["a", "b"],
        gold_candidate="a",
        metadata={},
    ),
    BenchmarkCase(
        case_id="2",
        decision_type="router",
        context={"intent": "logs"},
        candidates=["a", "b"],
        gold_candidate="b",
        metadata={},
    ),
]


def test_benchmark_runner_perfect_mapping() -> None:
    def mapping(request):
        if request.context["intent"] == "metrics":
            return {"a": 0.9, "b": 0.1}
        return {"a": 0.2, "b": 0.8}

    report = run_benchmark(MappingScoreAdapter(mapping), CASES)
    assert report.accuracy == 1.0
    assert len(report.cases) == 2


def test_structured_output_adapter() -> None:
    def generate(request):
        if request.context["intent"] == "metrics":
            return '{"candidate":"a","confidence":0.8}'
        return '{"candidate":"b","confidence":0.9}'

    report = run_benchmark(StructuredOutputAdapter(generate), CASES)
    assert report.accuracy == 1.0
