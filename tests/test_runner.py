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
    assert report.macro_f1 == 1.0
    assert report.nll > 0.0
    assert len(report.cases) == 2


def test_benchmark_runner_selects_risk_budget_operating_point() -> None:
    def mapping(request):
        if request.context["intent"] == "metrics":
            return {"a": 0.95, "b": 0.05}
        return {"a": 0.55, "b": 0.45}

    report = run_benchmark(
        MappingScoreAdapter(mapping),
        CASES,
        thresholds=[0.5, 0.8, 0.9],
        risk_budget=0.0,
    )

    assert report.risk_budget == 0.0
    assert report.operating_point is not None
    assert report.operating_point.threshold == 0.9
    assert report.operating_point.coverage == 0.5
    assert report.operating_point.fallback_rate == 0.5
    assert report.operating_point.false_automation_rate == 0.0


def test_structured_output_adapter() -> None:
    def generate(request):
        if request.context["intent"] == "metrics":
            return '{"candidate":"a","confidence":0.8}'
        return '{"candidate":"b","confidence":0.9}'

    report = run_benchmark(StructuredOutputAdapter(generate), CASES)
    assert report.accuracy == 1.0



def test_benchmark_runner_rejects_no_gold_abstention_cases() -> None:
    case = BenchmarkCase(
        case_id="abstain-1",
        decision_type="router",
        context={"intent": "ambiguous"},
        candidates=["a", "b"],
        gold_candidate=None,
        metadata={"expected_abstain": True},
    )

    try:
        run_benchmark(
            MappingScoreAdapter(
                lambda request: {"a": 0.5, "b": 0.5}
            ),
            [case],
        )
    except ValueError as exc:
        assert "requires single-gold cases" in str(exc)
    else:
        raise AssertionError("closed-set benchmark must reject no-gold cases")
