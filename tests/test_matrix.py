from decision_lab.matrix import benchmark_matrix_dict
from decision_lab.runner import BenchmarkReport


def report(adapter: str, accuracy: float) -> BenchmarkReport:
    return BenchmarkReport(
        adapter=adapter,
        accuracy=accuracy,
        brier=0.1,
        ece=0.02,
        mean_latency_ms=10.0,
        p50_latency_ms=9.0,
        p95_latency_ms=15.0,
        mean_tokens_processed_per_decision=32.0,
        coverage=[],
        cases=[],
    )


def test_benchmark_matrix_is_stable_and_comparable() -> None:
    matrix = benchmark_matrix_dict(
        {
            "E-structured": report("structured-output", 0.91),
            "A-logits": report("candidate-logits", 0.94),
        }
    )

    assert matrix["schema_version"] == "v1"
    assert [row["arm"] for row in matrix["arms"]] == [
        "A-logits",
        "E-structured",
    ]
    assert matrix["arms"][0]["accuracy"] == 0.94


def test_empty_matrix_fails_closed() -> None:
    try:
        benchmark_matrix_dict({})
    except ValueError as exc:
        assert str(exc) == "reports must not be empty"
    else:
        raise AssertionError("empty benchmark matrix must fail")
