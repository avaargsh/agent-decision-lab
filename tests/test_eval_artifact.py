from decision_lab.adapters import MappingScoreAdapter
from decision_lab.benchmark import BenchmarkCase
from decision_lab.eval_artifact import (
    build_eval_artifact,
    verify_eval_artifact,
)
from decision_lab.runner import run_benchmark


CASES = [
    BenchmarkCase(
        case_id="route-metrics",
        decision_type="mcp_tool_router",
        context={"intent": "metrics"},
        candidates=["prometheus.query", "logs.search"],
        gold_candidate="prometheus.query",
        metadata={},
    ),
    BenchmarkCase(
        case_id="route-logs",
        decision_type="mcp_tool_router",
        context={"intent": "logs"},
        candidates=["prometheus.query", "logs.search"],
        gold_candidate="logs.search",
        metadata={},
    ),
]


def report():
    def mapping(request):
        if request.context["intent"] == "metrics":
            return {"prometheus.query": 0.95, "logs.search": 0.05}
        return {"prometheus.query": 0.55, "logs.search": 0.45}

    return run_benchmark(
        MappingScoreAdapter(mapping),
        CASES,
        thresholds=[0.5, 0.8, 0.9],
        risk_budget=0.0,
    )


def dataset():
    return {
        "sha256": "sha256:" + "a" * 64,
        "case_count": 2,
        "case_ids": ["route-metrics", "route-logs"],
    }


def test_eval_artifact_is_content_addressed_and_verifiable():
    artifact = build_eval_artifact(
        report(),
        decision_type="mcp_tool_router",
        dataset=dataset(),
        model_ref="qwen/test-scorer",
        calibration_sha256="sha256:" + "b" * 64,
    )

    assert artifact["schema_version"] == "decision-eval/v1"
    assert artifact["artifact_id"].startswith("decision-eval:sha256:")
    assert artifact["fallback_evaluation"]["measured"] is False
    assert artifact["operating_point"]["fallback_rate"] == 0.5
    assert verify_eval_artifact(artifact)


def test_eval_artifact_digest_detects_metric_tampering():
    artifact = build_eval_artifact(
        report(),
        decision_type="mcp_tool_router",
        dataset=dataset(),
    )
    artifact["metrics"]["accuracy"] = 0.0
    assert not verify_eval_artifact(artifact)


def test_eval_artifact_rejects_dataset_case_mismatch():
    bad = dataset()
    bad["case_ids"] = ["wrong-a", "wrong-b"]

    try:
        build_eval_artifact(
            report(),
            decision_type="mcp_tool_router",
            dataset=bad,
        )
    except ValueError as exc:
        assert "report cases" in str(exc)
    else:
        raise AssertionError("mismatched dataset provenance must fail")
