import hashlib
import json

import pytest

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



def test_eval_artifact_seals_measured_fallback_metrics():
    fallback = {
        "measured": True,
        "adapter": "transformers-structured-output",
        "threshold": 0.8,
        "eligible_case_count": 2,
        "fallback_case_count": 1,
        "fallback_rate": 0.5,
        "accuracy": 1.0,
        "p50_latency_ms": 22.0,
        "p95_latency_ms": 22.0,
        "mean_tokens_processed": 48.0,
        "cases": [
            {
                "case_id": "route-logs",
                "fast_confidence": 0.55,
                "fast_predicted": "prometheus.query",
                "fallback_predicted": "logs.search",
                "gold": "logs.search",
                "correct": True,
                "latency_ms": 22.0,
                "tokens_processed": 48,
            }
        ],
    }
    artifact = build_eval_artifact(
        report(),
        decision_type="mcp_tool_router",
        dataset=dataset(),
        fallback_evaluation=fallback,
    )

    assert artifact["fallback_evaluation"]["measured"] is True
    assert artifact["fallback_evaluation"]["fallback_case_count"] == 1
    assert artifact["fallback_evaluation"]["accuracy"] == 1.0
    assert verify_eval_artifact(artifact)

    artifact["fallback_evaluation"]["accuracy"] = 0.0
    assert not verify_eval_artifact(artifact)



def _reseal(artifact):
    payload = {
        key: value
        for key, value in artifact.items()
        if key not in {"artifact_id", "content_digest"}
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    digest = "sha256:" + hashlib.sha256(canonical).hexdigest()
    artifact["content_digest"] = digest
    artifact["artifact_id"] = "decision-eval:" + digest
    return artifact


def measured_fallback():
    return {
        "measured": True,
        "adapter": "transformers-structured-output",
        "threshold": 0.8,
        "eligible_case_count": 2,
        "fallback_case_count": 1,
        "fallback_rate": 0.5,
        "accuracy": 1.0,
        "p50_latency_ms": 22.0,
        "p95_latency_ms": 22.0,
        "mean_tokens_processed": 48.0,
        "parse_valid_rate": 1.0,
        "cases": [
            {
                "case_id": "route-logs",
                "fast_confidence": 0.55,
                "fast_predicted": "prometheus.query",
                "fallback_predicted": "logs.search",
                "gold": "logs.search",
                "correct": True,
                "latency_ms": 22.0,
                "tokens_processed": 48,
                "parse_valid": True,
            }
        ],
    }


def test_build_rejects_inconsistent_fallback_rate():
    fallback = measured_fallback()
    fallback["fallback_rate"] = 0.9

    with pytest.raises(
        ValueError,
        match="fallback_rate does not match",
    ):
        build_eval_artifact(
            report(),
            decision_type="mcp_tool_router",
            dataset=dataset(),
            fallback_evaluation=fallback,
        )


def test_build_rejects_fallback_case_outside_dataset():
    fallback = measured_fallback()
    fallback["cases"][0]["case_id"] = "unknown-case"

    with pytest.raises(
        ValueError,
        match="case_id must belong to dataset",
    ):
        build_eval_artifact(
            report(),
            decision_type="mcp_tool_router",
            dataset=dataset(),
            fallback_evaluation=fallback,
        )


def test_build_rejects_inconsistent_fallback_accuracy():
    fallback = measured_fallback()
    fallback["accuracy"] = 0.0

    with pytest.raises(
        ValueError,
        match="accuracy does not match",
    ):
        build_eval_artifact(
            report(),
            decision_type="mcp_tool_router",
            dataset=dataset(),
            fallback_evaluation=fallback,
        )


def test_verify_rejects_semantic_mismatch_even_when_digest_is_resealed():
    artifact = build_eval_artifact(
        report(),
        decision_type="mcp_tool_router",
        dataset=dataset(),
        fallback_evaluation=measured_fallback(),
    )
    artifact["fallback_evaluation"]["fallback_rate"] = 0.75
    _reseal(artifact)

    assert not verify_eval_artifact(artifact)



def test_build_rejects_fallback_case_at_or_above_threshold():
    fallback = measured_fallback()
    fallback["cases"][0]["fast_confidence"] = 0.8

    with pytest.raises(
        ValueError,
        match="fast_confidence must be below threshold",
    ):
        build_eval_artifact(
            report(),
            decision_type="mcp_tool_router",
            dataset=dataset(),
            fallback_evaluation=fallback,
        )


def test_verify_rejects_resealed_high_confidence_fallback_case():
    artifact = build_eval_artifact(
        report(),
        decision_type="mcp_tool_router",
        dataset=dataset(),
        fallback_evaluation=measured_fallback(),
    )
    artifact["fallback_evaluation"]["cases"][0][
        "fast_confidence"
    ] = 0.95
    _reseal(artifact)

    assert not verify_eval_artifact(artifact)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("coverage", 1.1),
        ("fallback_rate", -0.1),
        ("risk", 1.1),
        ("risk_budget", 1.1),
    ],
)
def test_verify_rejects_resealed_out_of_range_operating_point(
    field,
    value,
):
    artifact = build_eval_artifact(
        report(),
        decision_type="mcp_tool_router",
        dataset=dataset(),
    )
    artifact["operating_point"][field] = value
    _reseal(artifact)

    assert not verify_eval_artifact(artifact)


def test_verify_rejects_resealed_operating_coverage_fallback_mismatch():
    artifact = build_eval_artifact(
        report(),
        decision_type="mcp_tool_router",
        dataset=dataset(),
    )
    artifact["operating_point"]["fallback_rate"] = 0.25
    _reseal(artifact)

    assert not verify_eval_artifact(artifact)


def test_verify_rejects_resealed_operating_false_automation_mismatch():
    artifact = build_eval_artifact(
        report(),
        decision_type="mcp_tool_router",
        dataset=dataset(),
    )
    artifact["operating_point"]["false_automation_rate"] = 0.25
    _reseal(artifact)

    assert not verify_eval_artifact(artifact)


def test_verify_rejects_resealed_operating_risk_over_budget():
    artifact = build_eval_artifact(
        report(),
        decision_type="mcp_tool_router",
        dataset=dataset(),
    )
    artifact["operating_point"]["risk"] = 0.25
    artifact["operating_point"]["false_automation_rate"] = (
        0.25 * artifact["operating_point"]["coverage"]
    )
    _reseal(artifact)

    assert not verify_eval_artifact(artifact)
