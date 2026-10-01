import copy

import pytest

from decision_lab.adapters import MappingScoreAdapter
from decision_lab.benchmark import BenchmarkCase
from decision_lab.contract_matrix import (
    ContractMatrixError,
    build_contract_matrix,
)
from decision_lab.eval_artifact import build_eval_artifact
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


def _artifact(
    *,
    adapter_name: str,
    model_ref: str,
    dataset_digest: str | None = None,
    decision_type: str = "mcp_tool_router",
):
    def mapping(request):
        if request.context["intent"] == "metrics":
            return {
                "prometheus.query": 0.95,
                "logs.search": 0.05,
            }
        return {
            "prometheus.query": 0.40,
            "logs.search": 0.60,
        }

    report = run_benchmark(
        MappingScoreAdapter(mapping, name=adapter_name),
        CASES,
        thresholds=[0.5, 0.8],
        risk_budget=0.0,
    )
    return build_eval_artifact(
        report,
        decision_type=decision_type,
        dataset={
            "sha256": dataset_digest or ("sha256:" + ("a" * 64)),
            "case_count": 2,
            "case_ids": ["route-metrics", "route-logs"],
        },
        model_ref=model_ref,
        calibration={
            "sha256": "sha256:" + ("b" * 64),
            "case_count": 2,
            "case_ids": ["cal-a", "cal-b"],
        },
    )


def test_contract_matrix_compares_multiple_adapters_on_exact_test_dataset():
    fast = _artifact(
        adapter_name="frozen-candidate-logits",
        model_ref="Qwen/Qwen3-0.6B",
    )
    structured = _artifact(
        adapter_name="structured-output",
        model_ref="Qwen/Qwen3-0.6B",
    )

    matrix = build_contract_matrix([fast, structured])

    assert matrix["schema_version"] == "decision-contract-matrix/v1"
    assert matrix["decision_eval_schema_version"] == "decision-eval/v1"
    assert matrix["decision_type"] == "mcp_tool_router"
    assert matrix["dataset"]["case_ids"] == [
        "route-metrics",
        "route-logs",
    ]
    assert [row["adapter"] for row in matrix["rows"]] == [
        "frozen-candidate-logits",
        "structured-output",
    ]
    assert all(
        row["calibration_sha256"] == "sha256:" + ("b" * 64)
        for row in matrix["rows"]
    )
    assert all("false_automation_rate" in row for row in matrix["rows"])
    assert all("fallback_measured" in row for row in matrix["rows"])


def test_contract_matrix_rejects_resealed_or_invalid_artifact():
    fast = _artifact(
        adapter_name="frozen-candidate-logits",
        model_ref="Qwen/Qwen3-0.6B",
    )
    invalid = copy.deepcopy(fast)
    invalid["metrics"]["accuracy"] = 0.0

    with pytest.raises(
        ContractMatrixError,
        match="invalid decision eval artifacts",
    ):
        build_contract_matrix([fast, invalid])


def test_contract_matrix_rejects_different_test_dataset_identity():
    fast = _artifact(
        adapter_name="frozen-candidate-logits",
        model_ref="Qwen/Qwen3-0.6B",
    )
    structured = _artifact(
        adapter_name="structured-output",
        model_ref="Qwen/Qwen3-0.6B",
        dataset_digest="sha256:" + ("c" * 64),
    )

    with pytest.raises(
        ContractMatrixError,
        match="exact same test dataset",
    ):
        build_contract_matrix([fast, structured])


def test_contract_matrix_rejects_different_decision_type():
    fast = _artifact(
        adapter_name="frozen-candidate-logits",
        model_ref="Qwen/Qwen3-0.6B",
    )
    structured = _artifact(
        adapter_name="structured-output",
        model_ref="Qwen/Qwen3-0.6B",
        decision_type="policy_gate",
    )

    with pytest.raises(
        ContractMatrixError,
        match="same decision_type",
    ):
        build_contract_matrix([fast, structured])
