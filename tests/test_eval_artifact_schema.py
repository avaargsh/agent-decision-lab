import json
from pathlib import Path


SCHEMA = (
    Path(__file__).parents[1]
    / "schemas"
    / "decision-eval-artifact.schema.json"
)


def test_decision_eval_schema_carries_verifier_numeric_bounds():
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))

    metrics = schema["properties"]["metrics"]["properties"]
    for field in ("accuracy", "macro_f1", "brier", "ece"):
        assert metrics[field]["minimum"] == 0
        assert metrics[field]["maximum"] == 1
    for field in (
        "nll",
        "mean_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "mean_tokens_processed_per_decision",
    ):
        assert metrics[field]["minimum"] == 0

    operating = schema["properties"]["operating_point"]["properties"]
    assert operating["coverage"]["exclusiveMinimum"] == 0
    for field in (
        "threshold",
        "coverage",
        "risk",
        "false_automation_rate",
        "fallback_rate",
        "risk_budget",
    ):
        assert operating[field]["maximum"] == 1
    assert operating["risk_budget"]["type"] == "number"
