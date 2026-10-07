from __future__ import annotations

from decision_lab.benchmark import BenchmarkCase
from decision_lab.dataset_validation import (
    validate_calibration_test_pair,
)
from decision_lab.inventory import ToolSpec, make_inventory


def _case(case_id: str, split: str, intent: str) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=case_id,
        decision_type="mcp_tool_router",
        context={"intent": intent},
        candidates=["server::a", "server::b"],
        gold_candidate="server::a",
        metadata={"split": split},
    )


def _inventory():
    return make_inventory(
        inventory_id="fixture",
        tools=[
            ToolSpec("server::a", "A", "server"),
            ToolSpec("server::b", "B", "server"),
        ],
        source={"kind": "fixture"},
    )


def test_split_validation_accepts_disjoint_calibration_and_test() -> None:
    report = validate_calibration_test_pair(
        [_case("c1", "calibration", "cal intent")],
        [_case("t1", "test", "test intent")],
        inventory=_inventory(),
    )

    assert report.calibration_count == 1
    assert report.test_count == 1


def test_split_validation_rejects_context_leakage() -> None:
    try:
        validate_calibration_test_pair(
            [_case("c1", "calibration", "same intent")],
            [_case("t1", "test", "same intent")],
        )
    except ValueError as exc:
        assert "context leakage" in str(exc)
    else:
        raise AssertionError("split leakage must fail")


def test_split_validation_rejects_wrong_split_label() -> None:
    try:
        validate_calibration_test_pair(
            [_case("c1", "test", "cal intent")],
            [_case("t1", "test", "test intent")],
        )
    except ValueError as exc:
        assert "wrong split" in str(exc)
    else:
        raise AssertionError("wrong split metadata must fail")


def test_split_validation_rejects_unknown_inventory_identity() -> None:
    bad = BenchmarkCase(
        case_id="t1",
        decision_type="mcp_tool_router",
        context={"intent": "test intent"},
        candidates=["server::a", "unknown::tool"],
        gold_candidate="server::a",
        metadata={"split": "test"},
    )
    try:
        validate_calibration_test_pair(
            [_case("c1", "calibration", "cal intent")],
            [bad],
            inventory=_inventory(),
        )
    except ValueError as exc:
        assert "absent from inventory" in str(exc)
    else:
        raise AssertionError("unknown tool identity must fail")
