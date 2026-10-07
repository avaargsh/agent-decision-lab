from __future__ import annotations

import json

from decision_lab.benchmark import BenchmarkCase
from decision_lab.inventory import (
    ToolSpec,
    inventory_digest,
    load_inventory,
    make_inventory,
    write_inventory,
)
from decision_lab.stress_suite import (
    build_candidate_count_suite,
    build_missing_candidate_suite,
    group_by_candidate_count,
    lexical_hard_negative_ranker,
)


def _inventory(size: int = 12):
    tools = [
        ToolSpec(
            name=f"server.tool_{index:03d}",
            description=f"tool {index} metrics logs deployment runbook",
            server="server",
        )
        for index in range(size)
    ]
    tools[0] = ToolSpec(
        name="prometheus.query",
        description="query metrics time series latency cpu error rate",
        server="prometheus",
    )
    tools[1] = ToolSpec(
        name="logs.search",
        description="search application logs stack traces error messages",
        server="logs",
    )
    return make_inventory(
        inventory_id="fixture-v1",
        tools=tools,
        source={
            "kind": "test-fixture",
            "revision": "fixture-v1",
        },
    )


def _case() -> BenchmarkCase:
    return BenchmarkCase(
        case_id="mcp-test-001",
        decision_type="mcp_tool_router",
        context={
            "intent": "check API latency metrics for checkout"
        },
        candidates=[
            "prometheus.query",
            "logs.search",
        ],
        gold_candidate="prometheus.query",
        metadata={
            "split": "test",
            "source": "fixture",
        },
    )


def test_inventory_round_trip_verifies_declared_digest(tmp_path) -> None:
    inventory = _inventory()
    path = tmp_path / "inventory.json"

    write_inventory(inventory, path)
    loaded = load_inventory(path)

    assert loaded == inventory
    assert inventory_digest(loaded) == inventory_digest(inventory)


def test_inventory_load_rejects_digest_mismatch(tmp_path) -> None:
    inventory = _inventory()
    path = tmp_path / "inventory.json"
    write_inventory(inventory, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tools"][0]["description"] = "tampered"
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    try:
        load_inventory(path)
    except ValueError as exc:
        assert "sha256 mismatch" in str(exc)
    else:
        raise AssertionError("tampered inventory must fail")


def test_candidate_count_suite_is_deterministic_and_preserves_gold() -> None:
    inventory = _inventory()
    first = build_candidate_count_suite(
        [_case()],
        inventory,
        candidate_counts=[5, 10],
        seed="fixed",
    )
    second = build_candidate_count_suite(
        [_case()],
        inventory,
        candidate_counts=[10, 5],
        seed="fixed",
    )

    assert first == second
    assert [len(case.candidates) for case in first] == [5, 10]
    assert all(
        case.gold_candidate in case.candidates
        for case in first
    )
    assert all(
        case.metadata["expected_abstain"] is False
        for case in first
    )
    assert all(
        case.metadata["inventory_sha256"]
        == inventory_digest(inventory)
        for case in first
    )


def test_candidate_count_suite_groups_by_k() -> None:
    generated = build_candidate_count_suite(
        [_case()],
        _inventory(),
        candidate_counts=[5, 10],
    )

    grouped = group_by_candidate_count(generated)

    assert sorted(grouped) == [5, 10]
    assert len(grouped[5][0].candidates) == 5
    assert len(grouped[10][0].candidates) == 10


def test_missing_candidate_suite_removes_gold_and_marks_abstention() -> None:
    generated = build_missing_candidate_suite(
        [_case()],
        _inventory(),
        candidate_counts=[5, 10],
        seed="fixed",
    )

    assert all(
        case.gold_candidate not in case.candidates
        for case in generated
    )
    assert all(
        case.metadata["expected_abstain"] is True
        for case in generated
    )
    assert all(
        case.metadata["shift"] == "missing_candidate"
        for case in generated
    )


def test_lexical_ranker_prioritizes_context_related_tools() -> None:
    case = _case()
    tools = [
        ToolSpec(
            "runbook.search",
            "search operational runbooks procedures",
            "runbook",
        ),
        ToolSpec(
            "metrics.latency",
            "query API latency metrics",
            "prometheus",
        ),
        ToolSpec(
            "logs.search",
            "search logs and tracebacks",
            "logs",
        ),
    ]

    ranked = lexical_hard_negative_ranker(case, tools)

    assert ranked[0].name == "metrics.latency"


def test_custom_ranker_contract_rejects_partial_results() -> None:
    def invalid_ranker(case, tools):
        return tools[:1]

    try:
        build_candidate_count_suite(
            [_case()],
            _inventory(),
            candidate_counts=[5],
            hard_negative_ranker=invalid_ranker,
        )
    except ValueError as exc:
        assert "every distractor exactly once" in str(exc)
    else:
        raise AssertionError("invalid ranker must fail")


def test_candidate_count_cannot_exceed_inventory_capacity() -> None:
    try:
        build_candidate_count_suite(
            [_case()],
            _inventory(size=6),
            candidate_counts=[10],
        )
    except ValueError as exc:
        assert "exceeds inventory capacity" in str(exc)
    else:
        raise AssertionError("oversized candidate count must fail")
