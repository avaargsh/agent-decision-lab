from __future__ import annotations

import json

from decision_lab.benchmark import BenchmarkCase
from decision_lab.inventory import (
    ToolSpec,
    inventory_digest,
    make_inventory,
)
from decision_lab.semantic_embeddings import (
    ToolEmbeddingSnapshot,
    embedding_digest,
    load_embedding_snapshot,
    semantic_gold_neighbor_ranker,
    tool_embedding_text,
    validate_embedding_snapshot,
    write_embedding_snapshot,
)


def _inventory():
    return make_inventory(
        inventory_id="fixture-v1",
        tools=[
            ToolSpec(
                "server::gold",
                "query service latency metrics",
                "server",
                {
                    "upstream_tool_name": "gold",
                    "source_path": "src/server/gold.py",
                },
            ),
            ToolSpec(
                "server::near",
                "query request latency metrics",
                "server",
            ),
            ToolSpec(
                "server::mid",
                "search application logs",
                "server",
            ),
            ToolSpec(
                "server::far",
                "restart deployment",
                "server",
            ),
        ],
        source={"kind": "fixture"},
    )


def _snapshot():
    inventory = _inventory()
    return ToolEmbeddingSnapshot(
        embedding_id="fixture-embeddings",
        inventory_id=inventory.inventory_id,
        inventory_sha256=inventory_digest(inventory),
        model={
            "provider": "fixture",
            "model": "fixture-model",
            "revision": "abc123",
        },
        vectors={
            "server::gold": (1.0, 0.0),
            "server::near": (0.9, 0.1),
            "server::mid": (0.4, 0.6),
            "server::far": (-1.0, 0.0),
        },
    )


def test_embedding_snapshot_round_trip_and_digest(tmp_path) -> None:
    snapshot = _snapshot()
    path = tmp_path / "embeddings.json"

    write_embedding_snapshot(snapshot, path)
    loaded = load_embedding_snapshot(path)

    assert loaded == snapshot
    assert embedding_digest(loaded) == embedding_digest(snapshot)


def test_embedding_snapshot_rejects_tampering(tmp_path) -> None:
    snapshot = _snapshot()
    path = tmp_path / "embeddings.json"
    write_embedding_snapshot(snapshot, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["vectors"][0]["vector"][0] = 0.5
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    try:
        load_embedding_snapshot(path)
    except ValueError as exc:
        assert "sha256 mismatch" in str(exc)
    else:
        raise AssertionError("tampered embedding snapshot must fail")


def test_snapshot_must_cover_exact_inventory() -> None:
    snapshot = _snapshot()
    inventory = _inventory()
    broken = ToolEmbeddingSnapshot(
        embedding_id=snapshot.embedding_id,
        inventory_id=snapshot.inventory_id,
        inventory_sha256=snapshot.inventory_sha256,
        model=snapshot.model,
        vectors={
            name: vector
            for name, vector in snapshot.vectors.items()
            if name != "server::far"
        },
    )

    try:
        validate_embedding_snapshot(
            broken,
            inventory,
        )
    except ValueError as exc:
        assert "coverage mismatch" in str(exc)
    else:
        raise AssertionError("partial embedding coverage must fail")


def test_semantic_ranker_uses_gold_tool_neighbours() -> None:
    inventory = _inventory()
    snapshot = _snapshot()
    ranker = semantic_gold_neighbor_ranker(
        snapshot,
        inventory,
    )
    case = BenchmarkCase(
        case_id="c1",
        decision_type="mcp_tool_router",
        context={"intent": "check latency"},
        candidates=["server::gold", "server::near"],
        gold_candidate="server::gold",
        metadata={"split": "test"},
    )
    distractors = [
        tool
        for tool in inventory.tools
        if tool.name != "server::gold"
    ]

    ranked = ranker(
        case,
        distractors,
    )

    assert [tool.name for tool in ranked] == [
        "server::near",
        "server::mid",
        "server::far",
    ]


def test_embedding_text_keeps_identity_when_description_missing() -> None:
    tool = ToolSpec(
        "server::tool",
        "",
        "server",
        {
            "upstream_tool_name": "tool",
            "source_path": "src/server/tool.py",
        },
    )

    text = tool_embedding_text(tool)

    assert "canonical_name: server::tool" in text
    assert "server: server" in text
    assert "upstream_tool_name: tool" in text
    assert "source_path: src/server/tool.py" in text
