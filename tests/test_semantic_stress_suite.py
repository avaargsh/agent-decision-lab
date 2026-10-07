from __future__ import annotations

from decision_lab.benchmark import BenchmarkCase
from decision_lab.inventory import (
    ToolSpec,
    inventory_digest,
    make_inventory,
)
from decision_lab.semantic_embeddings import (
    ToolEmbeddingSnapshot,
    semantic_gold_neighbor_ranker,
)
from decision_lab.stress_suite import (
    build_candidate_count_suite,
)


def _inventory():
    return make_inventory(
        inventory_id="fixture",
        tools=[
            ToolSpec("s::gold", "gold", "s"),
            ToolSpec("s::near", "near", "s"),
            ToolSpec("s::mid", "mid", "s"),
            ToolSpec("s::far", "far", "s"),
        ],
        source={"kind": "fixture"},
    )


def _case():
    return BenchmarkCase(
        case_id="case-1",
        decision_type="mcp_tool_router",
        context={"intent": "fixture"},
        candidates=["s::gold", "s::near"],
        gold_candidate="s::gold",
        metadata={"split": "test"},
    )


def test_semantic_strategy_uses_explicit_ranker_and_metadata() -> None:
    inventory = _inventory()
    snapshot = ToolEmbeddingSnapshot(
        embedding_id="fixture-emb",
        inventory_id=inventory.inventory_id,
        inventory_sha256=inventory_digest(inventory),
        model={"model": "fixture"},
        vectors={
            "s::gold": (1.0, 0.0),
            "s::near": (0.9, 0.1),
            "s::mid": (0.5, 0.5),
            "s::far": (-1.0, 0.0),
        },
    )
    ranker = semantic_gold_neighbor_ranker(
        snapshot,
        inventory,
    )

    generated = build_candidate_count_suite(
        [_case()],
        inventory,
        candidate_counts=[3],
        strategy="semantic",
        hard_negative_ranker=ranker,
        seed="fixed",
    )

    case = generated[0]
    assert set(case.candidates) == {
        "s::gold",
        "s::near",
        "s::mid",
    }
    assert case.metadata["distractor_strategy"] == "semantic"


def test_semantic_strategy_fails_without_ranker() -> None:
    try:
        build_candidate_count_suite(
            [_case()],
            _inventory(),
            candidate_counts=[3],
            strategy="semantic",
        )
    except ValueError as exc:
        assert "requires an explicit hard-negative ranker" in str(exc)
    else:
        raise AssertionError("semantic strategy must fail without ranker")
