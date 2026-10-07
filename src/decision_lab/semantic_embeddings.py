from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .benchmark import BenchmarkCase
from .inventory import ToolInventory, ToolSpec, inventory_digest


SEMANTIC_EMBEDDING_SCHEMA_VERSION = "semantic-tool-embeddings/v1"
TEXT_RECIPE = "tool-identity-description-source-v1"


@dataclass(frozen=True)
class ToolEmbeddingSnapshot:
    embedding_id: str
    inventory_id: str
    inventory_sha256: str
    model: dict[str, Any]
    vectors: dict[str, tuple[float, ...]]
    text_recipe: str = TEXT_RECIPE
    schema_version: str = SEMANTIC_EMBEDDING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.embedding_id:
            raise ValueError("embedding_id must not be empty")
        if self.schema_version != SEMANTIC_EMBEDDING_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported embedding schema: {self.schema_version}"
            )
        if not self.vectors:
            raise ValueError("embedding snapshot must not be empty")

        dimensions = {
            len(vector)
            for vector in self.vectors.values()
        }
        if len(dimensions) != 1:
            raise ValueError(
                "all embedding vectors must have the same dimension"
            )

        dimension = next(iter(dimensions))
        if dimension <= 0:
            raise ValueError("embedding dimension must be > 0")

        for tool_name, vector in self.vectors.items():
            if not tool_name:
                raise ValueError("embedding tool name must not be empty")
            if any(not math.isfinite(value) for value in vector):
                raise ValueError(
                    f"embedding contains non-finite value: {tool_name}"
                )
            if _norm(vector) == 0.0:
                raise ValueError(
                    f"embedding vector must be non-zero: {tool_name}"
                )

    @property
    def dimension(self) -> int:
        return len(next(iter(self.vectors.values())))


def tool_embedding_text(tool: ToolSpec) -> str:
    upstream_name = str(
        tool.metadata.get(
            "upstream_tool_name",
            "",
        )
    )
    source_path = str(
        tool.metadata.get(
            "source_path",
            "",
        )
    )
    parts = [
        f"canonical_name: {tool.name}",
        f"server: {tool.server}",
    ]
    if upstream_name:
        parts.append(
            f"upstream_tool_name: {upstream_name}"
        )
    if tool.description:
        parts.append(
            f"description: {tool.description}"
        )
    if source_path:
        parts.append(
            f"source_path: {source_path}"
        )
    return "\n".join(parts)


def embedding_payload(
    snapshot: ToolEmbeddingSnapshot,
) -> dict[str, Any]:
    return {
        "schema_version": snapshot.schema_version,
        "embedding_id": snapshot.embedding_id,
        "inventory_id": snapshot.inventory_id,
        "inventory_sha256": snapshot.inventory_sha256,
        "text_recipe": snapshot.text_recipe,
        "model": snapshot.model,
        "dimension": snapshot.dimension,
        "vectors": [
            {
                "tool": tool_name,
                "vector": list(snapshot.vectors[tool_name]),
            }
            for tool_name in sorted(snapshot.vectors)
        ],
    }


def embedding_digest(
    snapshot: ToolEmbeddingSnapshot,
) -> str:
    encoded = json.dumps(
        embedding_payload(snapshot),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_embedding_snapshot(
    snapshot: ToolEmbeddingSnapshot,
    path: str | Path,
) -> None:
    payload = embedding_payload(snapshot)
    payload["sha256"] = embedding_digest(snapshot)
    Path(path).write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def load_embedding_snapshot(
    path: str | Path,
) -> ToolEmbeddingSnapshot:
    obj = json.loads(
        Path(path).read_text(encoding="utf-8")
    )
    vectors = {
        item["tool"]: tuple(
            float(value)
            for value in item["vector"]
        )
        for item in obj["vectors"]
    }
    snapshot = ToolEmbeddingSnapshot(
        embedding_id=obj["embedding_id"],
        inventory_id=obj["inventory_id"],
        inventory_sha256=obj["inventory_sha256"],
        text_recipe=obj.get(
            "text_recipe",
            TEXT_RECIPE,
        ),
        model=dict(obj.get("model", {})),
        vectors=vectors,
        schema_version=obj.get(
            "schema_version",
            SEMANTIC_EMBEDDING_SCHEMA_VERSION,
        ),
    )

    declared_dimension = obj.get("dimension")
    if (
        declared_dimension is not None
        and int(declared_dimension)
        != snapshot.dimension
    ):
        raise ValueError(
            "embedding dimension declaration mismatch"
        )

    declared_digest = obj.get("sha256")
    if declared_digest is not None:
        actual = embedding_digest(snapshot)
        if declared_digest != actual:
            raise ValueError(
                "embedding sha256 mismatch: "
                f"declared {declared_digest}, actual {actual}"
            )

    return snapshot


def validate_embedding_snapshot(
    snapshot: ToolEmbeddingSnapshot,
    inventory: ToolInventory,
) -> None:
    expected_digest = inventory_digest(inventory)
    if snapshot.inventory_id != inventory.inventory_id:
        raise ValueError(
            "embedding inventory_id does not match tool inventory"
        )
    if snapshot.inventory_sha256 != expected_digest:
        raise ValueError(
            "embedding inventory sha256 does not match tool inventory"
        )

    expected_tools = {
        tool.name
        for tool in inventory.tools
    }
    embedded_tools = set(snapshot.vectors)
    if embedded_tools != expected_tools:
        missing = sorted(
            expected_tools - embedded_tools
        )
        extra = sorted(
            embedded_tools - expected_tools
        )
        raise ValueError(
            "embedding tool coverage mismatch: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )


def semantic_gold_neighbor_ranker(
    snapshot: ToolEmbeddingSnapshot,
    inventory: ToolInventory,
):
    """Rank distractors by cosine similarity to the gold tool embedding."""
    validate_embedding_snapshot(
        snapshot,
        inventory,
    )

    def rank(
        case: BenchmarkCase,
        tools: Sequence[ToolSpec],
    ) -> Sequence[ToolSpec]:
        if case.gold_candidate not in snapshot.vectors:
            raise ValueError(
                f"gold candidate lacks embedding: {case.gold_candidate}"
            )

        gold_vector = snapshot.vectors[
            case.gold_candidate
        ]
        scored = [
            (
                _cosine(
                    gold_vector,
                    snapshot.vectors[tool.name],
                ),
                tool.name,
                tool,
            )
            for tool in tools
        ]
        scored.sort(
            key=lambda item: (
                -item[0],
                item[1],
            )
        )
        return [
            item[2]
            for item in scored
        ]

    return rank


def _cosine(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right):
        raise ValueError(
            "cosine vectors must have equal dimension"
        )
    denominator = _norm(left) * _norm(right)
    if denominator == 0.0:
        raise ValueError(
            "cosine vectors must be non-zero"
        )
    return sum(
        a * b
        for a, b in zip(
            left,
            right,
            strict=True,
        )
    ) / denominator


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(
        sum(value * value for value in vector)
    )
