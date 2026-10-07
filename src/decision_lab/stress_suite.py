from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from typing import Callable, Iterable, Literal, Sequence

from .benchmark import BenchmarkCase
from .inventory import ToolInventory, ToolSpec, inventory_digest


DistractorStrategy = Literal["random", "lexical"]
DistractorRanker = Callable[
    [BenchmarkCase, Sequence[ToolSpec]],
    Sequence[ToolSpec],
]


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_DEFAULT_COUNTS = (5, 10, 20, 50, 100)


def build_candidate_count_suite(
    cases: Sequence[BenchmarkCase],
    inventory: ToolInventory,
    *,
    candidate_counts: Iterable[int] = _DEFAULT_COUNTS,
    strategy: DistractorStrategy = "random",
    seed: str = "agent-decision-benchmark-v0.2",
    hard_negative_ranker: DistractorRanker | None = None,
) -> list[BenchmarkCase]:
    if not cases:
        raise ValueError("cases must not be empty")

    counts = _validate_counts(candidate_counts, len(inventory.tools))
    tool_by_name = {tool.name: tool for tool in inventory.tools}
    digest = inventory_digest(inventory)
    generated: list[BenchmarkCase] = []

    for case in cases:
        if case.gold_candidate not in tool_by_name:
            raise ValueError(
                f"{case.case_id} gold candidate {case.gold_candidate!r} "
                "is absent from the inventory"
            )

        distractors = [
            tool
            for tool in inventory.tools
            if tool.name != case.gold_candidate
        ]

        for count in counts:
            selected_distractors = _select_distractors(
                case,
                distractors,
                count=count - 1,
                strategy=strategy,
                seed=seed,
                hard_negative_ranker=hard_negative_ranker,
            )
            selected = [case.gold_candidate] + [
                tool.name for tool in selected_distractors
            ]
            candidates = _stable_candidate_order(
                selected,
                case_id=case.case_id,
                count=count,
                seed=seed,
                variant=f"covered:{strategy}",
            )

            generated.append(
                replace(
                    case,
                    case_id=f"{case.case_id}::k{count}::{strategy}",
                    candidates=candidates,
                    metadata={
                        **case.metadata,
                        "parent_case_id": case.case_id,
                        "inventory_id": inventory.inventory_id,
                        "inventory_sha256": digest,
                        "candidate_count": count,
                        "distractor_strategy": strategy,
                        "generation_seed": seed,
                        "expected_abstain": False,
                        "shift": "candidate_count",
                    },
                )
            )

    return generated


def build_missing_candidate_suite(
    cases: Sequence[BenchmarkCase],
    inventory: ToolInventory,
    *,
    candidate_counts: Iterable[int] = _DEFAULT_COUNTS,
    strategy: DistractorStrategy = "random",
    seed: str = "agent-decision-benchmark-v0.2",
    hard_negative_ranker: DistractorRanker | None = None,
) -> list[BenchmarkCase]:
    if not cases:
        raise ValueError("cases must not be empty")

    counts = _validate_counts(
        candidate_counts,
        len(inventory.tools) - 1,
        minimum=1,
    )
    tool_names = {tool.name for tool in inventory.tools}
    digest = inventory_digest(inventory)
    generated: list[BenchmarkCase] = []

    for case in cases:
        if case.gold_candidate not in tool_names:
            raise ValueError(
                f"{case.case_id} gold candidate {case.gold_candidate!r} "
                "is absent from the inventory"
            )

        distractors = [
            tool
            for tool in inventory.tools
            if tool.name != case.gold_candidate
        ]

        for count in counts:
            selected_distractors = _select_distractors(
                case,
                distractors,
                count=count,
                strategy=strategy,
                seed=seed,
                hard_negative_ranker=hard_negative_ranker,
            )
            candidates = _stable_candidate_order(
                [tool.name for tool in selected_distractors],
                case_id=case.case_id,
                count=count,
                seed=seed,
                variant=f"missing:{strategy}",
            )

            generated.append(
                replace(
                    case,
                    case_id=f"{case.case_id}::missing::k{count}::{strategy}",
                    candidates=candidates,
                    metadata={
                        **case.metadata,
                        "parent_case_id": case.case_id,
                        "inventory_id": inventory.inventory_id,
                        "inventory_sha256": digest,
                        "candidate_count": count,
                        "distractor_strategy": strategy,
                        "generation_seed": seed,
                        "expected_abstain": True,
                        "shift": "missing_candidate",
                    },
                )
            )

    return generated


def lexical_hard_negative_ranker(
    case: BenchmarkCase,
    tools: Sequence[ToolSpec],
) -> Sequence[ToolSpec]:
    context_tokens = _tokens(
        json.dumps(
            case.context,
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    scored = []
    for tool in tools:
        tool_tokens = _tokens(
            f"{tool.name} {tool.description} {tool.server}"
        )
        union = context_tokens | tool_tokens
        similarity = (
            len(context_tokens & tool_tokens) / len(union)
            if union
            else 0.0
        )
        scored.append((similarity, tool.name, tool))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored]


def group_by_candidate_count(
    cases: Sequence[BenchmarkCase],
) -> dict[int, list[BenchmarkCase]]:
    grouped: dict[int, list[BenchmarkCase]] = {}
    for case in cases:
        count = int(
            case.metadata.get(
                "candidate_count",
                len(case.candidates),
            )
        )
        grouped.setdefault(count, []).append(case)
    return grouped


def _select_distractors(
    case: BenchmarkCase,
    distractors: Sequence[ToolSpec],
    *,
    count: int,
    strategy: DistractorStrategy,
    seed: str,
    hard_negative_ranker: DistractorRanker | None,
) -> list[ToolSpec]:
    if count > len(distractors):
        raise ValueError(
            f"{case.case_id} needs {count} distractors but only "
            f"{len(distractors)} are available"
        )

    if hard_negative_ranker is not None:
        ranked = list(hard_negative_ranker(case, distractors))
        _validate_ranked_tools(distractors, ranked)
        return ranked[:count]

    if strategy == "lexical":
        return list(
            lexical_hard_negative_ranker(case, distractors)
        )[:count]

    if strategy != "random":
        raise ValueError(f"unsupported distractor strategy: {strategy}")

    return sorted(
        distractors,
        key=lambda tool: _stable_key(
            seed,
            case.case_id,
            "distractor",
            tool.name,
        ),
    )[:count]


def _stable_candidate_order(
    candidates: Sequence[str],
    *,
    case_id: str,
    count: int,
    seed: str,
    variant: str,
) -> list[str]:
    return sorted(
        candidates,
        key=lambda candidate: _stable_key(
            seed,
            case_id,
            str(count),
            variant,
            candidate,
        ),
    )


def _stable_key(*parts: str) -> str:
    return hashlib.sha256(
        "\x1f".join(parts).encode("utf-8")
    ).hexdigest()


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN_RE.findall(text)
        if token
    }


def _validate_counts(
    counts: Iterable[int],
    maximum: int,
    *,
    minimum: int = 2,
) -> tuple[int, ...]:
    normalized = tuple(sorted(set(int(value) for value in counts)))
    if not normalized:
        raise ValueError("candidate_counts must not be empty")
    if normalized[0] < minimum:
        raise ValueError(
            f"candidate counts must be >= {minimum}"
        )
    if normalized[-1] > maximum:
        raise ValueError(
            f"candidate count {normalized[-1]} exceeds inventory capacity "
            f"{maximum}"
        )
    return normalized


def _validate_ranked_tools(
    source: Sequence[ToolSpec],
    ranked: Sequence[ToolSpec],
) -> None:
    source_names = {tool.name for tool in source}
    ranked_names = [tool.name for tool in ranked]
    if len(ranked_names) != len(set(ranked_names)):
        raise ValueError("hard-negative ranker returned duplicate tools")
    if set(ranked_names) != source_names:
        raise ValueError(
            "hard-negative ranker must return every distractor exactly once"
        )
