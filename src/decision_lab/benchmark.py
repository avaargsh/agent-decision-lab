from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    decision_type: str
    context: dict[str, Any]
    candidates: list[str]
    gold_candidate: str
    metadata: dict[str, Any]


def load_jsonl(path: str | Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}") from exc

            cases.append(
                BenchmarkCase(
                    case_id=obj["case_id"],
                    decision_type=obj["decision_type"],
                    context=dict(obj.get("context", {})),
                    candidates=list(obj["candidates"]),
                    gold_candidate=obj["gold_candidate"],
                    metadata=dict(obj.get("metadata", {})),
                )
            )

    return cases


def dump_jsonl(cases: Iterable[BenchmarkCase], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(
                json.dumps(
                    {
                        "case_id": case.case_id,
                        "decision_type": case.decision_type,
                        "context": case.context,
                        "candidates": case.candidates,
                        "gold_candidate": case.gold_candidate,
                        "metadata": case.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
