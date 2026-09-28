from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Sequence

from .models import CandidateScore, DecisionRequest
from .runner import DecisionAdapter


def _request_key(
    request: DecisionRequest,
) -> str:
    return json.dumps(
        {
            "decision_type": request.decision_type,
            "candidates": list(request.candidates),
            "context": request.context,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


@dataclass
class CachingDecisionAdapter:
    """Cache model scores while preserving original latency/usage metadata."""

    base: DecisionAdapter
    _cache: dict[
        str,
        tuple[
            tuple[CandidateScore, ...],
            float,
            int | None,
        ],
    ] = field(
        default_factory=dict,
        init=False,
    )

    @property
    def name(self) -> str:
        return self.base.name

    def __post_init__(self) -> None:
        self.last_latency_ms: float | None = None
        self.last_tokens_processed: int | None = None

    def score(
        self,
        request: DecisionRequest,
    ) -> Sequence[CandidateScore]:
        key = _request_key(request)

        cached = self._cache.get(key)
        if cached is not None:
            scores, latency_ms, tokens = cached
            self.last_latency_ms = latency_ms
            self.last_tokens_processed = tokens
            return list(scores)

        started = time.perf_counter()
        scores = tuple(
            self.base.score(request)
        )
        latency_ms = (
            time.perf_counter() - started
        ) * 1000.0
        tokens = getattr(
            self.base,
            "last_tokens_processed",
            None,
        )
        tokens = (
            int(tokens)
            if tokens is not None
            else None
        )

        self._cache[key] = (
            scores,
            latency_ms,
            tokens,
        )
        self.last_latency_ms = latency_ms
        self.last_tokens_processed = tokens
        return list(scores)
