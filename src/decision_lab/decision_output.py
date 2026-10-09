"""Adapter-neutral bounded decision values; never an execution authorization."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA_VERSION = "decision-output/v1"
KINDS = {"choice", "boolean", "score"}
FIELDS = {"schema_version", "kind", "value", "abstain", "confidence"}


def validate_decision_output(
    payload: Mapping[str, Any],
    *,
    candidates: Sequence[str] | None = None,
) -> bool:
    """Validate exactly one typed value, including abstention semantics.

    Choice membership can only be checked if the originating candidate set is
    provided. This schema represents a recommendation, never permission to act.
    """
    if not isinstance(payload, Mapping) or set(payload) != FIELDS:
        return False
    if payload.get("schema_version") != SCHEMA_VERSION:
        return False
    kind = payload.get("kind")
    if not isinstance(kind, str) or kind not in KINDS:
        return False
    abstain = payload.get("abstain")
    if type(abstain) is not bool:
        return False
    value = payload.get("value")
    confidence = payload.get("confidence")
    if abstain:
        return value is None and confidence is None
    if type(confidence) not in (int, float):
        return False
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return False
    if kind == "choice":
        if not isinstance(value, str) or not value:
            return False
        if candidates is not None and value not in candidates:
            return False
        return True
    if kind == "boolean":
        return type(value) is bool
    return (
        type(value) in (int, float)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


def build_decision_output(
    *,
    kind: str,
    value: str | bool | float | None,
    confidence: float | None,
    candidates: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a typed recommendation, failing closed on invalid or ambiguous data."""
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "value": value,
        "abstain": value is None,
        "confidence": confidence,
    }
    if kind == "choice" and candidates is None and value is not None:
        raise ValueError("choice output requires originating candidates")
    if not validate_decision_output(result, candidates=candidates):
        raise ValueError("invalid typed decision output")
    return result
