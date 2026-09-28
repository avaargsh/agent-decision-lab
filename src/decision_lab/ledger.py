from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from .models import DecisionResult


def decision_ledger_entry(result: DecisionResult) -> dict[str, Any]:
    """Create an audit-friendly record without storing model chain-of-thought."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decision_type": result.request.decision_type,
        "candidate_set": list(result.request.candidates),
        "scores": [asdict(score) for score in result.scores],
        "action": result.action,
        "reason_code": result.reason_code,
        "selected_candidate": (
            result.decision.candidate if result.decision is not None else None
        ),
        "confidence": (
            result.decision.confidence if result.decision is not None else None
        ),
        "evidence": (
            dict(result.decision.evidence) if result.decision is not None else {}
        ),
    }
