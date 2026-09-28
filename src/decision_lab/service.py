from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Sequence

from .gateway import DecisionGateway
from .ledger import decision_ledger_entry
from .models import CandidateScore, DecisionRequest


FallbackFn = Callable[[DecisionRequest], dict[str, Any]]


class DecisionService:
    """Application-facing decision service with explicit fallback behavior."""

    def __init__(
        self,
        gateway: DecisionGateway,
        *,
        fallback: FallbackFn | None = None,
    ) -> None:
        self.gateway = gateway
        self.fallback = fallback

    def decide(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = DecisionRequest(
            decision_type=str(payload["decision_type"]),
            candidates=list(payload["candidates"]),
            context=dict(payload.get("context", {})),
        )

        result = self.gateway.decide(request)
        response: dict[str, Any] = {
            "action": result.action,
            "reason_code": result.reason_code,
            "scores": [asdict(item) for item in result.scores],
            "decision": asdict(result.decision) if result.decision is not None else None,
            "ledger": decision_ledger_entry(result),
        }

        if result.action == "FALLBACK" and self.fallback is not None:
            response["fallback"] = self.fallback(request)

        return response
