from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from typing import Any, Callable

from .gateway import DecisionGateway
from .ledger import decision_ledger_entry
from .models import DecisionRequest
from .telemetry import set_attribute, span


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

        with span(
            "decision",
            {
                "decision.type": request.decision_type,
                "decision.candidate_count": len(request.candidates),
            },
        ) as current:
            result = self.gateway.decide(request)

            set_attribute(current, "decision.action", result.action)
            set_attribute(current, "decision.reason_code", result.reason_code)
            set_attribute(
                current,
                "decision.fallback",
                result.action == "FALLBACK",
            )
            if result.decision is not None:
                set_attribute(
                    current,
                    "decision.confidence",
                    result.decision.confidence,
                )
                set_attribute(
                    current,
                    "decision.selected_candidate",
                    result.decision.candidate,
                )

            identity_payload = {
                "decision_type": request.decision_type,
                "candidates": list(request.candidates),
                "context": dict(request.context),
                "action": result.action,
                "reason_code": result.reason_code,
                "scores": [asdict(item) for item in result.scores],
                "decision": asdict(result.decision) if result.decision is not None else None,
            }
            canonical = json.dumps(identity_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            decision_id = "decision-sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()

            response: dict[str, Any] = {
                "decision_id": decision_id,
                "action": result.action,
                "reason_code": result.reason_code,
                "scores": [asdict(item) for item in result.scores],
                "decision": (
                    asdict(result.decision)
                    if result.decision is not None
                    else None
                ),
                "ledger": {**decision_ledger_entry(result), "decision_id": decision_id},
            }

            if result.action == "FALLBACK" and self.fallback is not None:
                response["fallback"] = self.fallback(request)

            return response
