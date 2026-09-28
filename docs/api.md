# Decision API

The first HTTP surface is intentionally small.

## POST /decision

Request:

```json
{
  "decision_type": "mcp_tool_router",
  "candidates": ["read_metrics", "read_logs", "restart_workload"],
  "context": {
    "intent": "metrics"
  }
}
```

Confident response:

```json
{
  "action": "EXECUTE",
  "reason_code": "CONFIDENT",
  "decision": {
    "candidate": "read_metrics",
    "confidence": 0.92,
    "evidence": {
      "runner_up_probability": 0.06,
      "margin": 0.86
    }
  }
}
```

Low-confidence response includes:

```json
{
  "action": "FALLBACK",
  "reason_code": "LOW_CONFIDENCE",
  "fallback": {
    "provider": "system-2-placeholder"
  }
}
```

## Security boundary

The endpoint returns a **decision recommendation**, not execution authorization.

A production caller must still apply deterministic policy, authorization, budget and approval rules.
