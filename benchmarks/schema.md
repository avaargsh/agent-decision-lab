# Agent Decision Benchmark JSONL Schema

Each line is one bounded decision case.

```json
{
  "case_id": "tool-001",
  "decision_type": "mcp_tool_router",
  "context": {"intent": "check CPU saturation"},
  "candidates": ["read_metrics", "read_logs", "restart_workload"],
  "gold_candidate": "read_metrics",
  "metadata": {
    "split": "dev",
    "risk": "low",
    "source": "synthetic"
  }
}
```

## Required fields

- `case_id`
- `decision_type`
- `context`
- `candidates`
- `gold_candidate`

## Metadata

Recommended metadata:

- dataset split,
- task family,
- risk class,
- source/provenance,
- evidence pointer,
- automation eligibility,
- notes about ambiguity.

The candidate set is part of the task definition. Baselines must solve the same candidate set.
