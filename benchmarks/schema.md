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

## Robustness cases

Closed-set benchmark cases require `gold_candidate` to be present in
`candidates`.

Robustness suites may intentionally violate that assumption when the benchmark
tests candidate coverage. A missing-candidate case must state:

```json
{
  "metadata": {
    "expected_abstain": true,
    "shift": "missing_candidate"
  }
}
```

The `gold_candidate` is still retained as labeling evidence even though it is
not presented to the model. Such cases must be evaluated with the abstention/OOD
robustness path rather than the ordinary closed-set benchmark runner.

Generated candidate-stress cases should additionally record:

- `parent_case_id`
- `inventory_id`
- `inventory_sha256`
- `candidate_count`
- `distractor_strategy`
- `generation_seed`
