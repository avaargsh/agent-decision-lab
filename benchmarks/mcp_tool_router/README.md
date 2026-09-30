# MCP Tool Router Benchmark v1

This dataset fixes one bounded Agent Decision Plane task: route an operator intent to exactly one read-oriented MCP capability.

Candidate universe:

- `prometheus.query`
- `logs.search`
- `kubernetes.get`
- `runbook.search`

The calibration and test splits are intentionally disjoint and balanced across the four candidates. The text is synthetic and should not be presented as production traffic. The model inference path is real when run through `examples/run_qwen_comparison.py`.

## Reproducible experiment

```bash
python examples/run_qwen_comparison.py \
  --model Qwen/Qwen3-0.6B \
  --calibration benchmarks/mcp_tool_router/v1.calibration.jsonl \
  --test benchmarks/mcp_tool_router/v1.test.jsonl \
  --output reports/mcp-tool-router-v1.json
```

Calibration must be fit only on `v1.calibration.jsonl`; model quality and risk/coverage reporting must use `v1.test.jsonl`.
