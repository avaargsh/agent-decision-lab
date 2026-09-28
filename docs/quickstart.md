# Quickstart

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Run the seed benchmark

```bash
decision-lab benchmark benchmarks/examples/tool_router.jsonl
```

The demo adapter is deterministic and exists only to validate the benchmark plumbing.

## Run the HTTP decision service

```bash
python examples/http_server.py
```

Then:

```bash
curl -s http://127.0.0.1:8080/decision \
  -H 'content-type: application/json' \
  -d '{
    "decision_type":"mcp_tool_router",
    "candidates":["read_metrics","read_logs","restart_workload"],
    "context":{"intent":"metrics"}
  }'
```

Expected behavior:

- confident bounded decisions return `EXECUTE`,
- uncertain decisions return `FALLBACK`,
- every result includes an audit-friendly Decision Ledger entry.

The service does **not** authorize or execute external actions.
