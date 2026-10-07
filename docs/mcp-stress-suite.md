# MCP candidate stress-suite generation

M5 requires candidate-count and missing-candidate experiments to be generated
from a frozen tool inventory rather than hand-edited JSONL files.

## Inventory contract

A snapshot uses `mcp-tool-inventory/v1` and contains:

- a stable `inventory_id`
- exact source/provenance metadata
- tool name
- server identity
- tool description
- optional tool metadata

The canonical payload has a SHA-256 digest. Generated benchmark cases record both
`inventory_id` and `inventory_sha256`.

A checked-in inventory is evidence only when its `source` identifies where the
tool list came from and which immutable revision/snapshot was captured. Synthetic
inventories are allowed for tests but must not be described as real MCP evidence.

## Candidate-count suite

`build_candidate_count_suite` expands a bounded closed-set case across declared
candidate counts such as 5, 10, 20, 50 and 100.

Invariants:

- the gold candidate is always present
- the source split is preserved
- candidate order is deterministic for a fixed seed
- the exact inventory digest is attached to every generated case
- test cases are not used to fit calibration or choose a threshold

## Missing-candidate suite

`build_missing_candidate_suite` deliberately removes the gold tool and marks:

```json
{
  "expected_abstain": true,
  "shift": "missing_candidate"
}
```

These cases belong in the abstention robustness evaluator and must not be passed
to the ordinary closed-set runner, whose metrics assume the gold candidate is
available.

## Distractors

Two deterministic strategies are available now:

- `random`: stable hash ordering for a fixed seed
- `lexical`: token-overlap hard negatives

The lexical strategy is deliberately named lexical, not semantic. It is a
dependency-free baseline. M5 may later plug an embedding or retrieval ranker into
the same `hard_negative_ranker` interface without changing the benchmark case
contract.

## Example

```bash
python examples/build_mcp_stress_suite.py \
  --inventory benchmarks/mcp_tool_router/inventory.json \
  --base benchmarks/mcp_tool_router/v1.test.jsonl \
  --candidate-counts 5,10,20,50,100 \
  --strategy random \
  --covered-output /tmp/mcp-covered.jsonl \
  --missing-output /tmp/mcp-missing.jsonl
```

The repository does not yet claim that a real MCP inventory snapshot is frozen.
That remains an explicit M5 acceptance item.
