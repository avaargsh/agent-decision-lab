# MCP routing annotation guidelines

Version: `mcp-routing-label/v1`

These annotations describe the **routing decision**, not authorization. A router
can correctly identify a write-capable tool while a separate policy/authority
plane still denies execution.

## Coverage

### `covered`

The intended gold tool is present in the candidate set and one candidate is
sufficiently better than the alternatives to support a single-gold label.

### `missing_candidate`

The intended tool exists in the frozen inventory but is intentionally absent
from the presented candidate set.

These cases must set `expected_abstain=true`.

### `unsupported`

The requested capability cannot be satisfied by the declared candidate
universe. These cases must also abstain.

Do not use low confidence as a substitute for deciding whether candidate
coverage is valid.

## Ambiguity

### `clear`

The request directly names or uniquely identifies one tool capability and the
other candidates are not plausible substitutes.

### `near_neighbor`

Multiple candidates are semantically close, but upstream tool contracts provide
a defensible single best answer.

Examples:

- instant PromQL vs range PromQL
- direct Logs Insights execution vs cross-region batch execution
- documentation search vs reading a known URL
- Kubernetes resource listing vs logs/events/metrics

This is the preferred difficulty class for closed-set routing evaluation.

### `underspecified`

The request lacks information needed to choose a single candidate.

Example: "check the checkout problem" when logs, metrics and events are all
plausible and no further signal is supplied.

These cases must set `expected_abstain=true`.

### `multi_valid`

Two or more candidates are materially valid for the stated task and the
benchmark cannot justify one unique gold choice from the request alone.

These cases must abstain rather than assigning an arbitrary gold winner.

## Automation risk

Risk describes the consequence of **automatically proceeding after routing**,
not the semantic difficulty of classification.

### `low`

Ordinary read-only discovery/retrieval with limited direct operational impact.

Examples: metrics, resource listing, public documentation.

### `medium`

Sensitive operational reads or actions where incorrect routing can expose,
retrieve, or materially change the scope of operational information.

Examples: pod logs, Kubernetes events, application log analysis.

### `high`

Write-capable or mutating operations.

Examples: create/update/delete Kubernetes resources.

A high-risk case must use `operation_effect=write_capable`. Routing confidence
never grants authority to execute it.

## Operation effect

- `read_only`: ordinary read/retrieval
- `sensitive_read`: potentially sensitive logs/events/data reads
- `write_capable`: candidate can mutate external state for this request
- `unknown`: evidence is insufficient to classify effect

A write-capable case may not be annotated low-risk.

## Label basis and provenance

Source-grounded cases use:

`label_basis=upstream_tool_contract`

and must provide `evidence_refs` containing:

- repository
- immutable revision
- source path

The gold rationale should explain **why this tool is better than its nearest
candidate**, not merely restate the tool name.

## Split discipline

Calibration and test cases must have:

- unique IDs
- disjoint decision contexts
- no threshold or temperature fitting on test
- no test examples used when authoring calibration-specific operating points

Near-paraphrases across the two splits should be avoided where they would make
the answer mechanically recoverable.

## Review rule

Do not call a corpus "real production traffic" unless requests originate from a
trace, user study, incident record, or another independently documented source.

The v3 corpus is **source-grounded authored benchmark traffic**: real upstream
tool contracts, authored intents. This is stronger than the v2 smoke seed but
is not production-trace evidence.
