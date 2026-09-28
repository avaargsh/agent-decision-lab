# Dataset Design

The seed datasets in this repository are **synthetic fixtures**, not benchmark claims.

Their purpose is to stabilize:

- task definitions,
- candidate semantics,
- file format,
- runner behavior,
- calibration/evaluation code.

## Task families

### MCP Tool Router

Choose one bounded tool/capability from an explicit candidate set.

### Policy Gate

Candidates:

- allow
- deny
- escalate

This is a decision benchmark, not the enforcement layer. Production authorization still belongs to deterministic policy.

### Severity

Map incident evidence/context to a bounded severity class.

Labels must be organization-specific in real datasets. The synthetic examples only exercise the benchmark contract.

### Escalation

Candidates:

- execute
- fallback
- human_review

This task is especially useful for selective automation and Risk-Coverage evaluation.

## Data quality rules

Real benchmark cases should include:

- provenance,
- a stable labeling rubric,
- ambiguity notes,
- risk class,
- automation eligibility,
- evidence references,
- train/calibration/test separation.

Avoid training and evaluating on lightly reworded copies of the same operational incident.
