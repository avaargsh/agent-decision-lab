# Typed decision output and calibration profile (experimental)

Two small **opt-in** contracts are available without modifying the existing
`/decision` request/response or the consumer-facing `decision-eval/v1`.

## Decision value: `decision-output/v1`

This is an adapter-neutral **recommendation**, not authority to execute.
The three variants have different, strict value semantics:

| Kind | Non-abstaining value | Abstain |
| --- | --- | --- |
| `choice` | one non-empty candidate identity | `value=null` |
| `boolean` | actual JSON boolean, never 0/1 | `value=null` |
| `score` | finite normalized number in [0,1] | `value=null` |

`abstain` is explicit. For abstention both `value` and `confidence`
must be `null`. `false` and `0.0` are **not** abstentions.

The confidence is a confidence **in the decision value**, not the score
value itself. It must be in [0,1] on non-abstaining output. Choice membership
requires supplying the original candidates to the Python verifier.
JSON Schema alone cannot validate membership against a separate request.

Example:

```python
from decision_lab.decision_output import build_decision_output

output = build_decision_output(
    kind="choice",
    value="prometheus.query",
    confidence=0.83,
    candidates=["prometheus.query", "logs.search"],
)
```

## Calibration: `calibration-profile/v1`

This is a content-addressed snapshot of the temperature and decision
threshold selected on a **held-out calibration split**. It binds:

- exact adapter and model reference;
- decision type and candidate-set schema version;
- calibration SHA-256, case count and case IDs;
- optional frozen inventory digest;
- temperature, confidence/margin threshold, risk budget and selection rule.

`threshold_selection` distinguishes `calibration_risk_budget` from
`calibration_median_confidence_fallback`. The latter indicates no feasible
calibration operating point was found; it is not approval to automate.
The profile-bound gateway **rejects** median-confidence fallback selection for
automatic decisions. Such profiles are retained as benchmark evidence only.
The profile records **a producer claim about the fitting method**, not proof
that the model is calibrated or that the selected risk holds in production.

### Building and applying a profile

```python
from decision_lab.calibration_profile import (
    build_calibration_profile,
    build_profile_bound_gateway,
)

profile = build_calibration_profile(
    adapter="frozen-logits",
    model_ref="Qwen/Qwen3-0.6B@pinned-revision",
    decision_type="mcp_tool_router",
    calibration={
        "sha256": "sha256:" + "a" * 64,
        "case_count": 2,
        "case_ids": ["cal-1", "cal-2"],
    },
    temperature=2.0,
    execute_threshold=0.7,
    risk_budget=0.1,
    threshold_selection="calibration_risk_budget",
)

gateway = build_profile_bound_gateway(
    scorer,  # caller's real candidate-score function
    profile=profile,
    adapter="frozen-logits",
    model_ref="Qwen/Qwen3-0.6B@pinned-revision",
    decision_type="mcp_tool_router",
    test_case_ids=["test-1"],
)
```

Example hashes and parameters above are illustrative, **not fitted or
measured experimental values**.

The bound gateway fails closed if profile integrity, adapter identity, model
revision, decision type, optional inventory identity, or supplied test split
is incompatible. It validates raw candidate probabilities *before* applying
temperature and threshold. The calling code is responsible for reliably
supplying/attesting the actual runtime model and inventory identities.
A matching string alone is not cryptographic attestation of the runtime.

## Compatibility and scope

- Current `DecisionRequest`/`DecisionGateway` remain choice-based.
- Typed boolean/score **representations** are defined, but no trained B/C
  model, Boolean/Score Gateway executor, or universal decision head is claimed.
- No automatic profile promotion to server `qwen` mode is included.
- Existing `decision-eval/v1` release evidence and Agent Control Plane
  consumers are unchanged.
- A decision recommendation never bypasses deterministic policy, budgets,
  approval or capability authorization.
- Keep calibration/test disjoint; do not fit thresholds on abstention cases.
- Before production deployment, require M5/M6 replay evidence and a
  trace-backed quality corpus.

Schemas: `schemas/decision-output-v1.schema.json` and
`schemas/calibration-profile-v1.schema.json`.
