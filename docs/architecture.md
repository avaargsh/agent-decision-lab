# Architecture

## Problem

Many Agent decisions are bounded choices: route to one of N tools, classify severity, allow/deny/escalate, or choose whether to fall back to a more capable reasoner.

Generating prose or structured JSON autoregressively for every such decision adds latency, tokens and failure modes that may be unnecessary.

## Decision plane

```text
Input Context
    |
Candidate Builder
    |
Decision Scorer
    |
Calibration
    |
Confidence / Risk Gate
    |---------------- low confidence --------------|
    v                                              v
Bounded Decision                              System-2 Reasoner
    |                                              |
    +----------------------v-----------------------+
                           |
                Deterministic Authorization
                           |
                     Decision Ledger
```

## Separation of responsibilities

### Learned decision plane

May produce:

- candidate scores,
- calibrated confidence,
- relevance / severity scores,
- fallback recommendations.

### Deterministic runtime

Owns:

- authorization,
- policy invariants,
- budget limits,
- irreversible-action approval,
- retries and idempotency,
- final execution.

The learned model is not the security boundary.

## Decision ledger

Audit records should contain externally meaningful decision facts rather than hidden reasoning:

- decision type,
- candidate set,
- scores / confidence,
- selected action,
- policy checks,
- evidence references,
- fallback / retry,
- outcome.

## Evaluation

A strong decision model is not simply the one with the highest raw accuracy.

Production evaluation must include calibration and selective automation:

- NLL / Brier / ECE,
- Risk-Coverage,
- false automation rate,
- fallback rate,
- latency and cost,
- downstream task outcome.

For selective automation, the benchmark may be given an explicit maximum observed
risk budget. It then selects the highest-coverage non-empty threshold that stays
inside that budget and reports the corresponding threshold, coverage, risk, false
automation rate, and fallback rate. This is an evaluation operating point, not an
authorization policy: production execution must still pass deterministic policy,
approval, and runtime safety controls.
