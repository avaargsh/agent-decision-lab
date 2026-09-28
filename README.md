# Agent Decision Lab

A research and engineering playground for moving **high-frequency, closed-set, verifiable Agent decisions** off the autoregressive LLM hot path.

The core hypothesis is simple:

> Use a small decision model / decision head for bounded choices, calibrate its confidence, and fall back to a System-2 reasoner only when uncertainty is high.

## Architecture

```text
Request
  |
Candidate Builder
  |
Decision Model / Head
  |
Calibration
  |
Confidence Gate
  |-------------------- low confidence -------------------->
  |                                                     System-2 LLM
  v
Deterministic Policy / Authorization
  |
Decision Ledger
  |
Execute
```

The model can recommend an action; the deterministic runtime still owns authorization and safety constraints.

## Initial benchmark tasks

- MCP tool routing
- policy gate / allow-deny-escalate
- severity and relevance classification
- escalation / fallback decision

## Metrics

- Accuracy / Macro-F1
- NLL / Brier Score / ECE
- AUROC / AUPRC where applicable
- Risk-Coverage
- False Automation Rate
- latency, tokens and cost
- fallback rate

## Experimental arms

| Arm | Approach |
| --- | --- |
| A | Frozen LM candidate logits / SemIf-style scoring |
| B | Qwen-family Decision LoRA / candidate scorer |
| C | Small LM with dedicated decision heads |
| D | Encoder model + decision head |
| E | Same-base autoregressive structured-output baseline |

## Repository layout

```text
src/decision_lab/      # decision gateway and calibration primitives
benchmarks/            # benchmark schemas and datasets
examples/              # minimal runnable examples
tests/                 # unit tests
docs/                  # architecture and experiment design
```

## v0.1 goal

Ship an implementation-neutral **Decision Gateway** with:

1. candidate-set input,
2. calibrated confidence,
3. configurable fallback,
4. structured decision evidence,
5. a benchmark harness that can compare decision models against generative baselines.

## Status

Private incubation repository. The goal is to make the design and experiments reproducible before public release.
