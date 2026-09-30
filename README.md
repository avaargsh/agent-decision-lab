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

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

Run the deterministic demo Decision Gateway:

```bash
decision-lab-server --mode demo --host 127.0.0.1 --port 8080
```

`DECISION_MODE=demo` is a deterministic reference scorer, **not Qwen inference**. Real-model Qwen experiments live in the experiment/benchmark path; `qwen` server mode intentionally fails closed until a serving adapter is promoted.

See `docs/quickstart.md` and `docs/api.md` for the benchmark and HTTP contracts.

Real-model comparison reports bind both calibration and test inputs to their exact SHA-256 bytes, case count and case IDs. This makes a reported metric replayable against the dataset revision that actually produced it instead of relying on a mutable file path alone.

## Status

Public pre-1.0 research and engineering repository. The Decision Gateway, calibration primitives, benchmark harness, deterministic demo server, and Qwen experiment path are implemented. Dedicated decision heads, Decision LoRA, and vLLM/SGLang serving remain experimental roadmap work.

## Contributing and license

Contributions are welcome through focused issues and pull requests. See `CONTRIBUTING.md`, `SECURITY.md`, and `CODE_OF_CONDUCT.md`.

Licensed under Apache License 2.0. See `LICENSE`.
