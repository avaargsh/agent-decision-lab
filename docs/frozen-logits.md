# Frozen-LM Candidate Logits

## Goal

Evaluate whether a pretrained causal LM can make bounded Agent decisions without an autoregressive structured-output loop.

```text
Context + Candidate Set
        |
        v
Frozen LM
        |
candidate continuation log-probabilities
        |
        v
Candidate-only normalization
        |
        v
Calibration
        |
        v
Confidence Gate
```

## Adapter contract

The core package defines a backend-neutral interface:

```python
logprob(prompt=..., candidate=...) -> float
```

`FrozenLogitAdapter` applies softmax only across the explicit candidate set.

## Hugging Face reference backend

Install optional model dependencies:

```bash
pip install -e ".[models]"
```

Example:

```bash
python examples/frozen_logits_qwen.py \
  --model Qwen/Qwen3-0.6B \
  --dataset benchmarks/examples/tool_router.jsonl
```

The example is intentionally configurable; the benchmark should not depend on one checkpoint.

## Important caveats

### Candidate wording matters

Candidate strings are part of the task definition. Different tokenization or semantic phrasing may alter scores.

### Sequence length matters

The reference backend can length-normalize continuation log-probability. Experiments should report whether sum or mean token log-probability is used.

### Calibration still matters

Candidate normalization produces a distribution, not a guarantee of calibrated confidence.

Always measure NLL, Brier, ECE and Risk-Coverage on held-out data.

### Fair baseline

The autoregressive structured-output baseline must receive the same context and the same candidate set.
