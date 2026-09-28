# Same-Base Structured-Output Baseline

The main comparison should not be:

```text
small local decision model
vs
large remote proprietary model
```

because that mixes model quality, serving stack and decision method.

The repository now supports a cleaner same-base comparison:

```text
Qwen3-0.6B
   |
   +-- Frozen candidate continuation log-probabilities
   |
   +-- Autoregressive JSON structured-choice generation
```

Both paths receive the same:

- decision type,
- context,
- candidate set,
- calibration split,
- test split.

## Structured output contract

The model is asked to generate:

```json
{"candidate":"<allowed>","confidence":0.0}
```

Generation is deterministic (`do_sample=False`).

If parsing fails or the model generates a candidate outside the bounded set, the reference adapter emits a uniform distribution. That makes malformed output visible as degraded benchmark performance rather than crashing the whole experiment.

## Metrics

Both arms feed the same benchmark runner:

- Accuracy
- Brier
- ECE
- p50 / p95 latency
- tokens processed per decision
- Risk-Coverage
- False Automation Rate

## Important limitation

The current token accounting is operational rather than theoretical FLOP accounting:

- Frozen logits counts tokens processed across candidate forward passes.
- Structured output counts prompt + generated tokens.

This is useful for comparing the current reference implementations, but should not be mistaken for hardware-normalized compute cost.
