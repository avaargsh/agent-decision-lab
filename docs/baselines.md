# Baselines

## Principle

All model families must solve the **same bounded decision task** with the same candidate set and evaluation cases.

## Structured-output autoregressive baseline

The baseline generates:

```json
{
  "candidate": "read_metrics",
  "confidence": 0.88
}
```

This measures the common Agent pattern of asking an LLM to produce a structured decision.

Caveat: a generated confidence value is not automatically calibrated. Treat it as model output that must be measured, not trusted.

## Frozen-logit baseline

Future adapter:

1. format the decision context and candidates,
2. run one forward/prefill pass,
3. extract candidate logits,
4. normalize only over the candidate set,
5. calibrate on a held-out calibration split.

## Dedicated decision head

Future adapter:

- shared language/encoder backbone,
- task-specific choice/boolean/score head,
- no autoregressive text generation on the hot path.

## Comparison

At minimum report:

- Accuracy / Macro-F1
- NLL / Brier / ECE
- Risk-Coverage
- False Automation Rate
- fallback rate
- latency
- tokens / decision
- cost / 1K decisions

Do not compare systems solving different candidate sets or using different label definitions.
