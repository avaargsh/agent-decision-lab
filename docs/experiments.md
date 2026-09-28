# Experiment Plan

## Arms

### A — Frozen LM candidate scoring
Use a pretrained LM without full autoregressive generation and score a bounded candidate set from model logits.

### B — Decision LoRA
Adapt a small/medium causal LM specifically for bounded candidate decisions.

### C — Dedicated decision heads
Attach task-specific Choice / Boolean / Score heads to a compact language-model backbone.

### D — Encoder baseline
Use an encoder model with a classification or ranking head.

### E — Generative baseline
Use the same or comparable base model with structured-output autoregressive generation.

## Task families

1. MCP Tool Router
2. Policy Gate
3. Severity / Relevance
4. Escalation

## Metrics

### Predictive
- Accuracy
- Macro-F1
- AUROC / AUPRC

### Calibration
- NLL
- Brier
- ECE

### Selective automation
- Risk-Coverage
- False Automation Rate
- fallback rate

### Systems
- p50 / p95 latency
- tokens per decision
- cost per 1K decisions
- throughput

## Experimental rule

Compare models at matched task definitions and candidate sets. Do not report a faster bounded scorer against a stronger baseline solving a different task.
