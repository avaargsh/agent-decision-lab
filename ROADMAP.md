# Roadmap

## v0.1 — Decision Gateway
- [x] candidate-set request schema
- [x] confidence threshold
- [x] margin threshold
- [x] structured decision result
- [x] audit-friendly decision ledger
- [x] calibration interfaces
- [x] JSONL benchmark format
- [x] benchmark runner
- [x] structured-output baseline adapter

## v0.2 — Benchmark
- [x] MCP tool-routing seed dataset
- [x] policy-gate seed dataset
- [x] severity seed dataset
- [x] escalation seed dataset
- [x] Risk-Coverage computation
- [x] false automation analysis
- [x] risk-budget operating-point selection
- [x] fallback-rate reporting in benchmark matrices
- [x] separate calibration/test demo fixtures
- [x] p50 / p95 latency reporting
- [x] tokens-processed accounting
- [x] test-score cache for raw/calibrated reuse
- [ ] real labeled benchmark corpus
- [ ] ambiguity/risk annotation guidelines

## v0.3 — Model adapters
- [x] frozen-LM candidate-logit adapter
- [x] optional Hugging Face causal-LM backend
- [x] batched candidate scoring
- [x] temperature fitting on held-out calibration split
- [x] calibrated adapter wrapper
- [x] reproducible JSON experiment report
- [x] real-model GitHub Actions workflow
- [x] first captured Qwen3-0.6B real-model artifact
- [x] same-base autoregressive structured-output adapter
- [x] captured Frozen vs Structured same-base comparison artifact
- [x] optimized batched-Frozen vs Structured performance rerun
- [ ] Decision LoRA
- [ ] dedicated decision-head model
- [ ] encoder baseline
- [ ] vLLM / SGLang serving adapter

## v0.4 — Decision SDK
- [x] HTTP /decision API
- [x] fallback provider interface
- [ ] policy hook
- [x] optional OpenTelemetry spans
- [x] replayable Decision Ledger record format
