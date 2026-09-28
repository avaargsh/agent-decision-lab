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
- [x] separate calibration/test demo fixtures
- [ ] real labeled benchmark corpus
- [ ] ambiguity/risk annotation guidelines

## v0.3 — Model adapters
- [x] frozen-LM candidate-logit adapter
- [x] optional Hugging Face causal-LM backend
- [x] temperature fitting on held-out calibration split
- [x] calibrated adapter wrapper
- [x] reproducible JSON experiment report
- [x] manual real-model GitHub Actions workflow
- [ ] first captured Qwen real-model benchmark artifact
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
