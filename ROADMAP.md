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
- [x] candidate-order permutation robustness / flip-rate evaluation
- [x] missing-candidate and OOD abstention FAR/FRR evaluation
- [x] source-to-target threshold-transfer drift evaluation
- [x] candidate-count workload at K = 5 / 10 / 20 / 50 / 100
- [ ] semantic-nearest-neighbour distractor construction
- [x] frozen real MCP inventory snapshot corpus
- [ ] real labeled benchmark corpus
- [ ] ambiguity/risk annotation guidelines

## v0.3 — Model adapters
- [x] frozen-LM candidate-logit adapter
- [x] optional Hugging Face causal-LM backend
- [x] batched candidate scoring
- [x] high-K candidate chunking with invariant scoring semantics
- [x] temperature fitting on held-out calibration split
- [x] calibrated adapter wrapper
- [x] reproducible JSON experiment report
- [x] fail-closed one-shot cloud-GPU replay harness
- [x] same-base autoregressive structured-output adapter
- [x] AnyJev-inspired candidate-prior correction baseline
- [ ] rented-GPU retained three-arm stress artifact
- [ ] exact option-rotation / label-prior ablation
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
