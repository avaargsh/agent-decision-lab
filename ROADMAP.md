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
- [x] semantic embedding snapshot / gold-neighbour ranker contract
- [ ] retained semantic-nearest-neighbour distractor construction
- [x] frozen real MCP inventory snapshot corpus
- [x] source-grounded near-neighbour labeled corpus
- [x] trace-backed corpus provenance / sanitization / split-isolation contract
- [ ] production/trace-backed labeled benchmark corpus
- [x] ambiguity/risk annotation guidelines
- [x] explicit underspecified / multi-valid / missing / unsupported abstention corpus

- [x] opt-in typed Choice / Boolean / Score result and abstention contract
- [x] sealed calibration-profile/v1 with calibration-split provenance
- [x] choice gateway profile binding with model/task/inventory guards
- [x] wire per-arm M5/M6 replay profile production from calibration reports
- [ ] retain and independently review real GPU-produced M5/M6 profiles and metrics
- [ ] promote a real-model `qwen` HTTP serving mode only after replay acceptance

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
- [x] combined M5 + M6 GPU replay protocol with grouped abstention FAR
- [x] fail-closed pinned HF model revision and byte-level snapshot evidence
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
