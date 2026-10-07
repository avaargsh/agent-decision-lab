# M5 model stress experiment

M5 model-quality evidence is intentionally split from day-to-day CPU development.

Current development environment has no usable GPU. Until a rented cloud GPU is
available:

- CPU runs are smoke evidence only
- latency / throughput numbers are not acceptance evidence
- M5 empirical acceptance remains open
- calibration, robustness, scoring and report logic continue to be implemented
  and unit-tested without blocking on hardware

The cloud-GPU replay issue is #40.

## Model arms prepared now

The replay currently supports:

1. Qwen frozen candidate continuation scoring
2. same-base autoregressive structured output

A bias-corrected continuation-scoring arm will be added next without changing
the frozen workload contract.

## Frozen prompt variant

The original frozen-logit baseline lists every candidate inside the prompt and
then scores every candidate continuation. That is acceptable at K=4 but causes
avoidable prompt-token growth when K reaches 50 or 100.

M5 therefore uses `compact_candidate_prompt`:

- decision type and context stay in the prompt
- the complete candidate list is not duplicated in the prompt
- the bounded set is still enforced by scoring only explicit candidate
  continuations
- the report records `compact-candidate-v1` as experiment provenance

The original prompt remains unchanged for historical v1 artifacts.

## High-K memory control

K=100 must not silently become a smaller benchmark just because a rented GPU has
limited memory.

`TransformersCausalLMBackend.candidate_batch_size` therefore chunks candidate
continuations across forward passes while preserving:

- exact candidate set
- exact prompt
- exact candidate order
- exact per-candidate continuation score semantics

The selected chunk size is recorded in model and GPU replay provenance.

## Calibration and transfer

For each adapter:

1. score the 16-case calibration seed
2. fit temperature on calibration only
3. select a confidence threshold on calibration only using the declared observed
   risk budget
4. if no non-empty risk-budget operating point exists, use the calibration
   median confidence and record that fallback method
5. reuse the exact threshold for K=5/10/20/50/100 without refitting

For every K the report records:

- Accuracy / Macro-F1 / NLL / Brier / ECE on covered cases
- Risk-Coverage at the transferred threshold
- FAR on missing-candidate cases
- FRR on covered cases
- threshold-transfer deltas
- latency / token accounting from the benchmark runner
- permutation flip rate on a fixed small sample

## One-shot rented GPU replay

From a fresh clone on a CUDA-capable machine:

```bash
MODEL=Qwen/Qwen3-0.6B \
CANDIDATE_BATCH_SIZE=16 \
bash scripts/run_m5_gpu_replay.sh
```

The script fails closed if `nvidia-smi` is missing or
`torch.cuda.is_available()` is false.

It then:

1. installs model dependencies unless `SKIP_INSTALL=1`
2. fetches the pinned AWS Labs MCP source revision
3. regenerates and verifies the frozen 804-tool inventory
4. rebuilds K=5/10/20/50/100 stress inputs
5. executes the prepared model matrix
6. captures GPU / Python / package / git provenance
7. writes a content-addressed run directory with `run-manifest.json` and
   `sha256sums.txt`

The GitHub workflow is manual-only and targets a self-hosted runner labeled
`gpu`. There is no automatic push-triggered CPU model run.

The initial replay uses deterministic random distractors. Lexical and true
semantic-nearest hard-negative model runs remain separate follow-up experiments.
