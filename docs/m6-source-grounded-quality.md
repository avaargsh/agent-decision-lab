# M6 source-grounded quality replay

M6 complements the existing M5 candidate-scaling protocol.

The two protocols deliberately remain separate in the same future GPU artifact:

## M5 — candidate-set stress

Dataset identity:

- v2 authored synthetic intents
- real frozen MCP tool identities
- K = 5 / 10 / 20 / 50 / 100
- random distractors for the first retained GPU run
- gold-present and gold-missing variants

Purpose:

- candidate-count scaling
- calibration drift with K
- missing-candidate FAR
- permutation robustness
- serving cost / latency

## M6 — source-grounded routing quality

Dataset identity:

- v3 source-grounded authored calibration set: 20 cases
- v3 source-grounded authored test set: 20 cases
- v3 explicit abstention set: 16 cases
- labels grounded in pinned upstream AWS Labs MCP tool contracts

Purpose:

- near-neighbour closed-set routing quality
- threshold transfer from source-grounded calibration to test
- explicit rejection quality for:
  - underspecified
  - multi-valid
  - missing candidate
  - unsupported capability

## Threshold discipline

M5 and M6 select their thresholds independently from their own **calibration
split only**.

For M6:

1. fit temperature on the 20-case v3 calibration split
2. select the operating threshold on the same calibration split
3. freeze that temperature and threshold
4. evaluate the 20-case v3 closed test
5. evaluate the 16-case abstention corpus
6. never refit from test or abstention evidence

The abstention corpus therefore cannot improve its own threshold after the fact.

## FAR reporting

The M6 report contains:

- overall false accept rate
- unsafe automation rate
- grouped false accept rate by:
  - `abstention_reason`
  - `coverage`
  - `ambiguity`
  - `automation_risk`

The canonical `abstention_reason` mapping is:

- covered + underspecified → `underspecified`
- covered + multi-valid → `multi_valid`
- missing coverage → `missing_candidate`
- unsupported capability → `unsupported`

Each reason currently has four retained authored cases.

Grouped FAR is calculated from one model-evaluation pass. Grouping does not
re-run the model and does not alter the threshold.

## Three comparable model arms

The future rented-GPU replay measures the same protocols for:

1. frozen candidate continuation logits
2. prior-corrected frozen logits
3. same-base structured output

The prior-corrected arm remains AnyJev-inspired candidate-prior subtraction; it
is not described as exact AnyJev L0.

## Combined artifact

`scripts/run_m5_gpu_replay.sh` now emits:

`decision-gpu-replay.json`

with schema:

`decision-gpu-replay/v1`

Top-level sections:

- `m5_candidate_scaling`
- `m6_source_grounded_quality`
- `calibration_profiles` (additive evidence; each M5/M6 model arm
  emits a sealed `calibration-profile/v1` under
  `decision-gpu-calibration-profiles/v1`)

The emitted profiles are derived from the measured calibration report and
the **exact calibration JSONL bytes**. Model/adapter, frozen inventory,
temperature, operating-point selection, and calibration case IDs are bound
into each content digest. M5 and M6 never share profiles or refit on test
or abstention cases. If a risk-budget operating point is unavailable, the
fallback-threshold profile remains replay evidence only: the profile-bound
gateway refuses to automate from it.

Content-addressing proves integrity, not independent fit correctness,
production routing quality, or cryptographic verification of loaded model
weights. This does not promote the model-serving HTTP path.

## Immutable model checkpoint evidence

The one-shot replay requires `MODEL_REVISION` to be an explicit lowercase,
40-character Hugging Face commit SHA. Mutable branches and tags are rejected.
The manual GitHub Actions workflow also requires `model_revision`; it has no
default.

```bash
# Resolve the commit once, and record the resulting SHA with the run request.
python -c 'from huggingface_hub import HfApi; print(HfApi().model_info("Qwen/Qwen3-0.6B").sha)'
MODEL=Qwen/Qwen3-0.6B MODEL_REVISION=<40-hex-commit> \\
  bash scripts/run_m5_gpu_replay.sh
```

All three arms (including the autoregressive baseline) load the same
**locally materialized snapshot**, not a separately resolved remote model.
The run writes `model-snapshot.json` (`model-snapshot/v1`) with the
model ID, commit, per-file SHA-256 and combined content digest. The
combined report, all six calibration profiles, environment and run manifest
carry the pinned model reference/digest. A wrong revision, missing weight
file, or cache symlink escape fails closed.

This establishes source and file-byte identity, not bitwise deterministic
CUDA execution, independent runtime attestation, or production model quality.
No actual GPU evaluation has been performed by this change.

The run directory also contains:

- regenerated frozen MCP inventory
- M5 generated stress inputs
- GPU / driver information
- Python/package environment
- `run-manifest.json`
- `sha256sums.txt`

The bundle manifest schema is:

`decision-gpu-replay-bundle/v1`

No GPU quality claim is made until Issue #40 has a retained successful cloud-GPU
artifact.
