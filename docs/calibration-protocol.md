# Calibration Protocol

A confidence gate is only meaningful when confidence is measured on data that was not used to fit the decision model.

## Required split

```text
train
  -> model / adapter fitting

calibration
  -> fit temperature / isotonic mapping

test
  -> final accuracy, calibration and Risk-Coverage
```

Do not fit temperature on the test set.

## Reference implementation

The repository includes dependency-free temperature fitting over a held-out calibration split.

```bash
python examples/run_qwen_experiment.py \
  --model Qwen/Qwen3-0.6B \
  --calibration data/tool_router.calibration.jsonl \
  --test data/tool_router.test.jsonl \
  --output reports/qwen3-0.6b-tool-router.json
```

The resulting report records:

- fitted temperature,
- calibration NLL before/after,
- raw test metrics,
- calibrated test metrics,
- Risk-Coverage points,
- model/dataset metadata.

## Evidence rule

A public benchmark claim should additionally record:

- model revision,
- tokenizer revision,
- runtime/library versions,
- hardware,
- candidate formatting,
- sequence length normalization,
- random seed where relevant,
- raw per-case outputs.

The current repository provides the experiment path but does not claim real-model results until an actual model run is captured.

## Split provenance contract

A calibration digest by itself is insufficient to prove that calibration and
final test data are disjoint. A release-gate `decision-eval/v1` artifact may
therefore include a `calibration` provenance object with:

- `sha256`
- `case_count`
- `case_ids`

When that object is present:

1. its digest must match `calibration_sha256`;
2. case IDs must be non-empty and unique;
3. calibration case IDs must be disjoint from the test dataset case IDs;
4. the provenance is sealed into the artifact content digest.

This makes split leakage a validation failure rather than a convention hidden
inside benchmark scripts.

