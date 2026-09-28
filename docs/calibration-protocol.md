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
