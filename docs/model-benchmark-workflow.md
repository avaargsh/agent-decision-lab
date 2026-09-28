# Manual Model Benchmark Workflow

The repository includes a **manual-only** GitHub Actions workflow for a first real Frozen-LM experiment.

It is intentionally not triggered on push or pull request because model downloads and inference are materially more expensive than unit tests.

## Workflow

`.github/workflows/model-benchmark.yml`

Inputs:

- Hugging Face model checkpoint,
- calibration JSONL,
- test JSONL.

Default model:

`Qwen/Qwen3-0.6B`

The default datasets under `benchmarks/demo/` are synthetic plumbing fixtures. They validate the experiment pipeline but are **not** a publishable benchmark corpus.

## Output artifact

The workflow uploads:

- fitted temperature,
- raw benchmark metrics,
- calibrated benchmark metrics,
- Risk-Coverage data,
- per-case results,
- model/dataset metadata,
- Python/package environment.

## Publication rule

Do not cite the demo dataset as evidence of model quality.

A publishable result should use:

1. independently labeled cases,
2. train/calibration/test separation,
3. versioned model/tokenizer revisions,
4. hardware/runtime metadata,
5. raw per-case evidence.
