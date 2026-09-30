import argparse
import json
from dataclasses import asdict
from pathlib import Path

from decision_lab.benchmark import load_jsonl
from decision_lab.experiment import (
    run_calibrated_experiment,
)
from decision_lab.logits import (
    FrozenLogitAdapter,
    default_candidate_prompt,
)
from decision_lab.provenance import dataset_provenance
from decision_lab.transformers_backend import (
    TransformersCausalLMBackend,
)
from decision_lab.transformers_structured import (
    TransformersStructuredOutputAdapter,
)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    default="Qwen/Qwen3-0.6B",
)
parser.add_argument(
    "--calibration",
    required=True,
)
parser.add_argument(
    "--test",
    required=True,
)
parser.add_argument(
    "--output",
    default="qwen-comparison.json",
)
args = parser.parse_args()

calibration_cases = load_jsonl(
    args.calibration
)
test_cases = load_jsonl(args.test)

frozen_backend = TransformersCausalLMBackend(
    model_id=args.model,
    length_normalize=True,
)
frozen = FrozenLogitAdapter(
    backend=frozen_backend,
    prompt_builder=default_candidate_prompt,
)

frozen_fit, frozen_raw, frozen_calibrated = (
    run_calibrated_experiment(
        frozen,
        calibration_cases=calibration_cases,
        test_cases=test_cases,
    )
)

# Release the first model before loading the generative baseline.
del frozen
del frozen_backend

structured = TransformersStructuredOutputAdapter(
    model_id=args.model,
    max_new_tokens=32,
)

structured_fit, structured_raw, structured_calibrated = (
    run_calibrated_experiment(
        structured,
        calibration_cases=calibration_cases,
        test_cases=test_cases,
    )
)

payload = {
    "metadata": {
        "model": args.model,
        "calibration_dataset": dataset_provenance(
            args.calibration,
            calibration_cases,
        ),
        "test_dataset": dataset_provenance(
            args.test,
            test_cases,
        ),
        "warning": (
            "Synthetic benchmark data; model inference is real. "
            "Do not treat results as production-traffic quality."
        ),
    },
    "frozen_logits": {
        "temperature_fit": asdict(
            frozen_fit
        ),
        "raw": asdict(frozen_raw),
        "calibrated": asdict(
            frozen_calibrated
        ),
    },
    "structured_output": {
        "temperature_fit": asdict(
            structured_fit
        ),
        "raw": asdict(structured_raw),
        "calibrated": asdict(
            structured_calibrated
        ),
    },
}

Path(args.output).write_text(
    json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print(
    json.dumps(
        {
            "frozen_accuracy": (
                frozen_raw.accuracy
            ),
            "structured_accuracy": (
                structured_raw.accuracy
            ),
            "output": args.output,
        },
        indent=2,
    )
)
