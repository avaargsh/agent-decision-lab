import argparse

from decision_lab.benchmark import load_jsonl
from decision_lab.experiment import (
    run_calibrated_experiment,
    write_experiment_report,
)
from decision_lab.logits import FrozenLogitAdapter, default_candidate_prompt
from decision_lab.transformers_backend import TransformersCausalLMBackend


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
parser.add_argument("--calibration", required=True)
parser.add_argument("--test", required=True)
parser.add_argument("--output", default="qwen-frozen-logits-report.json")
args = parser.parse_args()

backend = TransformersCausalLMBackend(
    model_id=args.model,
    length_normalize=True,
)
adapter = FrozenLogitAdapter(
    backend=backend,
    prompt_builder=default_candidate_prompt,
)

fit, raw, calibrated = run_calibrated_experiment(
    adapter,
    calibration_cases=load_jsonl(args.calibration),
    test_cases=load_jsonl(args.test),
)

write_experiment_report(
    args.output,
    fit=fit,
    raw=raw,
    calibrated=calibrated,
    metadata={
        "model": args.model,
        "adapter": adapter.name,
        "calibration_dataset": args.calibration,
        "test_dataset": args.test,
    },
)

print(f"temperature={fit.temperature:.4f}")
print(f"raw_accuracy={raw.accuracy:.4f}")
print(f"calibrated_accuracy={calibrated.accuracy:.4f}")
print(f"report={args.output}")
