import argparse
import json
from dataclasses import asdict
from pathlib import Path

from decision_lab.benchmark import load_jsonl
from decision_lab.eval_artifact import build_eval_artifact
from decision_lab.experiment import run_calibrated_experiment
from decision_lab.fallback_eval import evaluate_system2_fallback
from decision_lab.logits import FrozenLogitAdapter, default_candidate_prompt
from decision_lab.provenance import dataset_provenance
from decision_lab.transformers_backend import TransformersCausalLMBackend
from decision_lab.transformers_structured import TransformersStructuredOutputAdapter


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
parser.add_argument("--calibration", required=True)
parser.add_argument("--test", required=True)
parser.add_argument("--output", default="qwen-comparison.json")
parser.add_argument("--fallback-threshold", type=float, default=0.8)
parser.add_argument(
    "--eval-artifact",
    default=None,
    help="Backward-compatible alias for --frozen-eval-artifact.",
)
parser.add_argument("--frozen-eval-artifact", default=None)
parser.add_argument("--structured-eval-artifact", default=None)
args = parser.parse_args()

calibration_cases = load_jsonl(args.calibration)
test_cases = load_jsonl(args.test)

frozen_backend = TransformersCausalLMBackend(
    model_id=args.model,
    length_normalize=True,
)
frozen = FrozenLogitAdapter(
    backend=frozen_backend,
    prompt_builder=default_candidate_prompt,
)

frozen_fit, frozen_raw, frozen_calibrated = run_calibrated_experiment(
    frozen,
    calibration_cases=calibration_cases,
    test_cases=test_cases,
)

# Release the first model before loading the generative baseline.
del frozen
del frozen_backend

structured = TransformersStructuredOutputAdapter(
    model_id=args.model,
    max_new_tokens=32,
)

structured_fit, structured_raw, structured_calibrated = run_calibrated_experiment(
    structured,
    calibration_cases=calibration_cases,
    test_cases=test_cases,
)

fallback_evaluation = evaluate_system2_fallback(
    frozen_calibrated,
    test_cases,
    structured,
    threshold=args.fallback_threshold,
)

calibration_provenance = dataset_provenance(
    args.calibration,
    calibration_cases,
)
test_provenance = dataset_provenance(
    args.test,
    test_cases,
)

payload = {
    "metadata": {
        "model": args.model,
        "calibration_dataset": calibration_provenance,
        "test_dataset": test_provenance,
        "warning": (
            "Synthetic benchmark data; model inference is real. "
            "Do not treat results as production-traffic quality."
        ),
    },
    "frozen_logits": {
        "temperature_fit": asdict(frozen_fit),
        "raw": asdict(frozen_raw),
        "calibrated": asdict(frozen_calibrated),
    },
    "structured_output": {
        "temperature_fit": asdict(structured_fit),
        "raw": asdict(structured_raw),
        "calibrated": asdict(structured_calibrated),
    },
    "system2_fallback": fallback_evaluation.artifact_payload(),
}

frozen_artifact_path = args.frozen_eval_artifact or args.eval_artifact
if frozen_artifact_path:
    frozen_artifact = build_eval_artifact(
        frozen_calibrated,
        decision_type="mcp_tool_router",
        dataset=test_provenance,
        model_ref=args.model,
        calibration=calibration_provenance,
        fallback_evaluation=fallback_evaluation.artifact_payload(),
    )
    Path(frozen_artifact_path).write_text(
        json.dumps(frozen_artifact, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

if args.structured_eval_artifact:
    structured_artifact = build_eval_artifact(
        structured_calibrated,
        decision_type="mcp_tool_router",
        dataset=test_provenance,
        model_ref=args.model,
        calibration=calibration_provenance,
    )
    Path(args.structured_eval_artifact).write_text(
        json.dumps(structured_artifact, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

Path(args.output).write_text(
    json.dumps(payload, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(
    json.dumps(
        {
            "frozen_accuracy": frozen_raw.accuracy,
            "structured_accuracy": structured_raw.accuracy,
            "fallback_case_count": fallback_evaluation.fallback_case_count,
            "fallback_accuracy": fallback_evaluation.accuracy,
            "fallback_p95_latency_ms": fallback_evaluation.p95_latency_ms,
            "output": args.output,
            "frozen_eval_artifact": frozen_artifact_path,
            "structured_eval_artifact": args.structured_eval_artifact,
        },
        indent=2,
    )
)
