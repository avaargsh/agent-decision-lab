from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict
from pathlib import Path

from decision_lab.benchmark import load_jsonl
from decision_lab.dataset_validation import validate_calibration_test_pair
from decision_lab.inventory import inventory_digest, load_inventory
from decision_lab.logits import (
    FrozenLogitAdapter,
    compact_candidate_prompt,
)
from decision_lab.provenance import dataset_provenance
from decision_lab.stress_experiment import (
    run_calibrated_stress_experiment,
)
from decision_lab.transformers_backend import TransformersCausalLMBackend
from decision_lab.transformers_structured import (
    TransformersStructuredOutputAdapter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run M5 real-identity Qwen stress matrix."
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--base-test", required=True)
    parser.add_argument("--covered", required=True)
    parser.add_argument("--missing", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--risk-budget", type=float, default=0.5)
    parser.add_argument("--min-coverage", type=float, default=0.25)
    parser.add_argument(
        "--permutation-cases-per-k",
        type=int,
        default=2,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    inventory = load_inventory(args.inventory)
    calibration_cases = load_jsonl(args.calibration)
    base_test_cases = load_jsonl(args.base_test)
    covered_cases = load_jsonl(args.covered)
    missing_cases = load_jsonl(args.missing)

    validate_calibration_test_pair(
        calibration_cases,
        base_test_cases,
        inventory=inventory,
    )

    frozen_backend = TransformersCausalLMBackend(
        model_id=args.model,
        length_normalize=True,
    )
    frozen = FrozenLogitAdapter(
        backend=frozen_backend,
        prompt_builder=compact_candidate_prompt,
        name="frozen-logits-compact",
    )
    frozen_report = run_calibrated_stress_experiment(
        frozen,
        calibration_cases=calibration_cases,
        base_test_cases=base_test_cases,
        covered_stress_cases=covered_cases,
        missing_stress_cases=missing_cases,
        risk_budget=args.risk_budget,
        min_coverage=args.min_coverage,
        permutation_cases_per_k=args.permutation_cases_per_k,
    )

    del frozen
    del frozen_backend
    gc.collect()

    structured = TransformersStructuredOutputAdapter(
        model_id=args.model,
        max_new_tokens=48,
    )
    structured_report = run_calibrated_stress_experiment(
        structured,
        calibration_cases=calibration_cases,
        base_test_cases=base_test_cases,
        covered_stress_cases=covered_cases,
        missing_stress_cases=missing_cases,
        risk_budget=args.risk_budget,
        min_coverage=args.min_coverage,
        permutation_cases_per_k=args.permutation_cases_per_k,
    )

    payload = {
        "schema_version": "m5-model-stress/v1",
        "metadata": {
            "model": args.model,
            "inventory_id": inventory.inventory_id,
            "inventory_sha256": inventory_digest(inventory),
            "inventory_tool_count": len(inventory.tools),
            "stress_strategy": "random",
            "frozen_prompt_variant": "compact-candidate-v1",
            "risk_budget": args.risk_budget,
            "min_coverage": args.min_coverage,
            "permutation_cases_per_k": args.permutation_cases_per_k,
            "warning": (
                "Synthetic intents over real upstream MCP tool identities. "
                "This is controlled benchmark evidence, not production traffic."
            ),
        },
        "datasets": {
            "calibration": dataset_provenance(
                args.calibration,
                calibration_cases,
            ),
            "base_test": dataset_provenance(
                args.base_test,
                base_test_cases,
            ),
            "covered_stress": dataset_provenance(
                args.covered,
                covered_cases,
            ),
            "missing_stress": dataset_provenance(
                args.missing,
                missing_cases,
            ),
        },
        "frozen_logits": asdict(frozen_report),
        "structured_output": asdict(structured_report),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output": str(output_path),
                "frozen_threshold": frozen_report.transfer_threshold,
                "structured_threshold": structured_report.transfer_threshold,
                "candidate_counts": [
                    item.candidate_count
                    for item in frozen_report.by_candidate_count
                ],
                "frozen_base_accuracy": frozen_report.base_test.accuracy,
                "structured_base_accuracy": structured_report.base_test.accuracy,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
