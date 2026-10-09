from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict
from pathlib import Path

from decision_lab.annotations import validate_routing_annotations
from decision_lab.benchmark import load_jsonl
from decision_lab.dataset_validation import (
    validate_calibration_test_pair,
    validate_inventory_references,
)
from decision_lab.inventory import inventory_digest, load_inventory
from decision_lab.model_snapshot import freeze_model_snapshot
from decision_lab.logits import (
    FrozenLogitAdapter,
    compact_candidate_prompt,
)
from decision_lab.prior_correction import (
    PriorCorrectedFrozenLogitAdapter,
)
from decision_lab.provenance import dataset_provenance
from decision_lab.replay_profile import build_replay_calibration_profile
from decision_lab.quality_experiment import (
    run_calibrated_quality_experiment,
)
from decision_lab.stress_experiment import (
    run_calibrated_stress_experiment,
)
from decision_lab.transformers_backend import TransformersCausalLMBackend
from decision_lab.transformers_structured import (
    TransformersStructuredOutputAdapter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the combined M5 candidate-scaling and M6 "
            "source-grounded decision replay."
        )
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-revision", required=True,
                        help="Immutable 40-hex Hugging Face commit SHA")
    parser.add_argument("--inventory", required=True)

    parser.add_argument("--m5-calibration", required=True)
    parser.add_argument("--m5-test", required=True)
    parser.add_argument("--m5-covered", required=True)
    parser.add_argument("--m5-missing", required=True)

    parser.add_argument("--m6-calibration", required=True)
    parser.add_argument("--m6-test", required=True)
    parser.add_argument("--m6-abstention", required=True)

    parser.add_argument("--output", required=True)
    parser.add_argument("--risk-budget", type=float, default=0.5)
    parser.add_argument("--min-coverage", type=float, default=0.25)
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--prior-strength",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--permutation-cases-per-k",
        type=int,
        default=2,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Resolve exactly once: all three arms load the same local HF snapshot,
    # rather than resolving the mutable remote model independently.
    snapshot_path, snapshot_evidence = freeze_model_snapshot(
        args.model, args.model_revision
    )
    model_ref = snapshot_evidence["model_ref"]
    evidence_path = Path(args.output).parent / "model-snapshot.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(snapshot_evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    inventory = load_inventory(args.inventory)

    m5_calibration = load_jsonl(args.m5_calibration)
    m5_test = load_jsonl(args.m5_test)
    m5_covered = load_jsonl(args.m5_covered)
    m5_missing = load_jsonl(args.m5_missing)

    m6_calibration = load_jsonl(args.m6_calibration)
    m6_test = load_jsonl(args.m6_test)
    m6_abstention = load_jsonl(args.m6_abstention)

    validate_calibration_test_pair(
        m5_calibration,
        m5_test,
        inventory=inventory,
    )
    validate_calibration_test_pair(
        m6_calibration,
        m6_test,
        inventory=inventory,
    )
    validate_inventory_references(
        m6_abstention,
        inventory,
    )
    validate_routing_annotations(
        m6_calibration,
        require_source_grounding=True,
    )
    validate_routing_annotations(
        m6_test,
        require_source_grounding=True,
    )
    validate_routing_annotations(
        m6_abstention,
        require_source_grounding=True,
    )

    frozen_backend = TransformersCausalLMBackend(
        model_id=str(snapshot_path),
        length_normalize=True,
        candidate_batch_size=args.candidate_batch_size,
    )
    frozen = FrozenLogitAdapter(
        backend=frozen_backend,
        prompt_builder=compact_candidate_prompt,
        name="frozen-logits-compact",
    )
    frozen_m5 = _run_m5(
        frozen,
        args,
        m5_calibration,
        m5_test,
        m5_covered,
        m5_missing,
    )
    frozen_m6 = _run_m6(
        frozen,
        args,
        m6_calibration,
        m6_test,
        m6_abstention,
    )

    prior_corrected = PriorCorrectedFrozenLogitAdapter(
        backend=frozen_backend,
        prompt_builder=compact_candidate_prompt,
        prior_strength=args.prior_strength,
        name="prior-corrected-frozen-logits",
    )
    prior_m5 = _run_m5(
        prior_corrected,
        args,
        m5_calibration,
        m5_test,
        m5_covered,
        m5_missing,
    )
    prior_m6 = _run_m6(
        prior_corrected,
        args,
        m6_calibration,
        m6_test,
        m6_abstention,
    )

    del prior_corrected
    del frozen
    del frozen_backend
    gc.collect()

    structured = TransformersStructuredOutputAdapter(
        model_id=str(snapshot_path),
        max_new_tokens=48,
    )
    structured_m5 = _run_m5(
        structured,
        args,
        m5_calibration,
        m5_test,
        m5_covered,
        m5_missing,
    )
    structured_m6 = _run_m6(
        structured,
        args,
        m6_calibration,
        m6_test,
        m6_abstention,
    )

    # Produced only from the measured M5/M6 calibration reports. The
    # protocols deliberately retain different fitted thresholds and digests.
    replay_profiles = {
        "schema_version": "decision-gpu-calibration-profiles/v1",
    }
    inventory_ref = "sha256:" + inventory_digest(inventory)
    for protocol, calibration_path, calibration_cases, test_cases, arms in (
        (
            "m5_candidate_scaling",
            args.m5_calibration,
            m5_calibration,
            m5_test,
            {
                "frozen_logits": frozen_m5,
                "prior_corrected_logits": prior_m5,
                "structured_output": structured_m5,
            },
        ),
        (
            "m6_source_grounded_quality",
            args.m6_calibration,
            m6_calibration,
            m6_test,
            {
                "frozen_logits": frozen_m6,
                "prior_corrected_logits": prior_m6,
                "structured_output": structured_m6,
            },
        ),
    ):
        replay_profiles[protocol] = {
            arm: build_replay_calibration_profile(
                report,
                calibration_path=calibration_path,
                calibration_cases=calibration_cases,
                test_cases=test_cases,
                model_ref=model_ref,
                inventory_sha256=inventory_ref,
            )
            for arm, report in arms.items()
        }

    payload = {
        "schema_version": "decision-gpu-replay/v1",
        "calibration_profiles": replay_profiles,
        "metadata": {
            "model": args.model,
            "model_ref": model_ref,
            "model_revision": args.model_revision,
            "model_snapshot_digest": snapshot_evidence["content_digest"],
            "inventory_id": inventory.inventory_id,
            "inventory_sha256": inventory_digest(inventory),
            "inventory_tool_count": len(inventory.tools),
            "frozen_prompt_variant": "compact-candidate-v1",
            "prior_correction_variant": "candidate-prior-subtraction-v1",
            "prior_strength": args.prior_strength,
            "candidate_batch_size": args.candidate_batch_size,
            "risk_budget": args.risk_budget,
            "min_coverage": args.min_coverage,
            "permutation_cases_per_k": args.permutation_cases_per_k,
            "protocols": {
                "m5_candidate_scaling": (
                    "v2 synthetic intents + real tool identities; "
                    "K=5/10/20/50/100 random distractors"
                ),
                "m6_source_grounded_quality": (
                    "v3 source-grounded authored near-neighbour closed-set "
                    "+ explicit no-gold abstention corpus"
                ),
            },
            "warning": (
                "These are benchmark results over authored requests, not "
                "production traffic. Routing confidence does not grant "
                "execution authority."
            ),
        },
        "datasets": {
            "m5": {
                "calibration": dataset_provenance(
                    args.m5_calibration,
                    m5_calibration,
                ),
                "test": dataset_provenance(
                    args.m5_test,
                    m5_test,
                ),
                "covered_stress": dataset_provenance(
                    args.m5_covered,
                    m5_covered,
                ),
                "missing_stress": dataset_provenance(
                    args.m5_missing,
                    m5_missing,
                ),
            },
            "m6": {
                "calibration": dataset_provenance(
                    args.m6_calibration,
                    m6_calibration,
                ),
                "test": dataset_provenance(
                    args.m6_test,
                    m6_test,
                ),
                "abstention": dataset_provenance(
                    args.m6_abstention,
                    m6_abstention,
                ),
            },
        },
        "m5_candidate_scaling": {
            "frozen_logits": asdict(frozen_m5),
            "prior_corrected_logits": asdict(prior_m5),
            "structured_output": asdict(structured_m5),
        },
        "m6_source_grounded_quality": {
            "frozen_logits": asdict(frozen_m6),
            "prior_corrected_logits": asdict(prior_m6),
            "structured_output": asdict(structured_m6),
        },
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
                "m5_candidate_counts": [
                    item.candidate_count
                    for item in frozen_m5.by_candidate_count
                ],
                "m6_thresholds": {
                    "frozen": frozen_m6.transfer_threshold,
                    "prior_corrected": prior_m6.transfer_threshold,
                    "structured": structured_m6.transfer_threshold,
                },
                "m6_overall_far": {
                    "frozen": frozen_m6.abstention.false_accept_rate,
                    "prior_corrected": (
                        prior_m6.abstention.false_accept_rate
                    ),
                    "structured": (
                        structured_m6.abstention.false_accept_rate
                    ),
                },
            },
            indent=2,
        )
    )


def _run_m5(
    adapter,
    args,
    calibration,
    test,
    covered,
    missing,
):
    return run_calibrated_stress_experiment(
        adapter,
        calibration_cases=calibration,
        base_test_cases=test,
        covered_stress_cases=covered,
        missing_stress_cases=missing,
        risk_budget=args.risk_budget,
        min_coverage=args.min_coverage,
        permutation_cases_per_k=args.permutation_cases_per_k,
    )


def _run_m6(
    adapter,
    args,
    calibration,
    test,
    abstention,
):
    return run_calibrated_quality_experiment(
        adapter,
        calibration_cases=calibration,
        test_cases=test,
        abstention_cases=abstention,
        risk_budget=args.risk_budget,
        min_coverage=args.min_coverage,
    )


if __name__ == "__main__":
    main()
