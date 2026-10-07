from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from decision_lab.benchmark import dump_jsonl, load_jsonl
from decision_lab.dataset_validation import validate_calibration_test_pair
from decision_lab.inventory import inventory_digest, load_inventory
from decision_lab.stress_suite import (
    build_candidate_count_suite,
    build_missing_candidate_suite,
)


SCHEMA_VERSION = "m5-stress-suite/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build M5 stress artifacts from a frozen MCP inventory."
    )
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--candidate-counts",
        default="5,10,20,50,100",
    )
    parser.add_argument(
        "--strategies",
        default="random,lexical",
    )
    parser.add_argument(
        "--seed",
        default="agent-decision-benchmark-v0.2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = tuple(
        int(value.strip())
        for value in args.candidate_counts.split(",")
        if value.strip()
    )
    strategies = tuple(
        value.strip()
        for value in args.strategies.split(",")
        if value.strip()
    )

    inventory = load_inventory(args.inventory)
    calibration_cases = load_jsonl(args.calibration)
    test_cases = load_jsonl(args.test)

    split_report = validate_calibration_test_pair(
        calibration_cases,
        test_cases,
        inventory=inventory,
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "seed": args.seed,
        "candidate_counts": list(counts),
        "strategies": list(strategies),
        "inventory": {
            "inventory_id": inventory.inventory_id,
            "sha256": inventory_digest(inventory),
            "tool_count": len(inventory.tools),
        },
        "base": {
            "calibration": _file_identity(
                Path(args.calibration),
                split_report.calibration_count,
            ),
            "test": _file_identity(
                Path(args.test),
                split_report.test_count,
            ),
        },
        "generated": [],
    }

    for split_name, cases in (
        ("calibration", calibration_cases),
        ("test", test_cases),
    ):
        for strategy in strategies:
            covered = build_candidate_count_suite(
                cases,
                inventory,
                candidate_counts=counts,
                strategy=strategy,
                seed=args.seed,
            )
            missing = build_missing_candidate_suite(
                cases,
                inventory,
                candidate_counts=counts,
                strategy=strategy,
                seed=args.seed,
            )

            for coverage, generated_cases in (
                ("covered", covered),
                ("missing", missing),
            ):
                filename = (
                    f"{split_name}.{strategy}.{coverage}.jsonl"
                )
                path = output_dir / filename
                dump_jsonl(generated_cases, path)
                manifest["generated"].append(
                    {
                        "file": filename,
                        "split": split_name,
                        "strategy": strategy,
                        "coverage": coverage,
                        "case_count": len(generated_cases),
                        "sha256": _sha256(path),
                    }
                )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"inventory={inventory.inventory_id} "
        f"tools={len(inventory.tools)} "
        f"generated_files={len(manifest['generated'])}"
    )


def _file_identity(
    path: Path,
    case_count: int,
) -> dict[str, object]:
    return {
        "file": path.name,
        "case_count": case_count,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
