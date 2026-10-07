from __future__ import annotations

import argparse
from pathlib import Path

from decision_lab.benchmark import dump_jsonl, load_jsonl
from decision_lab.inventory import load_inventory
from decision_lab.stress_suite import (
    build_candidate_count_suite,
    build_missing_candidate_suite,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic MCP candidate stress suites."
    )
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument(
        "--candidate-counts",
        default="5,10,20,50,100",
        help="comma-separated candidate counts",
    )
    parser.add_argument(
        "--strategy",
        choices=("random", "lexical"),
        default="random",
    )
    parser.add_argument(
        "--seed",
        default="agent-decision-benchmark-v0.2",
    )
    parser.add_argument("--covered-output", required=True)
    parser.add_argument("--missing-output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = [
        int(value.strip())
        for value in args.candidate_counts.split(",")
        if value.strip()
    ]
    inventory = load_inventory(args.inventory)
    cases = load_jsonl(args.base)

    covered = build_candidate_count_suite(
        cases,
        inventory,
        candidate_counts=counts,
        strategy=args.strategy,
        seed=args.seed,
    )
    Path(args.covered_output).parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    dump_jsonl(covered, args.covered_output)

    if args.missing_output:
        missing = build_missing_candidate_suite(
            cases,
            inventory,
            candidate_counts=counts,
            strategy=args.strategy,
            seed=args.seed,
        )
        Path(args.missing_output).parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        dump_jsonl(missing, args.missing_output)


if __name__ == "__main__":
    main()
