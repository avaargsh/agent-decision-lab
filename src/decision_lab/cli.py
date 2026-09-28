from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .adapters import MappingScoreAdapter
from .benchmark import load_jsonl
from .runner import run_benchmark


def _demo_mapping(request):
    candidates = list(request.candidates)
    scores = {candidate: 1.0 / len(candidates) for candidate in candidates}

    intent = str(request.context.get("intent", "")).lower()
    preferred = None

    if "metric" in intent or "cpu" in intent:
        preferred = "read_metrics"
    elif "log" in intent or "error" in intent:
        preferred = "read_logs"

    if preferred in scores and len(candidates) > 1:
        remainder = 0.1 / (len(candidates) - 1)
        scores = {candidate: remainder for candidate in candidates}
        scores[preferred] = 0.9

    return scores


def main() -> None:
    parser = argparse.ArgumentParser(prog="decision-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bench = subparsers.add_parser("benchmark")
    bench.add_argument("path")
    bench.add_argument(
        "--adapter",
        choices=["demo"],
        default="demo",
    )

    args = parser.parse_args()

    if args.command == "benchmark":
        cases = load_jsonl(args.path)
        report = run_benchmark(
            MappingScoreAdapter(_demo_mapping, name="demo"),
            cases,
        )
        print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
