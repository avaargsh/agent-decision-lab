from __future__ import annotations

import argparse
import json

from decision_lab.annotations import validate_routing_annotations
from decision_lab.benchmark import load_jsonl
from decision_lab.dataset_validation import validate_inventory_references
from decision_lab.inventory import load_inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate explicit MCP abstention benchmark cases."
    )
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--cases", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory = load_inventory(args.inventory)
    cases = load_jsonl(args.cases)

    if any(case.metadata.get("split") != "test" for case in cases):
        raise ValueError("abstention corpus must be test-only")

    validate_inventory_references(
        cases,
        inventory,
    )
    summary = validate_routing_annotations(
        cases,
        require_source_grounding=True,
    )

    payload = {
        "schema_version": "abstention-corpus-validation/v1",
        "inventory_id": inventory.inventory_id,
        "inventory_tool_count": len(inventory.tools),
        "case_count": summary.case_count,
        "expected_abstain_count": summary.expected_abstain_count,
        "coverage": summary.coverage_counts,
        "ambiguity": summary.ambiguity_counts,
        "risk": summary.risk_counts,
        "effect": summary.effect_counts,
        "no_gold_count": sum(
            1
            for case in cases
            if case.gold_candidate is None
        ),
        "missing_gold_identity_count": sum(
            1
            for case in cases
            if case.metadata.get("coverage") == "missing_candidate"
            and case.gold_candidate is not None
        ),
    }

    print(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
