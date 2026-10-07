from __future__ import annotations

import argparse
import json

from decision_lab.annotations import validate_routing_annotations
from decision_lab.benchmark import load_jsonl
from decision_lab.dataset_validation import validate_calibration_test_pair
from decision_lab.inventory import load_inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a source-grounded MCP routing corpus."
    )
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--test", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    inventory = load_inventory(args.inventory)
    calibration = load_jsonl(args.calibration)
    test = load_jsonl(args.test)

    split = validate_calibration_test_pair(
        calibration,
        test,
        inventory=inventory,
    )
    cal_annotations = validate_routing_annotations(
        calibration,
        require_source_grounding=True,
    )
    test_annotations = validate_routing_annotations(
        test,
        require_source_grounding=True,
    )

    payload = {
        "schema_version": "source-grounded-corpus-validation/v1",
        "inventory_id": inventory.inventory_id,
        "inventory_tool_count": len(inventory.tools),
        "calibration": {
            "case_count": split.calibration_count,
            "annotations": {
                "ambiguity": cal_annotations.ambiguity_counts,
                "risk": cal_annotations.risk_counts,
                "effect": cal_annotations.effect_counts,
                "expected_abstain": cal_annotations.expected_abstain_count,
            },
        },
        "test": {
            "case_count": split.test_count,
            "annotations": {
                "ambiguity": test_annotations.ambiguity_counts,
                "risk": test_annotations.risk_counts,
                "effect": test_annotations.effect_counts,
                "expected_abstain": test_annotations.expected_abstain_count,
            },
        },
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
