from __future__ import annotations

import argparse
import json

from decision_lab.annotations import validate_routing_annotations
from decision_lab.benchmark import load_jsonl
from decision_lab.dataset_validation import validate_calibration_test_pair
from decision_lab.inventory import load_inventory
from decision_lab.trace_corpus import (
    validate_trace_backed_cases,
    validate_trace_split_isolation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a trace-backed MCP routing corpus without reading "
            "or publishing the raw trace source."
        )
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
        require_source_grounding=False,
    )
    test_annotations = validate_routing_annotations(
        test,
        require_source_grounding=False,
    )
    cal_trace = validate_trace_backed_cases(
        calibration
    )
    test_trace = validate_trace_backed_cases(
        test
    )
    isolation = validate_trace_split_isolation(
        calibration,
        test,
    )

    if any(
        bool(case.metadata.get("expected_abstain", False))
        for case in calibration
    ):
        raise ValueError(
            "trace-backed calibration must not contain expected-abstain cases"
        )
    if any(
        case.gold_candidate is None
        for case in calibration
    ):
        raise ValueError(
            "trace-backed calibration must contain single-gold cases only"
        )

    payload = {
        "schema_version": "trace-backed-corpus-validation/v1",
        "inventory_id": inventory.inventory_id,
        "inventory_tool_count": len(inventory.tools),
        "calibration": {
            "case_count": split.calibration_count,
            "source_groups": isolation.calibration_group_count,
            "source_types": cal_trace.source_type_counts,
            "transformations": cal_trace.transformation_counts,
            "label_sources": cal_trace.label_source_counts,
            "coverage": cal_annotations.coverage_counts,
            "ambiguity": cal_annotations.ambiguity_counts,
            "risk": cal_annotations.risk_counts,
        },
        "test": {
            "case_count": split.test_count,
            "source_groups": isolation.test_group_count,
            "source_types": test_trace.source_type_counts,
            "transformations": test_trace.transformation_counts,
            "label_sources": test_trace.label_source_counts,
            "coverage": test_annotations.coverage_counts,
            "ambiguity": test_annotations.ambiguity_counts,
            "risk": test_annotations.risk_counts,
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
