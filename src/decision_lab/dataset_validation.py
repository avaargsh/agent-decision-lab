from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

from .benchmark import BenchmarkCase
from .inventory import ToolInventory


@dataclass(frozen=True)
class SplitValidationReport:
    calibration_count: int
    test_count: int
    calibration_case_ids: tuple[str, ...]
    test_case_ids: tuple[str, ...]


def validate_calibration_test_pair(
    calibration_cases: Sequence[BenchmarkCase],
    test_cases: Sequence[BenchmarkCase],
    *,
    inventory: ToolInventory | None = None,
) -> SplitValidationReport:
    if not calibration_cases:
        raise ValueError("calibration_cases must not be empty")
    if not test_cases:
        raise ValueError("test_cases must not be empty")

    _validate_declared_split(calibration_cases, "calibration")
    _validate_declared_split(test_cases, "test")

    calibration_ids = [case.case_id for case in calibration_cases]
    test_ids = [case.case_id for case in test_cases]
    _validate_unique(calibration_ids, label="calibration case ids")
    _validate_unique(test_ids, label="test case ids")

    overlap = set(calibration_ids) & set(test_ids)
    if overlap:
        raise ValueError(
            "calibration/test case id overlap: "
            + ", ".join(sorted(overlap))
        )

    calibration_contexts = {
        _context_fingerprint(case)
        for case in calibration_cases
    }
    test_contexts = {
        _context_fingerprint(case)
        for case in test_cases
    }
    context_overlap = calibration_contexts & test_contexts
    if context_overlap:
        raise ValueError(
            "calibration/test context leakage detected"
        )

    if inventory is not None:
        validate_inventory_references(
            [*calibration_cases, *test_cases],
            inventory,
        )

    return SplitValidationReport(
        calibration_count=len(calibration_cases),
        test_count=len(test_cases),
        calibration_case_ids=tuple(calibration_ids),
        test_case_ids=tuple(test_ids),
    )


def _validate_declared_split(
    cases: Sequence[BenchmarkCase],
    expected: str,
) -> None:
    bad = [
        case.case_id
        for case in cases
        if case.metadata.get("split") != expected
    ]
    if bad:
        raise ValueError(
            f"cases with wrong split for {expected}: "
            + ", ".join(sorted(bad))
        )


def _validate_unique(
    values: Sequence[str],
    *,
    label: str,
) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")


def _context_fingerprint(case: BenchmarkCase) -> str:
    return json.dumps(
        {
            "decision_type": case.decision_type,
            "context": case.context,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )



def validate_inventory_references(
    cases: Sequence[BenchmarkCase],
    inventory: ToolInventory,
) -> None:
    inventory_names = {tool.name for tool in inventory.tools}

    for case in cases:
        missing = [
            candidate
            for candidate in case.candidates
            if candidate not in inventory_names
        ]
        if missing:
            raise ValueError(
                f"{case.case_id} candidates absent from inventory: "
                + ", ".join(sorted(missing))
            )

        coverage = case.metadata.get("coverage")
        if case.gold_candidate is not None:
            if case.gold_candidate not in inventory_names:
                raise ValueError(
                    f"{case.case_id} gold candidate absent from inventory: "
                    f"{case.gold_candidate}"
                )
        elif coverage not in {"unsupported", "covered"}:
            raise ValueError(
                f"{case.case_id} missing gold candidate is only valid for "
                "unsupported or ambiguous covered abstention cases"
            )
