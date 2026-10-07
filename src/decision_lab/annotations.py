from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

from .benchmark import BenchmarkCase


ANNOTATION_VERSION = "mcp-routing-label/v1"

CoverageLabel = Literal["covered", "missing_candidate", "unsupported"]
AmbiguityLabel = Literal[
    "clear",
    "near_neighbor",
    "underspecified",
    "multi_valid",
]
AutomationRisk = Literal["low", "medium", "high"]
OperationEffect = Literal[
    "read_only",
    "sensitive_read",
    "write_capable",
    "unknown",
]


@dataclass(frozen=True)
class AnnotationSummary:
    case_count: int
    covered_count: int
    expected_abstain_count: int
    coverage_counts: dict[str, int]
    ambiguity_counts: dict[str, int]
    risk_counts: dict[str, int]
    effect_counts: dict[str, int]


def validate_routing_annotations(
    cases: Sequence[BenchmarkCase],
    *,
    require_source_grounding: bool = False,
) -> AnnotationSummary:
    if not cases:
        raise ValueError("cases must not be empty")

    coverage_counts: dict[str, int] = {}
    ambiguity_counts: dict[str, int] = {}
    risk_counts: dict[str, int] = {}
    effect_counts: dict[str, int] = {}
    covered_count = 0
    expected_abstain_count = 0

    for case in cases:
        metadata = case.metadata

        if metadata.get("annotation_version") != ANNOTATION_VERSION:
            raise ValueError(
                f"{case.case_id} annotation_version must be "
                f"{ANNOTATION_VERSION!r}"
            )

        coverage = _one_of(
            case.case_id,
            "coverage",
            metadata.get("coverage"),
            {"covered", "missing_candidate", "unsupported"},
        )
        ambiguity = _one_of(
            case.case_id,
            "ambiguity",
            metadata.get("ambiguity"),
            {
                "clear",
                "near_neighbor",
                "underspecified",
                "multi_valid",
            },
        )
        risk = _one_of(
            case.case_id,
            "automation_risk",
            metadata.get("automation_risk"),
            {"low", "medium", "high"},
        )
        effect = _one_of(
            case.case_id,
            "operation_effect",
            metadata.get("operation_effect"),
            {
                "read_only",
                "sensitive_read",
                "write_capable",
                "unknown",
            },
        )

        expected_abstain = metadata.get("expected_abstain")
        if not isinstance(expected_abstain, bool):
            raise ValueError(
                f"{case.case_id} expected_abstain must be boolean"
            )

        rationale = metadata.get("gold_rationale")
        if not isinstance(rationale, str) or len(rationale.strip()) < 20:
            raise ValueError(
                f"{case.case_id} gold_rationale must be a substantive string"
            )

        label_basis = metadata.get("label_basis")
        if not isinstance(label_basis, str) or not label_basis.strip():
            raise ValueError(
                f"{case.case_id} label_basis must not be empty"
            )

        if require_source_grounding:
            if label_basis != "upstream_tool_contract":
                raise ValueError(
                    f"{case.case_id} must use upstream_tool_contract label_basis"
                )
            _validate_evidence_refs(case)

        if coverage == "covered":
            covered_count += 1
            if expected_abstain and ambiguity in {"underspecified", "multi_valid"}:
                if case.gold_candidate is not None:
                    raise ValueError(
                        f"{case.case_id} ambiguous abstention case must not declare a single gold"
                    )
                _validate_plausible_candidates(case)
            else:
                if case.gold_candidate is None:
                    raise ValueError(
                        f"{case.case_id} covered single-gold case requires gold candidate"
                    )
                if case.gold_candidate not in case.candidates:
                    raise ValueError(
                        f"{case.case_id} covered case must contain gold candidate"
                    )
        elif coverage == "missing_candidate":
            if case.gold_candidate is None:
                raise ValueError(
                    f"{case.case_id} missing_candidate requires the omitted gold identity"
                )
            if case.gold_candidate in case.candidates:
                raise ValueError(
                    f"{case.case_id} missing_candidate case must omit gold"
                )
            if not expected_abstain:
                raise ValueError(
                    f"{case.case_id} missing_candidate must expect abstention"
                )
        else:
            if case.gold_candidate is not None:
                raise ValueError(
                    f"{case.case_id} unsupported case must not declare a gold candidate"
                )
            if not expected_abstain:
                raise ValueError(
                    f"{case.case_id} unsupported case must expect abstention"
                )

        if ambiguity in {"underspecified", "multi_valid"} and not expected_abstain:
            raise ValueError(
                f"{case.case_id} {ambiguity} case must expect abstention"
            )

        if risk == "high" and effect != "write_capable":
            raise ValueError(
                f"{case.case_id} high automation risk requires write_capable effect"
            )

        if effect == "write_capable" and risk == "low":
            raise ValueError(
                f"{case.case_id} write_capable effect cannot be low risk"
            )

        if expected_abstain:
            expected_abstain_count += 1

        _increment(coverage_counts, coverage)
        _increment(ambiguity_counts, ambiguity)
        _increment(risk_counts, risk)
        _increment(effect_counts, effect)

    return AnnotationSummary(
        case_count=len(cases),
        covered_count=covered_count,
        expected_abstain_count=expected_abstain_count,
        coverage_counts=dict(sorted(coverage_counts.items())),
        ambiguity_counts=dict(sorted(ambiguity_counts.items())),
        risk_counts=dict(sorted(risk_counts.items())),
        effect_counts=dict(sorted(effect_counts.items())),
    )


def _validate_plausible_candidates(case: BenchmarkCase) -> None:
    plausible = case.metadata.get("plausible_candidates")
    if not isinstance(plausible, list) or len(plausible) < 2:
        raise ValueError(
            f"{case.case_id} ambiguous abstention case requires at least two plausible_candidates"
        )
    if any(
        not isinstance(candidate, str)
        or candidate not in case.candidates
        for candidate in plausible
    ):
        raise ValueError(
            f"{case.case_id} plausible_candidates must be candidate identities"
        )
    if len(plausible) != len(set(plausible)):
        raise ValueError(
            f"{case.case_id} plausible_candidates must be unique"
        )


def _validate_evidence_refs(case: BenchmarkCase) -> None:
    refs = case.metadata.get("evidence_refs")
    if not isinstance(refs, list) or not refs:
        raise ValueError(
            f"{case.case_id} evidence_refs must be a non-empty list"
        )

    for index, ref in enumerate(refs):
        if not isinstance(ref, dict):
            raise ValueError(
                f"{case.case_id} evidence_refs[{index}] must be an object"
            )
        for key in ("repository", "revision", "path"):
            value = ref.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{case.case_id} evidence_refs[{index}].{key} must be set"
                )


def _one_of(
    case_id: str,
    field: str,
    value: object,
    allowed: Iterable[str],
) -> str:
    allowed_set = set(allowed)
    if not isinstance(value, str) or value not in allowed_set:
        raise ValueError(
            f"{case_id} {field} must be one of {sorted(allowed_set)}"
        )
    return value


def _increment(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1
