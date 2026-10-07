from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Sequence

from .benchmark import BenchmarkCase


TRACE_PROVENANCE_VERSION = "trace-backed-routing/v1"

_ALLOWED_SOURCE_TYPES = {
    "production_trace",
    "incident_record",
    "user_study",
    "support_case",
}
_ALLOWED_TRANSFORMATIONS = {
    "verbatim_sanitized",
    "semantic_rewrite",
}
_ALLOWED_LABEL_SOURCES = {
    "observed_tool_selection",
    "human_adjudication",
    "upstream_contract_relabel",
}
_REQUIRED_SANITIZATION_FLAGS = (
    "pii_removed",
    "secrets_removed",
    "customer_identifiers_removed",
    "free_text_reviewed",
)
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class TraceCorpusSummary:
    case_count: int
    source_type_counts: dict[str, int]
    transformation_counts: dict[str, int]
    label_source_counts: dict[str, int]
    source_group_count: int
    verbatim_sanitized_count: int
    semantic_rewrite_count: int


@dataclass(frozen=True)
class TraceSplitIsolationReport:
    calibration_count: int
    test_count: int
    calibration_group_count: int
    test_group_count: int


def trace_case_fingerprint(case: BenchmarkCase) -> str:
    """Content fingerprint over the released, sanitized benchmark case."""
    payload = {
        "decision_type": case.decision_type,
        "context": case.context,
        "candidates": list(case.candidates),
        "gold_candidate": case.gold_candidate,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def validate_trace_backed_cases(
    cases: Sequence[BenchmarkCase],
) -> TraceCorpusSummary:
    if not cases:
        raise ValueError("cases must not be empty")

    source_types: dict[str, int] = {}
    transformations: dict[str, int] = {}
    label_sources: dict[str, int] = {}
    source_groups: set[str] = set()

    for case in cases:
        provenance = case.metadata.get("trace_provenance")
        if not isinstance(provenance, dict):
            raise ValueError(
                f"{case.case_id} trace_provenance must be an object"
            )

        if provenance.get("version") != TRACE_PROVENANCE_VERSION:
            raise ValueError(
                f"{case.case_id} trace provenance version must be "
                f"{TRACE_PROVENANCE_VERSION!r}"
            )

        source_type = _required_enum(
            case.case_id,
            provenance,
            "source_type",
            _ALLOWED_SOURCE_TYPES,
        )
        transformation = _required_enum(
            case.case_id,
            provenance,
            "transformation",
            _ALLOWED_TRANSFORMATIONS,
        )
        label_source = _required_enum(
            case.case_id,
            provenance,
            "label_source",
            _ALLOWED_LABEL_SOURCES,
        )

        source_artifact = _required_sha(
            case.case_id,
            provenance,
            "source_artifact_sha256",
        )
        if not source_artifact:
            raise AssertionError("unreachable")

        source_record = _required_sha(
            case.case_id,
            provenance,
            "source_record_sha256",
        )
        if not source_record:
            raise AssertionError("unreachable")

        source_group = _required_sha(
            case.case_id,
            provenance,
            "source_group_sha256",
        )
        source_groups.add(source_group)

        if provenance.get("identifier_pseudonymization") != "salted_sha256":
            raise ValueError(
                f"{case.case_id} identifier_pseudonymization must be "
                "'salted_sha256'"
            )

        declared_fingerprint = _required_sha(
            case.case_id,
            provenance,
            "released_case_sha256",
        )
        actual_fingerprint = trace_case_fingerprint(case)
        if declared_fingerprint != actual_fingerprint:
            raise ValueError(
                f"{case.case_id} released_case_sha256 mismatch: "
                f"declared {declared_fingerprint}, actual {actual_fingerprint}"
            )

        sanitization = provenance.get("sanitization")
        if not isinstance(sanitization, dict):
            raise ValueError(
                f"{case.case_id} sanitization must be an object"
            )
        for flag in _REQUIRED_SANITIZATION_FLAGS:
            if sanitization.get(flag) is not True:
                raise ValueError(
                    f"{case.case_id} sanitization.{flag} must be true"
                )

        if provenance.get("benchmark_release_approved") is not True:
            raise ValueError(
                f"{case.case_id} benchmark_release_approved must be true"
            )

        reviewer_count = provenance.get("reviewer_count")
        if (
            not isinstance(reviewer_count, int)
            or isinstance(reviewer_count, bool)
            or reviewer_count < 1
        ):
            raise ValueError(
                f"{case.case_id} reviewer_count must be an integer >= 1"
            )

        ambiguity = case.metadata.get("ambiguity")
        if ambiguity in {"underspecified", "multi_valid"} and reviewer_count < 2:
            raise ValueError(
                f"{case.case_id} ambiguous trace-backed labels require "
                "at least two reviewers"
            )

        _increment(source_types, source_type)
        _increment(transformations, transformation)
        _increment(label_sources, label_source)

    return TraceCorpusSummary(
        case_count=len(cases),
        source_type_counts=dict(sorted(source_types.items())),
        transformation_counts=dict(sorted(transformations.items())),
        label_source_counts=dict(sorted(label_sources.items())),
        source_group_count=len(source_groups),
        verbatim_sanitized_count=transformations.get(
            "verbatim_sanitized",
            0,
        ),
        semantic_rewrite_count=transformations.get(
            "semantic_rewrite",
            0,
        ),
    )


def validate_trace_split_isolation(
    calibration_cases: Sequence[BenchmarkCase],
    test_cases: Sequence[BenchmarkCase],
) -> TraceSplitIsolationReport:
    if not calibration_cases:
        raise ValueError("calibration_cases must not be empty")
    if not test_cases:
        raise ValueError("test_cases must not be empty")

    validate_trace_backed_cases(calibration_cases)
    validate_trace_backed_cases(test_cases)

    calibration_groups = {
        _trace_value(case, "source_group_sha256")
        for case in calibration_cases
    }
    test_groups = {
        _trace_value(case, "source_group_sha256")
        for case in test_cases
    }
    group_overlap = calibration_groups & test_groups
    if group_overlap:
        raise ValueError(
            "trace source-group leakage across calibration/test: "
            + ", ".join(sorted(group_overlap))
        )

    calibration_records = {
        _trace_value(case, "source_record_sha256")
        for case in calibration_cases
    }
    test_records = {
        _trace_value(case, "source_record_sha256")
        for case in test_cases
    }
    record_overlap = calibration_records & test_records
    if record_overlap:
        raise ValueError(
            "trace source-record leakage across calibration/test: "
            + ", ".join(sorted(record_overlap))
        )

    calibration_fingerprints = {
        trace_case_fingerprint(case)
        for case in calibration_cases
    }
    test_fingerprints = {
        trace_case_fingerprint(case)
        for case in test_cases
    }
    released_overlap = (
        calibration_fingerprints
        & test_fingerprints
    )
    if released_overlap:
        raise ValueError(
            "released benchmark-case leakage across calibration/test: "
            + ", ".join(sorted(released_overlap))
        )

    return TraceSplitIsolationReport(
        calibration_count=len(calibration_cases),
        test_count=len(test_cases),
        calibration_group_count=len(calibration_groups),
        test_group_count=len(test_groups),
    )


def build_trace_provenance(
    case: BenchmarkCase,
    *,
    source_type: str,
    transformation: str,
    label_source: str,
    source_artifact_sha256: str,
    source_record_sha256: str,
    source_group_sha256: str,
    reviewer_count: int,
    identifier_pseudonymization: str = "salted_sha256",
    benchmark_release_approved: bool,
    sanitization: dict[str, bool],
) -> dict[str, Any]:
    """Build provenance after the released case content is finalized."""
    provenance = {
        "version": TRACE_PROVENANCE_VERSION,
        "source_type": source_type,
        "transformation": transformation,
        "label_source": label_source,
        "source_artifact_sha256": source_artifact_sha256,
        "source_record_sha256": source_record_sha256,
        "source_group_sha256": source_group_sha256,
        "identifier_pseudonymization": identifier_pseudonymization,
        "released_case_sha256": trace_case_fingerprint(case),
        "reviewer_count": reviewer_count,
        "benchmark_release_approved": benchmark_release_approved,
        "sanitization": dict(sanitization),
    }
    # Reuse the validator semantics on a temporary case so callers cannot
    # construct a provenance object that this module would later reject.
    temporary = BenchmarkCase(
        case_id=case.case_id,
        decision_type=case.decision_type,
        context=case.context,
        candidates=case.candidates,
        gold_candidate=case.gold_candidate,
        metadata={
            **case.metadata,
            "trace_provenance": provenance,
        },
    )
    validate_trace_backed_cases([temporary])
    return provenance


def _trace_value(
    case: BenchmarkCase,
    key: str,
) -> str:
    provenance = case.metadata.get("trace_provenance")
    if not isinstance(provenance, dict):
        raise ValueError(
            f"{case.case_id} trace_provenance must be an object"
        )
    value = provenance.get(key)
    if not isinstance(value, str):
        raise ValueError(
            f"{case.case_id} trace_provenance.{key} must be a string"
        )
    return value


def _required_enum(
    case_id: str,
    provenance: dict[str, Any],
    key: str,
    allowed: set[str],
) -> str:
    value = provenance.get(key)
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(
            f"{case_id} trace_provenance.{key} must be one of "
            f"{sorted(allowed)}"
        )
    return value


def _required_sha(
    case_id: str,
    provenance: dict[str, Any],
    key: str,
) -> str:
    value = provenance.get(key)
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(
            f"{case_id} trace_provenance.{key} must be sha256:<64 lowercase hex>"
        )
    return value


def _increment(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1
