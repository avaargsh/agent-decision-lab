from __future__ import annotations

import hashlib

from decision_lab.benchmark import BenchmarkCase
from decision_lab.trace_corpus import (
    build_trace_provenance,
    trace_case_fingerprint,
    validate_trace_backed_cases,
    validate_trace_split_isolation,
)


def _sha(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _case(
    case_id: str,
    *,
    split: str,
    intent: str,
    group: str,
    record: str,
    transformation: str = "verbatim_sanitized",
    ambiguity: str = "near_neighbor",
    reviewer_count: int = 1,
) -> BenchmarkCase:
    base = BenchmarkCase(
        case_id=case_id,
        decision_type="mcp_tool_router",
        context={"intent": intent},
        candidates=["server::a", "server::b"],
        gold_candidate="server::a",
        metadata={
            "split": split,
            "ambiguity": ambiguity,
        },
    )
    provenance = build_trace_provenance(
        base,
        source_type="production_trace",
        transformation=transformation,
        label_source="human_adjudication",
        source_artifact_sha256=_sha("artifact"),
        source_record_sha256=_sha(record),
        source_group_sha256=_sha(group),
        reviewer_count=reviewer_count,
        benchmark_release_approved=True,
        sanitization={
            "pii_removed": True,
            "secrets_removed": True,
            "customer_identifiers_removed": True,
            "free_text_reviewed": True,
        },
    )
    return BenchmarkCase(
        case_id=base.case_id,
        decision_type=base.decision_type,
        context=base.context,
        candidates=base.candidates,
        gold_candidate=base.gold_candidate,
        metadata={
            **base.metadata,
            "trace_provenance": provenance,
        },
    )


def test_trace_backed_case_requires_content_bound_provenance() -> None:
    case = _case(
        "c1",
        split="calibration",
        intent="query current latency",
        group="incident-1",
        record="event-1",
    )

    summary = validate_trace_backed_cases([case])

    assert summary.case_count == 1
    assert summary.verbatim_sanitized_count == 1
    assert (
        case.metadata["trace_provenance"]["released_case_sha256"]
        == trace_case_fingerprint(case)
    )


def test_semantic_rewrite_stays_distinct_from_verbatim_trace() -> None:
    case = _case(
        "c1",
        split="test",
        intent="rewritten operational request",
        group="incident-1",
        record="event-1",
        transformation="semantic_rewrite",
    )

    summary = validate_trace_backed_cases([case])

    assert summary.verbatim_sanitized_count == 0
    assert summary.semantic_rewrite_count == 1


def test_trace_case_fingerprint_detects_released_content_change() -> None:
    case = _case(
        "c1",
        split="test",
        intent="original",
        group="incident-1",
        record="event-1",
    )
    mutated = BenchmarkCase(
        case_id=case.case_id,
        decision_type=case.decision_type,
        context={"intent": "changed after provenance was created"},
        candidates=case.candidates,
        gold_candidate=case.gold_candidate,
        metadata=case.metadata,
    )

    try:
        validate_trace_backed_cases([mutated])
    except ValueError as exc:
        assert "released_case_sha256 mismatch" in str(exc)
    else:
        raise AssertionError("content mutation must invalidate provenance")


def test_trace_split_rejects_same_source_group_across_splits() -> None:
    calibration = [
        _case(
            "c1",
            split="calibration",
            intent="calibration",
            group="same-incident",
            record="event-1",
        )
    ]
    test = [
        _case(
            "t1",
            split="test",
            intent="test",
            group="same-incident",
            record="event-2",
        )
    ]

    try:
        validate_trace_split_isolation(
            calibration,
            test,
        )
    except ValueError as exc:
        assert "source-group leakage" in str(exc)
    else:
        raise AssertionError("same incident/session must not cross splits")


def test_trace_split_rejects_same_source_record_across_splits() -> None:
    calibration = [
        _case(
            "c1",
            split="calibration",
            intent="calibration",
            group="incident-a",
            record="same-event",
        )
    ]
    test = [
        _case(
            "t1",
            split="test",
            intent="test",
            group="incident-b",
            record="same-event",
        )
    ]

    try:
        validate_trace_split_isolation(
            calibration,
            test,
        )
    except ValueError as exc:
        assert "source-record leakage" in str(exc)
    else:
        raise AssertionError("same source record must not cross splits")


def test_trace_case_requires_all_sanitization_declarations() -> None:
    base = BenchmarkCase(
        case_id="c1",
        decision_type="router",
        context={"intent": "safe released text"},
        candidates=["a", "b"],
        gold_candidate="a",
        metadata={"ambiguity": "clear"},
    )

    try:
        build_trace_provenance(
            base,
            source_type="production_trace",
            transformation="verbatim_sanitized",
            label_source="observed_tool_selection",
            source_artifact_sha256=_sha("artifact"),
            source_record_sha256=_sha("record"),
            source_group_sha256=_sha("group"),
            reviewer_count=1,
            benchmark_release_approved=True,
            sanitization={
                "pii_removed": True,
                "secrets_removed": False,
                "customer_identifiers_removed": True,
                "free_text_reviewed": True,
            },
        )
    except ValueError as exc:
        assert "secrets_removed" in str(exc)
    else:
        raise AssertionError("unsanitized trace record must fail closed")


def test_ambiguous_trace_label_requires_two_reviewers() -> None:
    try:
        _case(
            "c1",
            split="test",
            intent="ambiguous request",
            group="incident",
            record="record",
            ambiguity="multi_valid",
            reviewer_count=1,
        )
    except ValueError as exc:
        assert "at least two reviewers" in str(exc)
    else:
        raise AssertionError(
            "ambiguous trace-backed label needs independent review"
        )


def test_distinct_source_groups_pass_split_isolation() -> None:
    report = validate_trace_split_isolation(
        [
            _case(
                "c1",
                split="calibration",
                intent="calibration",
                group="incident-a",
                record="event-a",
            )
        ],
        [
            _case(
                "t1",
                split="test",
                intent="test",
                group="incident-b",
                record="event-b",
            )
        ],
    )

    assert report.calibration_group_count == 1
    assert report.test_group_count == 1
