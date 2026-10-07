from __future__ import annotations

from decision_lab.annotations import validate_routing_annotations
from decision_lab.benchmark import BenchmarkCase


def _case(**metadata_overrides) -> BenchmarkCase:
    metadata = {
        "split": "test",
        "annotation_version": "mcp-routing-label/v1",
        "coverage": "covered",
        "ambiguity": "near_neighbor",
        "automation_risk": "low",
        "operation_effect": "read_only",
        "expected_abstain": False,
        "label_basis": "upstream_tool_contract",
        "gold_rationale": (
            "The upstream contract distinguishes an instant query from "
            "range-query and inventory alternatives."
        ),
        "evidence_refs": [
            {
                "repository": "awslabs/mcp",
                "revision": "abc",
                "path": "src/server/README.md",
            }
        ],
    }
    metadata.update(metadata_overrides)
    return BenchmarkCase(
        case_id="case-1",
        decision_type="mcp_tool_router",
        context={"intent": "query current target health"},
        candidates=["server::instant", "server::range"],
        gold_candidate="server::instant",
        metadata=metadata,
    )


def test_source_grounded_annotation_contract_accepts_valid_case() -> None:
    summary = validate_routing_annotations(
        [_case()],
        require_source_grounding=True,
    )

    assert summary.case_count == 1
    assert summary.covered_count == 1
    assert summary.ambiguity_counts == {"near_neighbor": 1}


def test_high_risk_requires_write_capable_effect() -> None:
    try:
        validate_routing_annotations(
            [
                _case(
                    automation_risk="high",
                    operation_effect="read_only",
                )
            ]
        )
    except ValueError as exc:
        assert "high automation risk" in str(exc)
    else:
        raise AssertionError("invalid high-risk annotation must fail")


def test_write_capable_cannot_be_low_risk() -> None:
    try:
        validate_routing_annotations(
            [_case(operation_effect="write_capable")]
        )
    except ValueError as exc:
        assert "cannot be low risk" in str(exc)
    else:
        raise AssertionError("low-risk write case must fail")


def test_missing_candidate_requires_abstention_and_omits_gold() -> None:
    case = _case(
        coverage="missing_candidate",
        expected_abstain=True,
    )
    case = BenchmarkCase(
        case_id=case.case_id,
        decision_type=case.decision_type,
        context=case.context,
        candidates=["server::range"],
        gold_candidate=case.gold_candidate,
        metadata=case.metadata,
    )

    summary = validate_routing_annotations([case])

    assert summary.expected_abstain_count == 1


def test_underspecified_case_requires_abstention() -> None:
    try:
        validate_routing_annotations(
            [_case(ambiguity="underspecified")]
        )
    except ValueError as exc:
        assert "must expect abstention" in str(exc)
    else:
        raise AssertionError("underspecified auto-execute case must fail")


def test_source_grounded_case_requires_evidence_refs() -> None:
    try:
        validate_routing_annotations(
            [_case(evidence_refs=[])],
            require_source_grounding=True,
        )
    except ValueError as exc:
        assert "evidence_refs" in str(exc)
    else:
        raise AssertionError("source-grounded case without evidence must fail")



def test_underspecified_abstention_uses_no_gold_and_plausible_candidates() -> None:
    base = _case(
        ambiguity="underspecified",
        expected_abstain=True,
        plausible_candidates=["server::instant", "server::range"],
    )
    case = BenchmarkCase(
        case_id=base.case_id,
        decision_type=base.decision_type,
        context=base.context,
        candidates=base.candidates,
        gold_candidate=None,
        metadata=base.metadata,
    )

    summary = validate_routing_annotations([case])

    assert summary.expected_abstain_count == 1


def test_unsupported_case_must_not_declare_gold() -> None:
    try:
        validate_routing_annotations(
            [
                _case(
                    coverage="unsupported",
                    expected_abstain=True,
                )
            ]
        )
    except ValueError as exc:
        assert "must not declare a gold" in str(exc)
    else:
        raise AssertionError("unsupported case with gold must fail")
