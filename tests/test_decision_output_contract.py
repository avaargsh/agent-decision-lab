import math

import pytest

from decision_lab.decision_output import (
    build_decision_output,
    validate_decision_output,
)


def test_choice_requires_candidate_membership():
    output = build_decision_output(
        kind="choice", value="metrics", confidence=0.8,
        candidates=["metrics", "logs"],
    )
    assert validate_decision_output(output, candidates=["metrics", "logs"])
    assert not validate_decision_output(output, candidates=["logs"])
    with pytest.raises(ValueError, match="originating candidates"):
        build_decision_output(kind="choice", value="metrics", confidence=0.8)


def test_boolean_and_score_types_are_distinct():
    assert build_decision_output(
        kind="boolean", value=False, confidence=0.75,
    )["value"] is False
    assert build_decision_output(
        kind="score", value=0.0, confidence=0.9,
    )["value"] == 0.0
    with pytest.raises(ValueError, match="invalid typed"):
        build_decision_output(kind="boolean", value=0, confidence=0.8)
    with pytest.raises(ValueError, match="invalid typed"):
        build_decision_output(kind="score", value=True, confidence=0.8)


@pytest.mark.parametrize("invalid_value", [-0.1, 1.1, math.inf, math.nan])
def test_score_rejects_invalid_values(invalid_value):
    with pytest.raises(ValueError, match="invalid typed"):
        build_decision_output(
            kind="score", value=invalid_value, confidence=0.8,
        )


def test_abstention_is_never_boolean_false_or_score_zero():
    for kind in ("choice", "boolean", "score"):
        output = build_decision_output(
            kind=kind, value=None, confidence=None,
        )
        assert output["abstain"] is True
        assert validate_decision_output(output)
        assert not validate_decision_output({
            **output, "confidence": 0.5,
        })
        assert not validate_decision_output({
            **output, "abstain": False,
        })


def test_rejects_extra_fields_wrong_version_and_invalid_confidence():
    payload = build_decision_output(
        kind="choice", value="a", confidence=1, candidates=["a"],
    )
    assert not validate_decision_output({**payload, "authorized": True})
    assert not validate_decision_output({
        **payload, "schema_version": "decision-output/v2",
    })
    assert not validate_decision_output({**payload, "confidence": math.nan})
    assert not validate_decision_output({**payload, "kind": "noul"})
