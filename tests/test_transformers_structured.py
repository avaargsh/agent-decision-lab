import pytest

from decision_lab.transformers_structured import (
    parse_structured_choice,
)


def test_parse_structured_choice() -> None:
    candidate, confidence = (
        parse_structured_choice(
            'prefix {"candidate":"read_logs","confidence":0.8} suffix',
            ["read_metrics", "read_logs"],
        )
    )

    assert candidate == "read_logs"
    assert confidence == pytest.approx(0.8)


def test_unknown_candidate_is_rejected() -> None:
    with pytest.raises(ValueError):
        parse_structured_choice(
            '{"candidate":"restart","confidence":0.9}',
            ["read_metrics", "read_logs"],
        )


def test_invalid_confidence_is_rejected() -> None:
    with pytest.raises(ValueError):
        parse_structured_choice(
            '{"candidate":"read_logs","confidence":2}',
            ["read_metrics", "read_logs"],
        )
