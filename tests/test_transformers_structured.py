import pytest

from decision_lab.adapters import structured_choice_scores
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



def test_explicit_choice_remains_top1_at_zero_confidence() -> None:
    candidates = [
        "prometheus.query",
        "logs.search",
        "kubernetes.get",
        "runbook.search",
    ]
    scores = structured_choice_scores(
        candidate="runbook.search",
        confidence=0.0,
        candidates=candidates,
    )

    assert sum(score.probability for score in scores) == pytest.approx(1.0)
    assert max(scores, key=lambda score: score.probability).candidate == (
        "runbook.search"
    )
    selected = next(
        score.probability
        for score in scores
        if score.candidate == "runbook.search"
    )
    others = [
        score.probability
        for score in scores
        if score.candidate != "runbook.search"
    ]
    assert all(selected > probability for probability in others)


def test_high_structured_confidence_is_preserved() -> None:
    scores = structured_choice_scores(
        candidate="logs.search",
        confidence=0.8,
        candidates=[
            "prometheus.query",
            "logs.search",
            "kubernetes.get",
            "runbook.search",
        ],
    )

    selected = next(
        score.probability
        for score in scores
        if score.candidate == "logs.search"
    )
    assert selected == pytest.approx(0.8)
