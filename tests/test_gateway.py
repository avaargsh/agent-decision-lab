from decision_lab import (
    CandidateScore,
    DecisionGateway,
    DecisionRequest,
    ThresholdPolicy,
)


def test_executes_high_confidence_decision() -> None:
    def scorer(request: DecisionRequest):
        values = {"a": 0.90, "b": 0.10}
        return [CandidateScore(c, values[c]) for c in request.candidates]

    gateway = DecisionGateway(scorer=scorer)
    result = gateway.decide(DecisionRequest("router", ["a", "b"]))

    assert result.action == "EXECUTE"
    assert result.decision is not None
    assert result.decision.candidate == "a"


def test_falls_back_on_low_confidence() -> None:
    def scorer(request: DecisionRequest):
        values = {"a": 0.60, "b": 0.40}
        return [CandidateScore(c, values[c]) for c in request.candidates]

    gateway = DecisionGateway(scorer=scorer)
    result = gateway.decide(DecisionRequest("router", ["a", "b"]))

    assert result.action == "FALLBACK"
    assert result.reason_code == "LOW_CONFIDENCE"


def test_falls_back_on_low_margin() -> None:
    def scorer(request: DecisionRequest):
        values = {"a": 0.88, "b": 0.12}
        return [CandidateScore(c, values[c]) for c in request.candidates]

    gateway = DecisionGateway(
        scorer=scorer,
        policy=ThresholdPolicy(execute_threshold=0.80, margin_threshold=0.80),
    )
    result = gateway.decide(DecisionRequest("router", ["a", "b"]))

    assert result.action == "FALLBACK"
    assert result.reason_code == "LOW_MARGIN"
