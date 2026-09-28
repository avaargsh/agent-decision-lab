from decision_lab import CandidateScore, DecisionGateway
from decision_lab.service import DecisionService


def test_service_executes_confident_decision() -> None:
    def scorer(request):
        return [
            CandidateScore(request.candidates[0], 0.9),
            CandidateScore(request.candidates[1], 0.1),
        ]

    service = DecisionService(DecisionGateway(scorer=scorer))
    result = service.decide(
        {
            "decision_type": "router",
            "candidates": ["a", "b"],
            "context": {},
        }
    )

    assert result["action"] == "EXECUTE"
    assert result["decision"]["candidate"] == "a"
    assert result["ledger"]["selected_candidate"] == "a"


def test_service_calls_fallback() -> None:
    def scorer(request):
        return [
            CandidateScore(request.candidates[0], 0.6),
            CandidateScore(request.candidates[1], 0.4),
        ]

    service = DecisionService(
        DecisionGateway(scorer=scorer),
        fallback=lambda request: {"candidate": request.candidates[1]},
    )

    result = service.decide(
        {
            "decision_type": "router",
            "candidates": ["a", "b"],
        }
    )

    assert result["action"] == "FALLBACK"
    assert result["fallback"]["candidate"] == "b"
