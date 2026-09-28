from decision_lab.logits import FrozenLogitAdapter, default_candidate_prompt
from decision_lab.models import DecisionRequest


class FakeBackend:
    name = "fake"

    def logprob(self, *, prompt: str, candidate: str) -> float:
        values = {
            "read_metrics": -0.1,
            "read_logs": -2.0,
            "restart_workload": -5.0,
        }
        return values[candidate]


def test_frozen_logit_adapter_normalizes_candidates() -> None:
    request = DecisionRequest(
        decision_type="mcp_tool_router",
        candidates=["read_metrics", "read_logs", "restart_workload"],
        context={"intent": "check CPU"},
    )
    scores = FrozenLogitAdapter(
        backend=FakeBackend(),
        prompt_builder=default_candidate_prompt,
    ).score(request)

    assert abs(sum(item.probability for item in scores) - 1.0) < 1e-9
    assert max(scores, key=lambda item: item.probability).candidate == "read_metrics"


def test_default_prompt_contains_bounded_candidates() -> None:
    request = DecisionRequest(
        decision_type="severity",
        candidates=["sev1", "sev2"],
        context={"signal": "regional outage"},
    )
    prompt = default_candidate_prompt(request)

    assert "Decision type: severity" in prompt
    assert "sev1, sev2" in prompt
    assert "regional outage" in prompt
