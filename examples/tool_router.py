from decision_lab import (
    CandidateScore,
    DecisionGateway,
    DecisionRequest,
    ThresholdPolicy,
)


def demo_scorer(request: DecisionRequest) -> list[CandidateScore]:
    scores = {
        "read_metrics": 0.90,
        "read_logs": 0.07,
        "restart_workload": 0.03,
    }
    return [CandidateScore(c, scores[c]) for c in request.candidates]


gateway = DecisionGateway(
    scorer=demo_scorer,
    policy=ThresholdPolicy(execute_threshold=0.85, margin_threshold=0.20),
)

result = gateway.decide(
    DecisionRequest(
        decision_type="mcp_tool_router",
        candidates=["read_metrics", "read_logs", "restart_workload"],
        context={"intent": "check current CPU saturation"},
    )
)

print(result)
