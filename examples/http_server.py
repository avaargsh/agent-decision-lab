from decision_lab import CandidateScore, DecisionGateway
from decision_lab.http import serve
from decision_lab.service import DecisionService


def scorer(request):
    if request.context.get("intent") == "metrics":
        values = {
            "read_metrics": 0.92,
            "read_logs": 0.06,
            "restart_workload": 0.02,
        }
    else:
        values = {
            "read_metrics": 0.45,
            "read_logs": 0.40,
            "restart_workload": 0.15,
        }

    return [CandidateScore(candidate, values[candidate]) for candidate in request.candidates]


def fallback(request):
    return {
        "provider": "system-2-placeholder",
        "status": "not-executed",
        "reason": "low-confidence bounded decision",
        "decision_type": request.decision_type,
    }


service = DecisionService(
    DecisionGateway(scorer=scorer),
    fallback=fallback,
)

serve(service)
