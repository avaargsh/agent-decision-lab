from decision_lab.server import build_service


def payload():
    return {
        "decision_type": "escalation",
        "candidates": ["execute", "fallback", "human_review"],
        "context": {"evidence_digest": "sha256:" + "a" * 64, "target": "deployment/checkout"},
    }


def test_decision_id_is_stable_for_replay_of_same_request_and_result():
    service = build_service("demo")
    first = service.decide(payload())
    second = service.decide(payload())
    assert first["decision_id"] == second["decision_id"]
    assert first["ledger"]["decision_id"] == first["decision_id"]


def test_decision_id_changes_when_frozen_evidence_changes():
    service = build_service("demo")
    first = service.decide(payload())
    changed = payload()
    changed["context"]["evidence_digest"] = "sha256:" + "b" * 64
    second = service.decide(changed)
    assert first["decision_id"] != second["decision_id"]
