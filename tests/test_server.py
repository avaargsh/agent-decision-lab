from decision_lab.server import build_service

def test_demo_server_selects_execute_for_golden_action_gate():
    service = build_service("demo")
    result = service.decide({"decision_type": "escalation", "candidates": ["execute", "fallback", "human_review"], "context": {"action_kind": "scale", "target": "deployment/checkout-api"}})
    assert result["decision"]["candidate"] == "execute"
    assert result["decision"]["confidence"] == 0.93

def test_qwen_mode_fails_closed_until_serving_adapter_is_promoted():
    try:
        build_service("qwen")
    except RuntimeError as exc:
        assert "configured model adapter" in str(exc)
    else:
        raise AssertionError("qwen mode must not silently fall back to demo scorer")
