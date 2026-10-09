import hashlib
import json
import math

import pytest

from decision_lab.calibration_profile import (
    build_calibration_profile,
    build_profile_bound_gateway,
    verify_calibration_profile,
)
from decision_lab.models import CandidateScore, DecisionRequest


DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64


def make_profile(**overrides):
    options = {
        "adapter": "frozen-logits",
        "model_ref": "Qwen/Qwen3-0.6B@revision",
        "decision_type": "mcp_tool_router",
        "calibration": {
            "sha256": DIGEST_A,
            "case_count": 2,
            "case_ids": ["cal-1", "cal-2"],
        },
        "temperature": 2.0,
        "execute_threshold": 0.7,
        "margin_threshold": 0.0,
        "risk_budget": 0.1,
        "threshold_selection": "calibration_risk_budget",
        "inventory_sha256": DIGEST_B,
    }
    options.update(overrides)
    return build_calibration_profile(**options)


def reseal(profile):
    payload = {
        k: v for k, v in profile.items()
        if k not in {"profile_id", "content_digest"}
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    digest = "sha256:" + hashlib.sha256(canonical).hexdigest()
    profile["content_digest"] = digest
    profile["profile_id"] = "calibration-profile:" + digest


def scorer(request):
    return [
        CandidateScore(request.candidates[0], 0.8),
        CandidateScore(request.candidates[1], 0.2),
    ]


def gateway(profile, **overrides):
    kwargs = {
        "adapter": "frozen-logits",
        "model_ref": "Qwen/Qwen3-0.6B@revision",
        "decision_type": "mcp_tool_router",
        "inventory_sha256": DIGEST_B,
    }
    kwargs.update(overrides)
    return build_profile_bound_gateway(
        scorer, profile=profile, **kwargs,
    )


def test_sealed_profile_can_drive_confidence_gate():
    profile = make_profile()
    assert verify_calibration_profile(profile)
    result = gateway(profile).decide(
        DecisionRequest("mcp_tool_router", ["metrics", "logs"])
    )
    # Raw top1=0.8; T=2 reduces it to 2/3, below profile threshold 0.7.
    assert result.action == "FALLBACK"
    assert result.reason_code == "LOW_CONFIDENCE"
    assert result.scores[0].probability == pytest.approx(2 / 3)
    assert result.decision is None


def test_profile_snapshot_is_not_changed_by_mutating_caller_dict():
    profile = make_profile()
    bound = gateway(profile)
    profile["temperature"] = 0.1
    result = bound.decide(
        DecisionRequest("mcp_tool_router", ["metrics", "logs"])
    )
    assert result.action == "FALLBACK"


@pytest.mark.parametrize("field, value", [
    ("adapter", "other-adapter"),
    ("model_ref", "another-model"),
    ("decision_type", "policy_gate"),
    ("inventory_sha256", None),
])
def test_gateway_rejects_profile_identity_mismatch(field, value):
    with pytest.raises(ValueError, match="mismatch"):
        gateway(make_profile(), **{field: value})


def test_gateway_rejects_calibration_test_overlap_and_task_drift():
    with pytest.raises(ValueError, match="overlap"):
        gateway(make_profile(), test_case_ids=["test-1", "cal-2"])
    with pytest.raises(ValueError, match="outside"):
        gateway(make_profile()).decide(
            DecisionRequest("severity", ["metrics", "logs"])
        )


def test_tampering_detected_even_if_resealed_with_bad_semantics():
    profile = make_profile()
    profile["temperature"] = -0.1
    reseal(profile)
    assert not verify_calibration_profile(profile)
    with pytest.raises(ValueError, match="invalid calibration profile"):
        gateway(profile)


@pytest.mark.parametrize("temperature", [0.0, -1, math.inf, math.nan, True])
def test_profile_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError):
        make_profile(temperature=temperature)


def test_profile_rejects_bad_provenance_and_selection():
    with pytest.raises(ValueError):
        make_profile(calibration={
            "sha256": DIGEST_A, "case_count": 2,
            "case_ids": ["cal-1", "cal-1"],
        })
    with pytest.raises(ValueError):
        make_profile(threshold_selection="calibration_risk_budget", risk_budget=None)
    with pytest.raises(ValueError):
        make_profile(inventory_sha256="not-a-digest")


def test_gateway_validates_raw_scores_before_calibration():
    bad_scorer = lambda request: [
        CandidateScore("metrics", 0.8),
        CandidateScore("logs", 0.1),
    ]
    bound = build_profile_bound_gateway(
        bad_scorer,
        profile=make_profile(),
        adapter="frozen-logits",
        model_ref="Qwen/Qwen3-0.6B@revision",
        decision_type="mcp_tool_router",
        inventory_sha256=DIGEST_B,
    )
    with pytest.raises(ValueError, match="sum to 1.0"):
        bound.decide(DecisionRequest("mcp_tool_router", ["metrics", "logs"]))
