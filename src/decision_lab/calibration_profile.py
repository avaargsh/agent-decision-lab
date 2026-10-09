"""Content-addressed calibration profiles for opt-in choice gateways."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .calibration import TemperatureCalibrator
from .gateway import DecisionGateway
from .models import CandidateScore, DecisionRequest
from .policy import ThresholdPolicy

SCHEMA_VERSION = "calibration-profile/v1"
FIELDS = {
    "schema_version", "profile_id", "content_digest",
    "adapter", "model_ref", "decision_type", "candidate_schema_version",
    "inventory_sha256", "calibration", "method", "temperature",
    "execute_threshold", "margin_threshold", "risk_budget",
    "threshold_selection",
}
SPLIT_FIELDS = {"sha256", "case_count", "case_ids"}
SELECTIONS = {"calibration_risk_budget", "calibration_median_confidence_fallback"}


def _digest_valid(value: object) -> bool:
    return isinstance(value, str) and bool(
        re.fullmatch(r"sha256:[0-9a-f]{64}", value)
    )


def _probability(value: object) -> bool:
    return (
        type(value) in (int, float)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def _validate_semantics(profile: Mapping[str, Any]) -> bool:
    if not isinstance(profile, Mapping) or set(profile) != FIELDS:
        return False
    if profile["schema_version"] != SCHEMA_VERSION:
        return False
    if not all(
        isinstance(profile[field], str) and profile[field]
        for field in ("adapter", "model_ref", "decision_type")
    ):
        return False
    if profile["candidate_schema_version"] != "candidate-set/v1":
        return False
    inventory = profile["inventory_sha256"]
    if inventory is not None and not _digest_valid(inventory):
        return False
    split = profile["calibration"]
    if not isinstance(split, Mapping) or set(split) != SPLIT_FIELDS:
        return False
    if not _digest_valid(split["sha256"]):
        return False
    case_ids = split["case_ids"]
    if (
        not isinstance(case_ids, list)
        or not case_ids
        or any(not isinstance(case_id, str) or not case_id for case_id in case_ids)
        or len(set(case_ids)) != len(case_ids)
        or type(split["case_count"]) is not int
        or split["case_count"] != len(case_ids)
    ):
        return False
    if profile["method"] != "temperature_scaling":
        return False
    temperature = profile["temperature"]
    if (
        type(temperature) not in (int, float)
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        return False
    if not all(
        _probability(profile[field])
        for field in ("execute_threshold", "margin_threshold")
    ):
        return False
    budget = profile["risk_budget"]
    if budget is not None and not _probability(budget):
        return False
    selection = profile["threshold_selection"]
    if selection not in SELECTIONS:
        return False
    if selection == "calibration_risk_budget" and budget is None:
        return False
    return True


def verify_calibration_profile(profile: Mapping[str, Any]) -> bool:
    """Check shape, semantics, and content identity. Not proof of fit quality."""
    try:
        if not _validate_semantics(profile):
            return False
        payload = {
            key: value for key, value in profile.items()
            if key not in {"profile_id", "content_digest"}
        }
        digest = "sha256:" + hashlib.sha256(_canonical(payload)).hexdigest()
        return (
            profile["content_digest"] == digest
            and profile["profile_id"] == "calibration-profile:" + digest
        )
    except (ValueError, TypeError, KeyError, OverflowError):
        return False


def build_calibration_profile(
    *,
    adapter: str,
    model_ref: str,
    decision_type: str,
    calibration: Mapping[str, Any],
    temperature: float,
    execute_threshold: float,
    margin_threshold: float = 0.0,
    risk_budget: float | None = None,
    threshold_selection: str = "calibration_median_confidence_fallback",
    inventory_sha256: str | None = None,
) -> dict[str, Any]:
    """Seal fitted settings and calibration lineage without observing test data."""
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "adapter": adapter,
        "model_ref": model_ref,
        "decision_type": decision_type,
        "candidate_schema_version": "candidate-set/v1",
        "inventory_sha256": inventory_sha256,
        "calibration": dict(calibration),
        "method": "temperature_scaling",
        "temperature": temperature,
        "execute_threshold": execute_threshold,
        "margin_threshold": margin_threshold,
        "risk_budget": risk_budget,
        "threshold_selection": threshold_selection,
    }
    if not _validate_semantics({
        **payload, "profile_id": "", "content_digest": "",
    }):
        raise ValueError("invalid calibration profile fields")
    digest = "sha256:" + hashlib.sha256(_canonical(payload)).hexdigest()
    profile = {
        **payload,
        "content_digest": digest,
        "profile_id": "calibration-profile:" + digest,
    }
    if not verify_calibration_profile(profile):
        raise ValueError("calibration profile verification failed")
    return profile


def build_profile_bound_gateway(
    scorer: Callable[[DecisionRequest], Sequence[CandidateScore]],
    *,
    profile: Mapping[str, Any],
    adapter: str,
    model_ref: str,
    decision_type: str,
    inventory_sha256: str | None = None,
    test_case_ids: Sequence[str] = (),
) -> DecisionGateway:
    """Opt-in choice gateway; reject mismatched model or calibration provenance.

    The caller must attest the actual adapter/model/inventory identity. A
    verified profile does not authorize any downstream tool execution.
    """
    if not verify_calibration_profile(profile):
        raise ValueError("invalid calibration profile")
    for field, actual in (
        ("adapter", adapter),
        ("model_ref", model_ref),
        ("decision_type", decision_type),
        ("inventory_sha256", inventory_sha256),
    ):
        if profile[field] != actual:
            raise ValueError(f"calibration profile {field} mismatch")
    if set(profile["calibration"]["case_ids"]).intersection(test_case_ids):
        raise ValueError("calibration/test case overlap")

    # Capture verified scalar values, not the caller's mutable profile object.
    expected_type = profile["decision_type"]
    calibrator = TemperatureCalibrator(profile["temperature"])
    policy = ThresholdPolicy(
        execute_threshold=profile["execute_threshold"],
        margin_threshold=profile["margin_threshold"],
    )

    def calibrated_scorer(request: DecisionRequest) -> Sequence[CandidateScore]:
        if request.decision_type != expected_type:
            raise ValueError("decision_type outside calibration profile scope")
        raw = list(scorer(request))
        DecisionGateway._validate_scores(request, raw)
        if len(raw) != len(request.candidates):
            raise ValueError("scorer must return exactly one score per candidate")
        return calibrator.calibrate(raw)

    return DecisionGateway(scorer=calibrated_scorer, policy=policy)
