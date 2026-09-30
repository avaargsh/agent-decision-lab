from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import math
from typing import Any, Mapping

from .runner import BenchmarkReport


SCHEMA_VERSION = "decision-eval/v1"


def _is_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return (
        ordered[lower] * (1.0 - weight)
        + ordered[upper] * weight
    )


_FALLBACK_FIELDS = {
    "measured",
    "adapter",
    "threshold",
    "eligible_case_count",
    "fallback_case_count",
    "fallback_rate",
    "accuracy",
    "p50_latency_ms",
    "p95_latency_ms",
    "mean_tokens_processed",
    "parse_valid_rate",
    "cases",
}
_FALLBACK_CASE_FIELDS = {
    "case_id",
    "fast_confidence",
    "fast_predicted",
    "fallback_predicted",
    "gold",
    "correct",
    "latency_ms",
    "tokens_processed",
    "parse_valid",
}


def _fallback_semantics_error(
    fallback: Mapping[str, Any],
    *,
    dataset_case_ids: list[str],
) -> str | None:
    if not set(fallback).issubset(_FALLBACK_FIELDS):
        return "fallback contains unsupported fields"

    measured = fallback.get("measured")
    if not isinstance(measured, bool):
        return "fallback.measured must be boolean"
    if not measured:
        return None

    adapter = fallback.get("adapter")
    if not isinstance(adapter, str) or not adapter:
        return "fallback.adapter is required"

    threshold = fallback.get("threshold")
    if not _is_number(threshold) or not 0.0 <= float(threshold) <= 1.0:
        return "fallback.threshold must be in [0, 1]"

    eligible = fallback.get("eligible_case_count")
    if isinstance(eligible, bool) or not isinstance(eligible, int):
        return "fallback.eligible_case_count must be an integer"
    if eligible != len(dataset_case_ids):
        return "fallback.eligible_case_count must match dataset case_count"

    fallback_count = fallback.get("fallback_case_count")
    if isinstance(fallback_count, bool) or not isinstance(
        fallback_count,
        int,
    ):
        return "fallback.fallback_case_count must be an integer"
    if not 0 <= fallback_count <= eligible:
        return "fallback.fallback_case_count is out of range"

    rate = fallback.get("fallback_rate")
    if not _is_number(rate):
        return "fallback.fallback_rate must be numeric"
    expected_rate = (
        fallback_count / eligible
        if eligible
        else 0.0
    )
    if not math.isclose(
        float(rate),
        expected_rate,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        return "fallback.fallback_rate does not match case counts"

    cases = fallback.get("cases")
    if not isinstance(cases, list):
        return "fallback.cases must be an array"
    if len(cases) != fallback_count:
        return "fallback.cases length does not match fallback_case_count"

    dataset_ids = set(dataset_case_ids)
    seen: set[str] = set()
    latencies: list[float] = []
    token_counts: list[int] = []
    correct_count = 0
    parse_values: list[bool] = []

    for case in cases:
        if not isinstance(case, Mapping):
            return "fallback case must be an object"
        if not set(case).issubset(_FALLBACK_CASE_FIELDS):
            return "fallback case contains unsupported fields"
        case_id = case.get("case_id")
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id not in dataset_ids
        ):
            return "fallback case_id must belong to dataset"
        if case_id in seen:
            return "fallback case_id must be unique"
        seen.add(case_id)

        confidence = case.get("fast_confidence")
        if (
            not _is_number(confidence)
            or not 0.0 <= float(confidence) <= 1.0
        ):
            return "fallback fast_confidence must be in [0, 1]"

        for field in (
            "fast_predicted",
            "fallback_predicted",
            "gold",
        ):
            value = case.get(field)
            if not isinstance(value, str) or not value:
                return f"fallback {field} is required"

        correct = case.get("correct")
        if not isinstance(correct, bool):
            return "fallback correct must be boolean"
        correct_count += int(correct)

        latency = case.get("latency_ms")
        if not _is_number(latency) or float(latency) < 0:
            return "fallback latency_ms must be non-negative"
        latencies.append(float(latency))

        tokens = case.get("tokens_processed")
        if tokens is not None:
            if (
                isinstance(tokens, bool)
                or not isinstance(tokens, int)
                or tokens < 0
            ):
                return "fallback tokens_processed must be a non-negative integer"
            token_counts.append(tokens)

        parse_valid = case.get("parse_valid")
        if parse_valid is not None:
            if not isinstance(parse_valid, bool):
                return "fallback parse_valid must be boolean"
            parse_values.append(parse_valid)

    accuracy = fallback.get("accuracy")
    p50 = fallback.get("p50_latency_ms")
    p95 = fallback.get("p95_latency_ms")
    mean_tokens = fallback.get("mean_tokens_processed")

    if fallback_count == 0:
        if any(
            value is not None
            for value in (accuracy, p50, p95, mean_tokens)
        ):
            return "empty fallback measurements must be null"
    else:
        if not _is_number(accuracy):
            return "fallback accuracy is required"
        expected_accuracy = correct_count / fallback_count
        if not math.isclose(
            float(accuracy),
            expected_accuracy,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            return "fallback accuracy does not match case outcomes"

        if not _is_number(p50) or not _is_number(p95):
            return "fallback latency percentiles are required"
        if not math.isclose(
            float(p50),
            _percentile(latencies, 0.5),
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            return "fallback p50 latency does not match cases"
        if not math.isclose(
            float(p95),
            _percentile(latencies, 0.95),
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            return "fallback p95 latency does not match cases"

        expected_tokens = (
            sum(token_counts) / len(token_counts)
            if token_counts
            else None
        )
        if expected_tokens is None:
            if mean_tokens is not None:
                return "fallback mean token count must be null"
        else:
            if not _is_number(mean_tokens) or not math.isclose(
                float(mean_tokens),
                expected_tokens,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                return "fallback mean token count does not match cases"

    parse_rate = fallback.get("parse_valid_rate")
    if parse_rate is not None:
        if (
            not _is_number(parse_rate)
            or not 0.0 <= float(parse_rate) <= 1.0
        ):
            return "fallback parse_valid_rate must be in [0, 1]"
        if not parse_values:
            return "fallback parse_valid_rate requires parse_valid cases"
        expected_parse_rate = (
            sum(parse_values) / len(parse_values)
        )
        if not math.isclose(
            float(parse_rate),
            expected_parse_rate,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            return "fallback parse_valid_rate does not match cases"

    return None


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(
        char in "0123456789abcdef"
        for char in digest
    )


_METRIC_FIELDS = {
    "accuracy",
    "macro_f1",
    "nll",
    "brier",
    "ece",
    "mean_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "mean_tokens_processed_per_decision",
}
_OPERATING_FIELDS = {
    "threshold",
    "coverage",
    "risk",
    "false_automation_rate",
    "fallback_rate",
    "risk_budget",
}
_TOP_LEVEL_PAYLOAD_FIELDS = {
    "schema_version",
    "decision_type",
    "adapter",
    "model_ref",
    "dataset",
    "calibration_sha256",
    "metrics",
    "operating_point",
    "fallback_evaluation",
}


def _artifact_semantics_error(
    payload: Mapping[str, Any],
) -> str | None:
    if set(payload) != _TOP_LEVEL_PAYLOAD_FIELDS:
        return "artifact payload fields do not match decision-eval/v1"

    if payload.get("schema_version") != SCHEMA_VERSION:
        return "schema_version mismatch"
    for field in ("decision_type", "adapter"):
        value = payload.get(field)
        if not isinstance(value, str) or not value:
            return f"{field} must be a non-empty string"

    model_ref = payload.get("model_ref")
    if model_ref is not None and not isinstance(model_ref, str):
        return "model_ref must be a string or null"

    calibration = payload.get("calibration_sha256")
    if calibration is not None and not _is_sha256(calibration):
        return "calibration_sha256 must be a sha256 digest or null"

    dataset = payload.get("dataset")
    if not isinstance(dataset, Mapping):
        return "dataset must be an object"
    if set(dataset) != {"sha256", "case_count", "case_ids"}:
        return "dataset fields do not match decision-eval/v1"
    if not _is_sha256(dataset.get("sha256")):
        return "dataset.sha256 must be a sha256 digest"
    case_ids = dataset.get("case_ids")
    case_count = dataset.get("case_count")
    if (
        not isinstance(case_ids, list)
        or not case_ids
        or not all(
            isinstance(item, str) and item
            for item in case_ids
        )
        or len(set(case_ids)) != len(case_ids)
        or isinstance(case_count, bool)
        or not isinstance(case_count, int)
        or case_count != len(case_ids)
    ):
        return "dataset case provenance is invalid"

    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != _METRIC_FIELDS:
        return "metrics fields do not match decision-eval/v1"

    unit_interval = {"accuracy", "macro_f1", "ece"}
    non_negative = {
        "nll",
        "brier",
        "mean_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
    }
    for name in unit_interval:
        value = metrics.get(name)
        if (
            not _is_number(value)
            or not 0.0 <= float(value) <= 1.0
        ):
            return f"metrics.{name} must be in [0, 1]"
    for name in non_negative:
        value = metrics.get(name)
        if not _is_number(value) or float(value) < 0:
            return f"metrics.{name} must be non-negative"
    token_mean = metrics.get("mean_tokens_processed_per_decision")
    if token_mean is not None and (
        not _is_number(token_mean)
        or float(token_mean) < 0
    ):
        return (
            "metrics.mean_tokens_processed_per_decision "
            "must be non-negative or null"
        )

    operating = payload.get("operating_point")
    if operating is not None:
        if (
            not isinstance(operating, Mapping)
            or set(operating) != _OPERATING_FIELDS
        ):
            return "operating_point fields do not match decision-eval/v1"
        for name in (
            "threshold",
            "coverage",
            "risk",
            "false_automation_rate",
            "fallback_rate",
        ):
            value = operating.get(name)
            if (
                not _is_number(value)
                or not 0.0 <= float(value) <= 1.0
            ):
                return f"operating_point.{name} must be in [0, 1]"
        risk_budget = operating.get("risk_budget")
        if risk_budget is not None and (
            not _is_number(risk_budget)
            or not 0.0 <= float(risk_budget) <= 1.0
        ):
            return "operating_point.risk_budget must be in [0, 1] or null"

    fallback = payload.get("fallback_evaluation")
    if not isinstance(fallback, Mapping):
        return "fallback_evaluation must be an object"
    return _fallback_semantics_error(
        fallback,
        dataset_case_ids=case_ids,
    )


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def build_eval_artifact(
    report: BenchmarkReport,
    *,
    decision_type: str,
    dataset: Mapping[str, Any],
    model_ref: str | None = None,
    calibration_sha256: str | None = None,
    fallback_evaluation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a content-addressed benchmark artifact for release/eval gates.

    The artifact is intentionally provider-neutral. It carries enough provenance
    for a control plane to decide whether a measured bounded-decision result is
    acceptable without importing this package or rereading a mutable dataset.
    """
    if not decision_type:
        raise ValueError("decision_type must not be empty")

    dataset_sha256 = dataset.get("sha256")
    if not _is_sha256(dataset_sha256):
        raise ValueError("dataset.sha256 must be a sha256: digest")

    if calibration_sha256 is not None and not _is_sha256(
        calibration_sha256
    ):
        raise ValueError(
            "calibration_sha256 must be a sha256: digest or null"
        )

    case_ids = list(dataset.get("case_ids", []))
    case_count = int(dataset.get("case_count", len(case_ids)))
    report_case_ids = [case.case_id for case in report.cases]

    if (
        not case_ids
        or len(set(case_ids)) != len(case_ids)
        or not all(
            isinstance(item, str) and item
            for item in case_ids
        )
    ):
        raise ValueError("dataset case_ids must be non-empty and unique")
    if case_count != len(case_ids):
        raise ValueError("dataset case_count does not match case_ids")
    if report_case_ids and report_case_ids != case_ids:
        raise ValueError("benchmark report cases do not match dataset case_ids")

    operating = report.operating_point
    fallback = dict(fallback_evaluation or {})
    fallback.setdefault("measured", False)
    fallback_error = _fallback_semantics_error(
        fallback,
        dataset_case_ids=case_ids,
    )
    if fallback_error is not None:
        raise ValueError(fallback_error)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "decision_type": decision_type,
        "adapter": report.adapter,
        "model_ref": model_ref,
        "dataset": {
            "sha256": dataset_sha256,
            "case_count": case_count,
            "case_ids": case_ids,
        },
        "calibration_sha256": calibration_sha256,
        "metrics": {
            "accuracy": report.accuracy,
            "macro_f1": report.macro_f1,
            "nll": report.nll,
            "brier": report.brier,
            "ece": report.ece,
            "mean_latency_ms": report.mean_latency_ms,
            "p50_latency_ms": report.p50_latency_ms,
            "p95_latency_ms": report.p95_latency_ms,
            "mean_tokens_processed_per_decision": (
                report.mean_tokens_processed_per_decision
            ),
        },
        "operating_point": (
            {
                **asdict(operating),
                "risk_budget": report.risk_budget,
            }
            if operating is not None
            else None
        ),
        "fallback_evaluation": fallback,
    }

    semantic_error = _artifact_semantics_error(payload)
    if semantic_error is not None:
        raise ValueError(semantic_error)

    digest = "sha256:" + sha256(_canonical_json(payload)).hexdigest()
    return {
        **payload,
        "artifact_id": f"decision-eval:{digest}",
        "content_digest": digest,
    }


def verify_eval_artifact(artifact: Mapping[str, Any]) -> bool:
    if artifact.get("schema_version") != SCHEMA_VERSION:
        return False

    expected = artifact.get("content_digest")
    artifact_id = artifact.get("artifact_id")
    if not isinstance(expected, str) or not expected.startswith("sha256:"):
        return False
    if artifact_id != f"decision-eval:{expected}":
        return False

    payload = {
        key: value
        for key, value in artifact.items()
        if key not in {"artifact_id", "content_digest"}
    }
    actual = "sha256:" + sha256(_canonical_json(payload)).hexdigest()
    if actual != expected:
        return False

    if set(artifact) != (
        _TOP_LEVEL_PAYLOAD_FIELDS
        | {"artifact_id", "content_digest"}
    ):
        return False

    semantic_error = _artifact_semantics_error(payload)
    if semantic_error is not None:
        return False

    dataset = artifact.get("dataset")
    fallback = artifact.get("fallback_evaluation")
    if not isinstance(dataset, Mapping) or not isinstance(
        fallback,
        Mapping,
    ):
        return False
    case_ids = dataset.get("case_ids")
    case_count = dataset.get("case_count")
    if (
        not isinstance(case_ids, list)
        or not case_ids
        or len(set(case_ids)) != len(case_ids)
        or not all(
            isinstance(item, str) and item
            for item in case_ids
        )
        or isinstance(case_count, bool)
        or not isinstance(case_count, int)
        or case_count != len(case_ids)
    ):
        return False

    return (
        _fallback_semantics_error(
            fallback,
            dataset_case_ids=case_ids,
        )
        is None
    )
