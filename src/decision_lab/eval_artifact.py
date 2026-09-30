from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from typing import Any, Mapping

from .runner import BenchmarkReport


SCHEMA_VERSION = "decision-eval/v1"


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

    dataset_sha256 = str(dataset.get("sha256", ""))
    if not dataset_sha256.startswith("sha256:"):
        raise ValueError("dataset.sha256 must be a sha256: digest")

    case_ids = list(dataset.get("case_ids", []))
    case_count = int(dataset.get("case_count", len(case_ids)))
    report_case_ids = [case.case_id for case in report.cases]

    if case_count != len(case_ids):
        raise ValueError("dataset case_count does not match case_ids")
    if report_case_ids and report_case_ids != case_ids:
        raise ValueError("benchmark report cases do not match dataset case_ids")

    operating = report.operating_point
    fallback = dict(fallback_evaluation or {})
    fallback.setdefault("measured", False)

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
    return actual == expected
