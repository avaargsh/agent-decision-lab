from __future__ import annotations

from typing import Any, Mapping, Sequence

from .eval_artifact import verify_eval_artifact


class ContractMatrixError(ValueError):
    pass


def _dataset_identity(artifact: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    dataset = artifact.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ContractMatrixError("decision eval artifact is missing dataset")
    digest = dataset.get("sha256")
    case_ids = dataset.get("case_ids")
    if not isinstance(digest, str) or not isinstance(case_ids, list):
        raise ContractMatrixError("decision eval dataset identity is invalid")
    return digest, tuple(str(item) for item in case_ids)


def build_contract_matrix(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare decision adapters only after they pass the same artifact contract.

    The matrix deliberately refuses to compare artifacts from different test
    datasets. Calibration inputs may differ by adapter, but their provenance is
    surfaced explicitly so readers can distinguish model/adapter differences
    from evaluation-data differences.
    """
    if not artifacts:
        raise ContractMatrixError("at least one decision eval artifact is required")

    invalid = [
        index
        for index, artifact in enumerate(artifacts)
        if not verify_eval_artifact(artifact)
    ]
    if invalid:
        raise ContractMatrixError(
            "invalid decision eval artifacts at indexes: "
            + ",".join(str(index) for index in invalid)
        )

    schema_version = artifacts[0]["schema_version"]
    decision_type = artifacts[0]["decision_type"]
    dataset_digest, dataset_case_ids = _dataset_identity(artifacts[0])

    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        if artifact.get("schema_version") != schema_version:
            raise ContractMatrixError(
                "all artifacts must use the same schema_version"
            )
        if artifact.get("decision_type") != decision_type:
            raise ContractMatrixError(
                "all artifacts must use the same decision_type"
            )
        digest, case_ids = _dataset_identity(artifact)
        if digest != dataset_digest or case_ids != dataset_case_ids:
            raise ContractMatrixError(
                "all artifacts must use the exact same test dataset"
            )

        metrics = artifact["metrics"]
        operating = artifact.get("operating_point")
        fallback = artifact["fallback_evaluation"]
        calibration = artifact.get("calibration")

        rows.append(
            {
                "artifact_id": artifact["artifact_id"],
                "content_digest": artifact["content_digest"],
                "adapter": artifact["adapter"],
                "model_ref": artifact.get("model_ref"),
                "calibration_sha256": artifact.get("calibration_sha256"),
                "calibration_case_count": (
                    calibration.get("case_count")
                    if isinstance(calibration, Mapping)
                    else None
                ),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "nll": metrics["nll"],
                "brier": metrics["brier"],
                "ece": metrics["ece"],
                "p50_latency_ms": metrics["p50_latency_ms"],
                "p95_latency_ms": metrics["p95_latency_ms"],
                "mean_tokens_processed_per_decision": metrics[
                    "mean_tokens_processed_per_decision"
                ],
                "operating_threshold": (
                    operating.get("threshold")
                    if isinstance(operating, Mapping)
                    else None
                ),
                "coverage": (
                    operating.get("coverage")
                    if isinstance(operating, Mapping)
                    else None
                ),
                "risk": (
                    operating.get("risk")
                    if isinstance(operating, Mapping)
                    else None
                ),
                "false_automation_rate": (
                    operating.get("false_automation_rate")
                    if isinstance(operating, Mapping)
                    else None
                ),
                "fallback_rate": (
                    operating.get("fallback_rate")
                    if isinstance(operating, Mapping)
                    else None
                ),
                "risk_budget": (
                    operating.get("risk_budget")
                    if isinstance(operating, Mapping)
                    else None
                ),
                "fallback_measured": fallback["measured"],
                "fallback_adapter": fallback.get("adapter"),
                "fallback_accuracy": fallback.get("accuracy"),
                "fallback_p50_latency_ms": fallback.get("p50_latency_ms"),
                "fallback_p95_latency_ms": fallback.get("p95_latency_ms"),
                "fallback_mean_tokens_processed": fallback.get(
                    "mean_tokens_processed"
                ),
            }
        )

    return {
        "schema_version": "decision-contract-matrix/v1",
        "decision_eval_schema_version": schema_version,
        "decision_type": decision_type,
        "dataset": {
            "sha256": dataset_digest,
            "case_count": len(dataset_case_ids),
            "case_ids": list(dataset_case_ids),
        },
        "rows": rows,
    }
