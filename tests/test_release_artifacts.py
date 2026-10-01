import json
from pathlib import Path

from decision_lab.contract_matrix import build_contract_matrix
from decision_lab.eval_artifact import verify_eval_artifact


RELEASE_DIR = Path("release-artifacts/qwen3-0.6b-20260930")


def load(name):
    return json.loads((RELEASE_DIR / name).read_text(encoding="utf-8"))


def test_frozen_release_artifacts_are_verifiable_and_comparable():
    frozen = load("frozen-logits.decision-eval.json")
    structured = load("structured-output.decision-eval.json")
    provenance = load("provenance.json")

    assert verify_eval_artifact(frozen)
    assert verify_eval_artifact(structured)

    matrix = build_contract_matrix([frozen, structured])
    assert matrix["schema_version"] == "decision-contract-matrix/v1"
    assert matrix["dataset"]["sha256"] == provenance["dataset_sha256"]
    assert len(matrix["rows"]) == 2

    assert frozen["artifact_id"] == provenance["artifacts"]["frozen_logits"]
    assert (
        structured["artifact_id"]
        == provenance["artifacts"]["structured_output"]
    )
    assert frozen["calibration_sha256"] == provenance["calibration_sha256"]
    assert structured["calibration_sha256"] == provenance["calibration_sha256"]

    assert frozen["fallback_evaluation"]["measured"] is True
    assert structured["fallback_evaluation"]["measured"] is False

    test_ids = set(frozen["dataset"]["case_ids"])
    calibration_ids = set(frozen["calibration"]["case_ids"])
    assert test_ids.isdisjoint(calibration_ids)


def test_release_artifact_provenance_binds_original_measured_run():
    provenance = load("provenance.json")

    assert provenance["source_workflow_run_id"] == 36705402666
    assert provenance["source_workflow_artifact_id"] == 11092303346
    assert provenance["source_workflow_artifact_digest"] == (
        "sha256:aa1f2520e1a1cd39de827c390cbe99d367956f46c4050a3ff1a0014f4febaa0a"
    )
    assert provenance["source_head_sha"] == (
        "fb1ba30bfb386080e101140f6630ea29bf11b004"
    )
