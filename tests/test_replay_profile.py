from __future__ import annotations

from dataclasses import replace

import pytest

from decision_lab.adapters import MappingScoreAdapter
from decision_lab.benchmark import BenchmarkCase, dump_jsonl
from decision_lab.calibration_profile import (
    build_profile_bound_gateway,
    verify_calibration_profile,
)
from decision_lab.quality_experiment import run_calibrated_quality_experiment
from decision_lab.replay_profile import build_replay_calibration_profile

INVENTORY_SHA = "sha256:" + "b" * 64
MODEL_REF = "Qwen/Qwen3-0.6B@fixture-revision"


def case(id_, split, intent, gold):
    return BenchmarkCase(
        case_id=id_,
        decision_type="router",
        context={"intent": intent},
        candidates=["a", "b"],
        gold_candidate=gold,
        metadata={"split": split},
    )


def fixture(tmp_path):
    def mapping(request):
        intent = request.context["intent"]
        return {"a": 0.95, "b": 0.05} if intent.endswith("-a") else {
            "a": 0.05, "b": 0.95
        }

    calibration = [
        case("c1", "calibration", "cal-a", "a"),
        case("c2", "calibration", "cal-b", "b"),
    ]
    test = [
        case("t1", "test", "test-a", "a"),
        case("t2", "test", "test-b", "b"),
    ]
    abstention = [
        BenchmarkCase(
            case_id="ab1", decision_type="router",
            context={"intent": "unknown-a"},
            candidates=["a", "b"], gold_candidate=None,
            metadata={
                "split": "test",
                "expected_abstain": True,
                "coverage": "unsupported",
                "ambiguity": "clear",
                "automation_risk": "medium",
            },
        )
    ]
    path = tmp_path / "calibration.jsonl"
    dump_jsonl(calibration, path)
    adapter = MappingScoreAdapter(mapping)
    report = run_calibrated_quality_experiment(
        adapter,
        calibration_cases=calibration,
        test_cases=test,
        abstention_cases=abstention,
        risk_budget=0.0,
        min_coverage=0.5,
        threshold_grid=[0.5],
    )
    return path, calibration, test, report, adapter


def produce(path, calibration, test, report):
    return build_replay_calibration_profile(
        report,
        calibration_path=path,
        calibration_cases=calibration,
        test_cases=test,
        model_ref=MODEL_REF,
        inventory_sha256=INVENTORY_SHA,
    )


def test_quality_report_generates_replayable_profile(tmp_path):
    path, calibration, test, report, adapter = fixture(tmp_path)
    profile = produce(path, calibration, test, report)
    assert verify_calibration_profile(profile)
    assert profile["temperature"] == report.temperature_fit.temperature
    assert profile["execute_threshold"] == report.transfer_threshold
    assert profile["threshold_selection"] == report.threshold_selection
    assert profile["calibration"]["case_ids"] == ["c1", "c2"]
    assert profile["inventory_sha256"] == INVENTORY_SHA
    assert profile == produce(path, calibration, test, report)
    bound = build_profile_bound_gateway(
        adapter.score,
        profile=profile,
        adapter=adapter.name,
        model_ref=MODEL_REF,
        decision_type="router",
        inventory_sha256=INVENTORY_SHA,
        test_case_ids=["t1", "t2"],
    )
    assert bound.decide(
        __import__("decision_lab.models", fromlist=["DecisionRequest"]).DecisionRequest(
            "router", ["a", "b"], {"intent": "test-a"}
        )
    ).action in {"EXECUTE", "FALLBACK"}


def test_edited_calibration_file_rejected(tmp_path):
    path, calibration, test, report, _ = fixture(tmp_path)
    path.write_text(path.read_text().replace("cal-a", "tampered"))
    with pytest.raises(ValueError, match="bytes do not match"):
        produce(path, calibration, test, report)


def test_report_case_order_and_gold_bound_to_calibration(tmp_path):
    path, calibration, test, report, _ = fixture(tmp_path)
    bad = replace(
        report,
        source_calibration=replace(
            report.source_calibration,
            cases=list(reversed(report.source_calibration.cases)),
        ),
    )
    with pytest.raises(ValueError, match="case identities"):
        produce(path, calibration, test, bad)
    bad_gold = replace(
        report,
        source_calibration=replace(
            report.source_calibration,
            cases=[replace(report.source_calibration.cases[0], gold="b"),
                   report.source_calibration.cases[1]],
        ),
    )
    with pytest.raises(ValueError, match="gold labels"):
        produce(path, calibration, test, bad_gold)


def test_rejects_threshold_and_adapter_mismatch(tmp_path):
    path, calibration, test, report, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="operating point"):
        produce(path, calibration, test, replace(
            report, transfer_threshold=0.95
        ))
    with pytest.raises(ValueError, match="adapter mismatch"):
        produce(path, calibration, test, replace(report, adapter="other"))


def test_rejects_test_leakage(tmp_path):
    path, calibration, _, report, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="overlap"):
        produce(path, calibration, [
            case("c1", "test", "test-a", "a"),
        ], report)


def test_rejects_conflicting_decision_type(tmp_path):
    path, calibration, test, report, _ = fixture(tmp_path)
    mixed = [replace(test[0], decision_type="policy_gate"), test[1]]
    with pytest.raises(ValueError, match="one decision type"):
        produce(path, calibration, mixed, report)
