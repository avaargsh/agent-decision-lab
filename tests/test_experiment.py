from pathlib import Path

from decision_lab.adapters import MappingScoreAdapter
from decision_lab.benchmark import BenchmarkCase
from decision_lab.experiment import (
    run_calibrated_experiment,
    write_experiment_report,
)


def make_cases():
    return [
        BenchmarkCase(
            case_id="1",
            decision_type="router",
            context={"kind": "a"},
            candidates=["a", "b"],
            gold_candidate="a",
            metadata={},
        ),
        BenchmarkCase(
            case_id="2",
            decision_type="router",
            context={"kind": "b"},
            candidates=["a", "b"],
            gold_candidate="b",
            metadata={},
        ),
    ]


def test_calibrated_experiment_writes_report(tmp_path: Path) -> None:
    def mapping(request):
        if request.context["kind"] == "a":
            return {"a": 0.85, "b": 0.15}
        return {"a": 0.20, "b": 0.80}

    adapter = MappingScoreAdapter(mapping)
    fit, raw, calibrated = run_calibrated_experiment(
        adapter,
        calibration_cases=make_cases(),
        test_cases=make_cases(),
    )

    output = tmp_path / "report.json"
    write_experiment_report(
        output,
        fit=fit,
        raw=raw,
        calibrated=calibrated,
        metadata={"model": "fixture"},
    )

    assert output.exists()
    assert raw.accuracy == 1.0
    assert calibrated.accuracy == 1.0
