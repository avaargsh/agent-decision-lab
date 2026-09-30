from decision_lab.calibration import TemperatureCalibrator
from decision_lab.metrics import (
    Prediction,
    brier_score,
    expected_calibration_error,
    macro_f1,
    multiclass_nll,
)
from decision_lab.models import CandidateScore
from decision_lab.selective import risk_coverage


def test_temperature_calibration_normalizes() -> None:
    calibrated = TemperatureCalibrator(temperature=2.0).calibrate(
        [CandidateScore("a", 0.9), CandidateScore("b", 0.1)]
    )
    assert abs(sum(item.probability for item in calibrated) - 1.0) < 1e-9


def test_brier_score_perfect_predictions() -> None:
    assert brier_score(
        [Prediction(1.0, True), Prediction(0.0, False)]
    ) == 0.0


def test_ece_is_zero_for_perfect_extremes() -> None:
    assert expected_calibration_error(
        [Prediction(1.0, True), Prediction(0.0, False)],
        bins=10,
    ) == 0.0


def test_risk_coverage() -> None:
    points = risk_coverage(
        [
            Prediction(0.95, True),
            Prediction(0.90, False),
            Prediction(0.60, True),
        ],
        thresholds=[0.5, 0.9, 0.96],
    )

    assert points[0].coverage == 1.0
    assert points[1].coverage == 2 / 3
    assert points[1].false_automation_rate == 1 / 3
    assert points[2].coverage == 0.0


def test_multiclass_nll_uses_gold_probability() -> None:
    value = multiclass_nll([0.8, 0.5])
    assert 0.0 < value < 1.0


def test_macro_f1_is_one_for_perfect_multiclass_predictions() -> None:
    assert macro_f1(
        ["a", "b", "c"],
        ["a", "b", "c"],
    ) == 1.0


def test_macro_f1_penalizes_class_specific_errors() -> None:
    score = macro_f1(
        ["a", "a", "b", "b"],
        ["a", "b", "b", "b"],
    )
    assert 0.0 < score < 1.0
