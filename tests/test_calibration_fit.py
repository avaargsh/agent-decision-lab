from decision_lab.calibration_fit import (
    CalibrationExample,
    fit_temperature,
    multiclass_nll,
)
from decision_lab.models import CandidateScore


def test_temperature_fit_reduces_nll_for_overconfident_errors() -> None:
    examples = [
        CalibrationExample(
            scores=[
                CandidateScore("a", 0.99),
                CandidateScore("b", 0.01),
            ],
            gold_candidate="b",
        ),
        CalibrationExample(
            scores=[
                CandidateScore("a", 0.90),
                CandidateScore("b", 0.10),
            ],
            gold_candidate="a",
        ),
    ]

    fit = fit_temperature(
        examples,
        min_temperature=0.1,
        max_temperature=20.0,
        steps=100,
    )

    assert fit.nll_after < fit.nll_before
    assert fit.temperature > 1.0


def test_multiclass_nll_perfect_distribution_is_small() -> None:
    examples = [
        CalibrationExample(
            scores=[
                CandidateScore("a", 0.999),
                CandidateScore("b", 0.001),
            ],
            gold_candidate="a",
        )
    ]

    assert multiclass_nll(examples) < 0.01
