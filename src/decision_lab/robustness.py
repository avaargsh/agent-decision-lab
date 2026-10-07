from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .benchmark import BenchmarkCase
from .metrics import Prediction
from .models import CandidateScore, DecisionRequest
from .runner import BenchmarkReport, DecisionAdapter
from .selective import CoveragePoint, risk_coverage


@dataclass(frozen=True)
class PermutationCaseResult:
    case_id: str
    permutation: str
    baseline_prediction: str
    perturbed_prediction: str
    baseline_correct: bool
    perturbed_correct: bool
    flipped: bool


@dataclass(frozen=True)
class PermutationRobustnessReport:
    adapter: str
    case_count: int
    comparison_count: int
    flip_count: int
    flip_rate: float
    baseline_accuracy: float
    perturbed_accuracy: float
    cases: list[PermutationCaseResult]


@dataclass(frozen=True)
class AbstentionCaseResult:
    case_id: str
    expected_abstain: bool
    accepted: bool
    predicted: str
    confidence: float
    correct_when_covered: bool | None


@dataclass(frozen=True)
class AbstentionRobustnessReport:
    adapter: str
    threshold: float
    case_count: int
    covered_case_count: int
    expected_abstain_case_count: int
    automation_coverage: float
    false_accept_rate: float | None
    false_rejection_rate: float | None
    covered_accuracy: float | None
    unsafe_automation_rate: float
    cases: list[AbstentionCaseResult]


@dataclass(frozen=True)
class ThresholdTransferReport:
    threshold: float
    source: CoveragePoint
    target: CoveragePoint
    source_ece: float
    target_ece: float
    coverage_delta: float
    risk_delta: float
    false_automation_rate_delta: float
    ece_delta: float


def _score_top1(
    adapter: DecisionAdapter,
    case: BenchmarkCase,
    candidates: Sequence[str],
) -> tuple[str, float]:
    if not candidates:
        raise ValueError(f"{case.case_id} has an empty candidate set")

    scores = list(
        adapter.score(
            DecisionRequest(
                decision_type=case.decision_type,
                candidates=list(candidates),
                context=case.context,
            )
        )
    )
    _validate_distribution(
        adapter_name=adapter.name,
        case_id=case.case_id,
        candidates=candidates,
        scores=scores,
    )
    winner = max(scores, key=lambda item: item.probability)
    return winner.candidate, winner.probability


def _validate_distribution(
    *,
    adapter_name: str,
    case_id: str,
    candidates: Sequence[str],
    scores: Sequence[CandidateScore],
) -> None:
    expected = set(candidates)
    actual = {score.candidate for score in scores}

    if len(scores) != len(candidates) or actual != expected:
        raise ValueError(
            f"{adapter_name} returned an invalid candidate set for {case_id}"
        )

    total = sum(score.probability for score in scores)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"{adapter_name} probabilities must sum to 1.0 for {case_id}; got {total}"
        )


def _candidate_permutations(
    candidates: Sequence[str],
) -> list[tuple[str, list[str]]]:
    original = list(candidates)
    if len(original) <= 1:
        return []

    variants: list[tuple[str, list[str]]] = []

    reversed_candidates = list(reversed(original))
    if reversed_candidates != original:
        variants.append(("reverse", reversed_candidates))

    rotated_candidates = original[1:] + original[:1]
    if (
        rotated_candidates != original
        and rotated_candidates != reversed_candidates
    ):
        variants.append(("rotate_left_1", rotated_candidates))

    return variants


def evaluate_permutation_robustness(
    adapter: DecisionAdapter,
    cases: Sequence[BenchmarkCase],
) -> PermutationRobustnessReport:
    if not cases:
        raise ValueError("cases must not be empty")

    baseline_correct = 0
    perturbed_correct = 0
    comparisons = 0
    flips = 0
    results: list[PermutationCaseResult] = []

    for case in cases:
        baseline_prediction, _ = _score_top1(
            adapter,
            case,
            case.candidates,
        )
        baseline_is_correct = baseline_prediction == case.gold_candidate
        baseline_correct += int(baseline_is_correct)

        for permutation_name, candidates in _candidate_permutations(
            case.candidates
        ):
            perturbed_prediction, _ = _score_top1(
                adapter,
                case,
                candidates,
            )
            perturbed_is_correct = (
                perturbed_prediction == case.gold_candidate
            )
            flipped = perturbed_prediction != baseline_prediction

            comparisons += 1
            flips += int(flipped)
            perturbed_correct += int(perturbed_is_correct)
            results.append(
                PermutationCaseResult(
                    case_id=case.case_id,
                    permutation=permutation_name,
                    baseline_prediction=baseline_prediction,
                    perturbed_prediction=perturbed_prediction,
                    baseline_correct=baseline_is_correct,
                    perturbed_correct=perturbed_is_correct,
                    flipped=flipped,
                )
            )

    return PermutationRobustnessReport(
        adapter=adapter.name,
        case_count=len(cases),
        comparison_count=comparisons,
        flip_count=flips,
        flip_rate=(flips / comparisons if comparisons else 0.0),
        baseline_accuracy=baseline_correct / len(cases),
        perturbed_accuracy=(
            perturbed_correct / comparisons
            if comparisons
            else baseline_correct / len(cases)
        ),
        cases=results,
    )


def evaluate_abstention_robustness(
    adapter: DecisionAdapter,
    cases: Sequence[BenchmarkCase],
    *,
    threshold: float,
) -> AbstentionRobustnessReport:
    if not cases:
        raise ValueError("cases must not be empty")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    expected_abstain_count = 0
    false_accept_count = 0
    covered_count = 0
    false_rejection_count = 0
    covered_correct_count = 0
    accepted_count = 0
    unsafe_automation_count = 0
    results: list[AbstentionCaseResult] = []

    for case in cases:
        expected_abstain = bool(
            case.metadata.get("expected_abstain", False)
        )
        predicted, confidence = _score_top1(
            adapter,
            case,
            case.candidates,
        )
        accepted = confidence >= threshold
        accepted_count += int(accepted)

        correct_when_covered: bool | None = None
        if expected_abstain:
            expected_abstain_count += 1
            if accepted:
                false_accept_count += 1
                unsafe_automation_count += 1
        else:
            if case.gold_candidate not in case.candidates:
                raise ValueError(
                    f"{case.case_id} is marked covered but gold candidate is missing"
                )
            covered_count += 1
            correct_when_covered = predicted == case.gold_candidate
            covered_correct_count += int(correct_when_covered)

            if not accepted:
                false_rejection_count += 1
            elif not correct_when_covered:
                unsafe_automation_count += 1

        results.append(
            AbstentionCaseResult(
                case_id=case.case_id,
                expected_abstain=expected_abstain,
                accepted=accepted,
                predicted=predicted,
                confidence=confidence,
                correct_when_covered=correct_when_covered,
            )
        )

    return AbstentionRobustnessReport(
        adapter=adapter.name,
        threshold=threshold,
        case_count=len(cases),
        covered_case_count=covered_count,
        expected_abstain_case_count=expected_abstain_count,
        automation_coverage=accepted_count / len(cases),
        false_accept_rate=(
            false_accept_count / expected_abstain_count
            if expected_abstain_count
            else None
        ),
        false_rejection_rate=(
            false_rejection_count / covered_count
            if covered_count
            else None
        ),
        covered_accuracy=(
            covered_correct_count / covered_count
            if covered_count
            else None
        ),
        unsafe_automation_rate=unsafe_automation_count / len(cases),
        cases=results,
    )


def evaluate_threshold_transfer(
    source: BenchmarkReport,
    target: BenchmarkReport,
    *,
    threshold: float | None = None,
) -> ThresholdTransferReport:
    if threshold is None:
        if source.operating_point is None:
            raise ValueError(
                "source report must have an operating point or threshold must be provided"
            )
        threshold = source.operating_point.threshold

    source_point = _coverage_at_threshold(source, threshold)
    target_point = _coverage_at_threshold(target, threshold)

    return ThresholdTransferReport(
        threshold=threshold,
        source=source_point,
        target=target_point,
        source_ece=source.ece,
        target_ece=target.ece,
        coverage_delta=target_point.coverage - source_point.coverage,
        risk_delta=target_point.risk - source_point.risk,
        false_automation_rate_delta=(
            target_point.false_automation_rate
            - source_point.false_automation_rate
        ),
        ece_delta=target.ece - source.ece,
    )


def _coverage_at_threshold(
    report: BenchmarkReport,
    threshold: float,
) -> CoveragePoint:
    predictions = [
        Prediction(
            confidence=case.confidence,
            correct=case.correct,
        )
        for case in report.cases
    ]
    return risk_coverage(
        predictions,
        thresholds=[threshold],
    )[0]
