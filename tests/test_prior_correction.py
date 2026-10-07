from __future__ import annotations

from decision_lab.models import DecisionRequest
from decision_lab.prior_correction import (
    PriorCorrectedFrozenLogitAdapter,
    neutral_candidate_prompt,
)


class FakeBatchBackend:
    name = "fake-prior"

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.last_tokens_processed = 0

    def logprob(self, *, prompt: str, candidate: str) -> float:
        raise AssertionError("batch path should be used")

    def logprobs(self, *, prompt: str, candidates):
        batch = tuple(candidates)
        self.calls.append((prompt, batch))
        self.last_tokens_processed = 10 * len(batch)

        if "task_evidence: unavailable" in prompt:
            values = {
                "a": -0.1,
                "b": -2.0,
                "c": -1.0,
            }
        else:
            values = {
                "a": -0.2,
                "b": -0.5,
                "c": -2.0,
            }

        return [values[candidate] for candidate in batch]


def _request(intent: str = "route this") -> DecisionRequest:
    return DecisionRequest(
        decision_type="mcp_tool_router",
        candidates=["a", "b", "c"],
        context={"intent": intent},
    )


def test_prior_correction_can_remove_candidate_prior_preference() -> None:
    backend = FakeBatchBackend()
    adapter = PriorCorrectedFrozenLogitAdapter(
        backend=backend,
        prompt_builder=lambda request: (
            f"intent={request.context['intent']}\nBest matching candidate: "
        ),
    )

    scores = adapter.score(_request())

    assert (
        max(
            scores,
            key=lambda item: item.probability,
        ).candidate
        == "b"
    )
    assert adapter.last_tokens_processed == 60


def test_prior_values_are_cached_across_contexts() -> None:
    backend = FakeBatchBackend()
    adapter = PriorCorrectedFrozenLogitAdapter(
        backend=backend,
        prompt_builder=lambda request: (
            f"intent={request.context['intent']}\nBest matching candidate: "
        ),
    )

    adapter.score(_request("first"))
    first_call_count = len(backend.calls)
    adapter.score(_request("second"))

    assert first_call_count == 2
    assert len(backend.calls) == 3
    assert adapter.last_tokens_processed == 30


def test_zero_prior_strength_preserves_raw_ranking() -> None:
    backend = FakeBatchBackend()
    adapter = PriorCorrectedFrozenLogitAdapter(
        backend=backend,
        prompt_builder=lambda request: "task\nBest matching candidate: ",
        prior_strength=0.0,
    )

    scores = adapter.score(_request())

    assert (
        max(
            scores,
            key=lambda item: item.probability,
        ).candidate
        == "a"
    )


def test_negative_prior_strength_is_rejected() -> None:
    try:
        PriorCorrectedFrozenLogitAdapter(
            backend=FakeBatchBackend(),
            prompt_builder=lambda request: "task",
            prior_strength=-0.1,
        )
    except ValueError as exc:
        assert "prior_strength" in str(exc)
    else:
        raise AssertionError("negative prior strength must fail")


def test_neutral_prompt_omits_task_specific_context() -> None:
    request = _request("secret task wording")

    prompt = neutral_candidate_prompt(request)

    assert "mcp_tool_router" in prompt
    assert "secret task wording" not in prompt
    assert "task_evidence: unavailable" in prompt
