from __future__ import annotations

from decision_lab.transformers_backend import TransformersCausalLMBackend


class FakeChunkBackend(TransformersCausalLMBackend):
    def __post_init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.last_tokens_processed = 0

    def _logprobs_batch(
        self,
        *,
        prompt: str,
        candidates,
    ):
        batch = tuple(candidates)
        self.calls.append(batch)
        return (
            [float(len(candidate)) for candidate in batch],
            10 * len(batch),
        )


def test_candidate_scoring_chunks_without_reordering() -> None:
    backend = FakeChunkBackend(
        model_id="fake",
        candidate_batch_size=2,
    )

    values = backend.logprobs(
        prompt="prompt",
        candidates=["a", "bb", "ccc", "dddd", "eeeee"],
    )

    assert backend.calls == [
        ("a", "bb"),
        ("ccc", "dddd"),
        ("eeeee",),
    ]
    assert values == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert backend.last_tokens_processed == 50


def test_candidate_scoring_uses_one_batch_when_chunking_disabled() -> None:
    backend = FakeChunkBackend(
        model_id="fake",
        candidate_batch_size=None,
    )

    backend.logprobs(
        prompt="prompt",
        candidates=["a", "b", "c"],
    )

    assert backend.calls == [("a", "b", "c")]
