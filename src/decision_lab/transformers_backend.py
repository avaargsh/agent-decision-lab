from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class TransformersCausalLMBackend:
    """Optional Hugging Face backend for candidate continuation scoring."""

    model_id: str
    device: str = "auto"
    dtype: str = "auto"
    length_normalize: bool = True
    candidate_batch_size: int | None = None
    name: str = "transformers-causal-lm"

    def __post_init__(self) -> None:
        if (
            self.candidate_batch_size is not None
            and self.candidate_batch_size <= 0
        ):
            raise ValueError(
                "candidate_batch_size must be > 0 when set"
            )

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                'install model dependencies with: pip install -e ".[models]"'
            ) from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id
        )

        kwargs = {}
        if self.dtype != "auto":
            kwargs["torch_dtype"] = getattr(
                torch,
                self.dtype,
            )

        if self.device == "auto":
            kwargs["device_map"] = "auto"

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **kwargs,
        )
        self._model.eval()
        self.last_tokens_processed = 0

    def logprobs(
        self,
        *,
        prompt: str,
        candidates: Sequence[str],
    ) -> list[float]:
        if not candidates:
            raise ValueError(
                "candidates must not be empty"
            )

        batch_size = (
            self.candidate_batch_size
            or len(candidates)
        )
        values: list[float] = []
        total_tokens = 0

        for start in range(
            0,
            len(candidates),
            batch_size,
        ):
            batch = candidates[
                start : start + batch_size
            ]
            batch_values, batch_tokens = (
                self._logprobs_batch(
                    prompt=prompt,
                    candidates=batch,
                )
            )
            values.extend(batch_values)
            total_tokens += batch_tokens

        self.last_tokens_processed = total_tokens
        return values

    def _logprobs_batch(
        self,
        *,
        prompt: str,
        candidates: Sequence[str],
    ) -> tuple[list[float], int]:
        torch = self._torch
        tokenizer = self._tokenizer
        model = self._model

        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=True,
        )["input_ids"]

        candidate_ids = [
            tokenizer(
                candidate,
                add_special_tokens=False,
            )["input_ids"]
            for candidate in candidates
        ]

        if any(
            len(ids) == 0
            for ids in candidate_ids
        ):
            raise ValueError(
                "candidate produced no tokens"
            )

        sequences = [
            prompt_ids + ids
            for ids in candidate_ids
        ]
        lengths = [
            len(sequence)
            for sequence in sequences
        ]
        max_length = max(lengths)

        pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        if pad_token_id is None:
            pad_token_id = 0

        input_ids = torch.full(
            (
                len(sequences),
                max_length,
            ),
            fill_value=pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (
                len(sequences),
                max_length,
            ),
            dtype=torch.long,
        )

        for row, sequence in enumerate(
            sequences
        ):
            length = len(sequence)
            input_ids[
                row,
                :length,
            ] = torch.tensor(
                sequence,
                dtype=torch.long,
            )
            attention_mask[
                row,
                :length,
            ] = 1

        device = next(
            model.parameters()
        ).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(
            device
        )

        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits[:, :-1, :]

            labels = input_ids[:, 1:]
            token_logprobs = torch.log_softmax(
                logits,
                dim=-1,
            )
            selected = token_logprobs.gather(
                -1,
                labels.unsqueeze(-1),
            ).squeeze(-1)

        start = max(
            len(prompt_ids) - 1,
            0,
        )
        values: list[float] = []

        for row, ids in enumerate(
            candidate_ids
        ):
            end = start + len(ids)
            continuation = selected[
                row,
                start:end,
            ]

            value = continuation.sum().item()
            if self.length_normalize:
                value /= len(ids)

            values.append(float(value))

        return values, sum(lengths)

    def logprob(
        self,
        *,
        prompt: str,
        candidate: str,
    ) -> float:
        return self.logprobs(
            prompt=prompt,
            candidates=[candidate],
        )[0]
