from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

from .adapters import structured_choice_scores
from .models import CandidateScore, DecisionRequest


def parse_structured_choice(
    text: str,
    candidates: Sequence[str],
) -> tuple[str, float]:
    start = text.find("{")
    end = text.rfind("}")

    if start < 0 or end < start:
        raise ValueError("no JSON object in generated text")

    payload = json.loads(
        text[start : end + 1]
    )

    candidate = str(payload["candidate"])
    confidence = float(
        payload.get("confidence", 1.0)
    )

    if candidate not in candidates:
        raise ValueError(
            f"generated unknown candidate: {candidate}"
        )
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            "generated confidence must be between 0 and 1"
        )

    return candidate, confidence


@dataclass
class TransformersStructuredOutputAdapter:
    """Same-base autoregressive structured-choice baseline."""

    model_id: str
    device: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = 32
    name: str = "transformers-structured-output"

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise RuntimeError(
                'install model dependencies with: '
                'pip install -e ".[models]"'
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

        self.last_tokens_processed: int | None = None
        self.last_parse_valid = False
        self.last_generated_text = ""

    def _prompt(
        self,
        request: DecisionRequest,
    ) -> str:
        context_lines = [
            f"{key}: {value}"
            for key, value in sorted(
                request.context.items()
            )
        ]
        candidates = ", ".join(
            request.candidates
        )

        return (
            "Choose exactly one candidate from the "
            "allowed set. Return only one JSON object "
            "with candidate and confidence.\n"
            f"Decision type: {request.decision_type}\n"
            f"Allowed candidates: {candidates}\n"
            + (
                "Context:\n"
                + "\n".join(context_lines)
                + "\n"
                if context_lines
                else ""
            )
            + 'Format: {"candidate":"<allowed>",'
            '"confidence":0.0}\n'
        )

    def score(
        self,
        request: DecisionRequest,
    ) -> Sequence[CandidateScore]:
        torch = self._torch
        tokenizer = self._tokenizer
        model = self._model

        prompt = self._prompt(request)
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )

        device = next(model.parameters()).device
        input_ids = encoded["input_ids"].to(
            device
        )
        attention_mask = encoded.get(
            "attention_mask"
        )
        if attention_mask is not None:
            attention_mask = attention_mask.to(
                device
            )

        kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
        }
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask

        with torch.no_grad():
            output_ids = model.generate(**kwargs)

        new_ids = output_ids[
            :,
            input_ids.shape[1] :,
        ]
        self.last_tokens_processed = int(
            input_ids.numel() + new_ids.numel()
        )
        self.last_generated_text = tokenizer.decode(
            new_ids[0],
            skip_special_tokens=True,
        ).strip()

        try:
            candidate, confidence = (
                parse_structured_choice(
                    self.last_generated_text,
                    request.candidates,
                )
            )
            self.last_parse_valid = True
        except (
            ValueError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
        ):
            self.last_parse_valid = False
            uniform = 1.0 / len(
                request.candidates
            )
            return [
                CandidateScore(
                    candidate,
                    uniform,
                )
                for candidate in request.candidates
            ]

        return structured_choice_scores(
            candidate=candidate,
            confidence=confidence,
            candidates=request.candidates,
        )
