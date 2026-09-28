from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TransformersCausalLMBackend:
    """Optional Hugging Face backend for candidate continuation scoring.

    Imports torch/transformers lazily so the core package and CI remain lightweight.
    """

    model_id: str
    device: str = "auto"
    dtype: str = "auto"
    length_normalize: bool = True
    name: str = "transformers-causal-lm"

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                'install model dependencies with: pip install -e ".[models]"'
            ) from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        kwargs = {}
        if self.dtype != "auto":
            kwargs["torch_dtype"] = getattr(torch, self.dtype)

        if self.device == "auto":
            kwargs["device_map"] = "auto"
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **kwargs,
        )
        self._model.eval()

    def logprob(self, *, prompt: str, candidate: str) -> float:
        torch = self._torch
        tokenizer = self._tokenizer
        model = self._model

        prompt_ids = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )["input_ids"]

        full_ids = tokenizer(
            prompt + candidate,
            return_tensors="pt",
            add_special_tokens=True,
        )["input_ids"]

        if full_ids.shape[1] <= prompt_ids.shape[1]:
            raise ValueError("candidate produced no additional tokens")

        device = next(model.parameters()).device
        full_ids = full_ids.to(device)

        with torch.no_grad():
            logits = model(full_ids).logits[:, :-1, :]
            labels = full_ids[:, 1:]

            token_logprobs = torch.log_softmax(logits, dim=-1)
            selected = token_logprobs.gather(
                -1,
                labels.unsqueeze(-1),
            ).squeeze(-1)

        # labels index i predicts token i+1; candidate starts after prompt token count.
        start = max(prompt_ids.shape[1] - 1, 0)
        candidate_logprobs = selected[:, start:]

        value = candidate_logprobs.sum().item()
        if self.length_normalize:
            value /= candidate_logprobs.shape[1]

        return float(value)
