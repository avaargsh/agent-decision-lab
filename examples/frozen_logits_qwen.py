import argparse

from decision_lab.benchmark import load_jsonl
from decision_lab.logits import FrozenLogitAdapter, default_candidate_prompt
from decision_lab.runner import run_benchmark
from decision_lab.transformers_backend import TransformersCausalLMBackend


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    default="Qwen/Qwen3-0.6B",
    help="Any compatible Hugging Face causal LM checkpoint.",
)
parser.add_argument(
    "--dataset",
    default="benchmarks/examples/tool_router.jsonl",
)
args = parser.parse_args()

backend = TransformersCausalLMBackend(
    model_id=args.model,
    length_normalize=True,
)
adapter = FrozenLogitAdapter(
    backend=backend,
    prompt_builder=default_candidate_prompt,
)

report = run_benchmark(
    adapter,
    load_jsonl(args.dataset),
)

print(report)
