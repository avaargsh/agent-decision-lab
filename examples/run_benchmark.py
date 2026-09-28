from decision_lab.adapters import MappingScoreAdapter, StructuredOutputAdapter
from decision_lab.benchmark import load_jsonl
from decision_lab.runner import run_benchmark


cases = load_jsonl("benchmarks/examples/tool_router.jsonl")


def deterministic_mapping(request):
    intent = str(request.context.get("intent", "")).lower()
    if "error" in intent or "log" in intent:
        return {
            "read_metrics": 0.10,
            "read_logs": 0.85,
            "restart_workload": 0.05,
        }
    return {
        "read_metrics": 0.90,
        "read_logs": 0.08,
        "restart_workload": 0.02,
    }


def fake_structured_output(request):
    intent = str(request.context.get("intent", "")).lower()
    if "error" in intent or "log" in intent:
        return '{"candidate":"read_logs","confidence":0.82}'
    return '{"candidate":"read_metrics","confidence":0.88}'


for adapter in [
    MappingScoreAdapter(deterministic_mapping),
    StructuredOutputAdapter(fake_structured_output),
]:
    report = run_benchmark(adapter, cases)
    print(adapter.name, report)
