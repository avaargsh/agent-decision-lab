from collections import Counter
from pathlib import Path

from decision_lab.benchmark import load_jsonl


ROOT = Path("benchmarks/mcp_tool_router")
EXPECTED = {
    "prometheus.query",
    "logs.search",
    "kubernetes.get",
    "runbook.search",
}


def test_mcp_tool_router_v1_has_disjoint_balanced_splits() -> None:
    calibration = load_jsonl(ROOT / "v1.calibration.jsonl")
    test = load_jsonl(ROOT / "v1.test.jsonl")

    assert len(calibration) == 16
    assert len(test) == 16

    calibration_ids = {case.case_id for case in calibration}
    test_ids = {case.case_id for case in test}
    assert calibration_ids.isdisjoint(test_ids)

    for split_name, cases in (
        ("calibration", calibration),
        ("test", test),
    ):
        assert all(case.decision_type == "mcp_tool_router" for case in cases)
        assert all(set(case.candidates) == EXPECTED for case in cases)
        assert all(case.gold_candidate in EXPECTED for case in cases)
        assert all(case.metadata["split"] == split_name for case in cases)
        assert all(case.metadata["source"] == "synthetic-mcp-router-v1" for case in cases)

        counts = Counter(case.gold_candidate for case in cases)
        assert counts == {candidate: 4 for candidate in EXPECTED}
