from pathlib import Path

from decision_lab.benchmark import BenchmarkCase, dump_jsonl, load_jsonl


def test_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    cases = [
        BenchmarkCase(
            case_id="1",
            decision_type="router",
            context={"intent": "read metrics"},
            candidates=["a", "b"],
            gold_candidate="a",
            metadata={"split": "test"},
        )
    ]

    dump_jsonl(cases, path)
    loaded = load_jsonl(path)

    assert loaded == cases



def test_jsonl_round_trip_allows_null_gold_for_abstention(tmp_path: Path) -> None:
    path = tmp_path / "abstain.jsonl"
    cases = [
        BenchmarkCase(
            case_id="abstain-1",
            decision_type="router",
            context={"intent": "ambiguous request"},
            candidates=["a", "b"],
            gold_candidate=None,
            metadata={"expected_abstain": True},
        )
    ]

    dump_jsonl(cases, path)
    loaded = load_jsonl(path)

    assert loaded == cases
