from decision_lab.provenance import dataset_provenance, file_sha256
from decision_lab.benchmark import load_jsonl


def test_file_sha256_changes_when_dataset_bytes_change(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"case_id":"a"}\n', encoding="utf-8")
    first = file_sha256(path)

    path.write_text('{"case_id":"b"}\n', encoding="utf-8")
    second = file_sha256(path)

    assert first.startswith("sha256:")
    assert second.startswith("sha256:")
    assert first != second


def test_dataset_provenance_binds_bytes_and_case_ids(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"case_id":"case-1","decision_type":"router","context":{},'
        '"candidates":["a","b"],"gold_candidate":"a","metadata":{}}\n',
        encoding="utf-8",
    )
    cases = load_jsonl(path)

    provenance = dataset_provenance(path, cases)

    assert provenance["path"] == str(path)
    assert provenance["sha256"] == file_sha256(path)
    assert provenance["case_count"] == 1
    assert provenance["case_ids"] == ["case-1"]
