"""CPU-only tests for immutable checkpoint identity and byte-level evidence."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from decision_lab.model_snapshot import (
    freeze_model_snapshot,
    validate_model_pin,
    verify_model_snapshot,
)

REV = "a" * 40
MODEL = "Qwen/Qwen3-0.6B"


def make_snapshot(tmp_path: Path) -> Path:
    cache = tmp_path / "models--Qwen--Qwen3-0.6B"
    snapshot = cache / "snapshots" / REV
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text('{"model_type":"qwen3"}')
    (snapshot / "tokenizer.json").write_text("{}")
    (cache / "blobs").mkdir()
    blob = cache / "blobs" / "weights"
    blob.write_bytes(b"test weight bytes")
    (snapshot / "model.safetensors").symlink_to(
        os.path.relpath(blob, start=snapshot)
    )
    return snapshot


def test_freezes_one_exact_hf_revision_and_fingerprints_files(tmp_path):
    path = make_snapshot(tmp_path)
    calls = []

    def downloader(*, repo_id, revision):
        calls.append((repo_id, revision))
        return str(path)

    actual_path, evidence = freeze_model_snapshot(
        MODEL, REV, downloader=downloader
    )
    assert calls == [(MODEL, REV)]
    assert actual_path == path
    assert evidence["model_ref"] == f"{MODEL}@{REV}"
    assert evidence["schema_version"] == "model-snapshot/v1"
    assert [f["path"] for f in evidence["files"]] == [
        "config.json", "model.safetensors", "tokenizer.json"
    ]
    assert verify_model_snapshot(actual_path, evidence)
    assert freeze_model_snapshot(MODEL, REV, downloader=downloader)[1] == evidence


def test_local_cache_mutation_invalidates_evidence(tmp_path):
    path = make_snapshot(tmp_path)
    _, evidence = freeze_model_snapshot(
        MODEL, REV, downloader=lambda **kwargs: str(path)
    )
    (path / "config.json").write_text('{"model_type":"tampered"}')
    assert not verify_model_snapshot(path, evidence)


@pytest.mark.parametrize("revision", [
    "main", "v1.0", "1234", "A" * 40, "../" + "a" * 37,
])
def test_mutable_or_invalid_revisions_fail_before_download(revision):
    invoked = []
    with pytest.raises(ValueError, match="40-hex"):
        freeze_model_snapshot(
            MODEL, revision, downloader=lambda **kwargs: invoked.append(1)
        )
    assert invoked == []


def test_snapshot_download_cannot_silently_return_different_commit(tmp_path):
    path = make_snapshot(tmp_path)
    with pytest.raises(ValueError, match="does not match"):
        freeze_model_snapshot(
            MODEL, "b" * 40, downloader=lambda **kwargs: str(path)
        )


def test_missing_weights_fails_closed(tmp_path):
    path = make_snapshot(tmp_path)
    (path / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="weight file"):
        freeze_model_snapshot(MODEL, REV, downloader=lambda **kwargs: str(path))


def test_snapshot_symlink_escape_fails_closed(tmp_path):
    path = make_snapshot(tmp_path)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"not within model cache")
    (path / "escape.bin").symlink_to(outside)
    with pytest.raises(ValueError, match="outside its cache"):
        freeze_model_snapshot(MODEL, REV, downloader=lambda **kwargs: str(path))


def test_model_path_or_unqualified_repo_is_rejected():
    for name in ["./local-checkpoint", "Qwen", "../Qwen/model"]:
        with pytest.raises(ValueError, match="owner/repository"):
            validate_model_pin(name, REV)
