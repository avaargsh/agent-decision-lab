"""Resolve a single immutable Hugging Face snapshot for every replay arm.

This hashes the materialized model files as evidence; a Git revision alone
is a source reference, not a guarantee that local cache bytes were unchanged.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable


def validate_model_pin(model_id: str, revision: str) -> None:
    if not isinstance(model_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*",
        model_id,
    ):
        raise ValueError("model must be a Hugging Face owner/repository id")
    if not isinstance(revision, str) or not re.fullmatch(
        r"[0-9a-f]{40}", revision
    ):
        raise ValueError("model_revision must be a full lowercase 40-hex commit SHA")


def _digest_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return "sha256:" + hasher.hexdigest()


def _snapshot_files(snapshot: Path) -> list[dict[str, Any]]:
    if not snapshot.is_dir() or snapshot.parent.name != "snapshots":
        raise ValueError("model snapshot must be under the hub snapshots directory")
    cache_root = snapshot.parent.parent.resolve()
    files: list[dict[str, Any]] = []
    for path in sorted(snapshot.rglob("*")):
        if not path.is_file():
            continue
        # Hugging Face cache symlinks commonly resolve into cache_root/blobs.
        if not path.resolve().is_relative_to(cache_root):
            raise ValueError("model snapshot file resolves outside its cache")
        files.append({
            "path": path.relative_to(snapshot).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _digest_file(path),
        })
    if not any(f["path"] == "config.json" for f in files):
        raise ValueError("model snapshot lacks config.json")
    if not any(
        f["path"].endswith((".safetensors", ".bin")) for f in files
    ):
        raise ValueError("model snapshot lacks a model weight file")
    return files


def freeze_model_snapshot(
    model_id: str,
    revision: str,
    *,
    downloader: Callable[..., str] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Download pinned bytes once, returning the local path and sealed evidence.

    The caller MUST load all model arms from the returned path, not the
    mutable remote repo name. No network access is required for unit tests.
    """
    validate_model_pin(model_id, revision)
    if downloader is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is required for GPU replay") from exc
        downloader = snapshot_download

    snapshot = Path(downloader(repo_id=model_id, revision=revision))
    if snapshot.name != revision or snapshot.parent.name != "snapshots":
        raise ValueError("downloaded snapshot path does not match pinned revision")

    base: dict[str, Any] = {
        "schema_version": "model-snapshot/v1",
        "model_id": model_id,
        "revision": revision,
        "model_ref": f"{model_id}@{revision}",
        "files": _snapshot_files(snapshot),
    }
    canonical = json.dumps(
        base, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    evidence = {
        **base,
        "content_digest": "sha256:" + hashlib.sha256(canonical).hexdigest(),
    }
    return snapshot, evidence


def verify_model_snapshot(
    snapshot: Path,
    evidence: dict[str, Any],
) -> bool:
    """Independently recompute snapshot file digests without a hub request."""
    try:
        validate_model_pin(evidence["model_id"], evidence["revision"])
        if Path(snapshot).name != evidence["revision"]:
            return False
        base = {
            "schema_version": "model-snapshot/v1",
            "model_id": evidence["model_id"],
            "revision": evidence["revision"],
            "model_ref": f"{evidence['model_id']}@{evidence['revision']}",
            "files": _snapshot_files(Path(snapshot)),
        }
        canonical = json.dumps(
            base, sort_keys=True, ensure_ascii=False,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
        expected = "sha256:" + hashlib.sha256(canonical).hexdigest()
        return evidence == {**base, "content_digest": expected}
    except (KeyError, TypeError, ValueError, OSError):
        return False
