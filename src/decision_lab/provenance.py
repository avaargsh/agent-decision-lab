from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Sequence

from .benchmark import BenchmarkCase


def file_sha256(path: str | Path) -> str:
    payload = Path(path).read_bytes()
    return "sha256:" + sha256(payload).hexdigest()


def dataset_provenance(
    path: str | Path,
    cases: Sequence[BenchmarkCase],
) -> dict[str, Any]:
    resolved = Path(path)
    return {
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "case_count": len(cases),
        "case_ids": [case.case_id for case in cases],
    }
