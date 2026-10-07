#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
CANDIDATE_BATCH_SIZE="${CANDIDATE_BATCH_SIZE:-16}"
RISK_BUDGET="${RISK_BUDGET:-0.5}"
MIN_COVERAGE="${MIN_COVERAGE:-0.25}"
PERMUTATION_CASES_PER_K="${PERMUTATION_CASES_PER_K:-2}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
AWS_MCP_REV="5abe7e82ba70556f80d2f42558d63a20aa3ce56c"
AWS_MCP_DIGEST="9e36175089bbba8714ad9e5db5efddd92812516937cf8acd3c124d61517e3d67"

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${OUT_DIR:-artifacts/gpu-replay/$RUN_ID}"
UPSTREAM_DIR="${UPSTREAM_DIR:-.work/awslabs-mcp}"

mkdir -p "$OUT_DIR" "$(dirname "$UPSTREAM_DIR")"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is required. GPU replay fails closed on CPU-only hosts." >&2
  exit 2
fi

nvidia-smi > "$OUT_DIR/nvidia-smi.txt"
nvidia-smi   --query-gpu=name,uuid,driver_version,memory.total   --format=csv,noheader   > "$OUT_DIR/gpu.csv"

if [[ "$SKIP_INSTALL" != "1" ]]; then
  python -m pip install -U pip
  python -m pip install -e ".[models]"
fi

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit(
        "ERROR: torch.cuda.is_available() is false; refusing GPU evidence run"
    )

print(
    f"cuda_available={torch.cuda.is_available()} "
    f"device_count={torch.cuda.device_count()} "
    f"device={torch.cuda.get_device_name(0)}"
)
PY

if [[ ! -d "$UPSTREAM_DIR/.git" ]]; then
  git clone     --filter=blob:none     --no-checkout     https://github.com/awslabs/mcp.git     "$UPSTREAM_DIR"
fi

git -C "$UPSTREAM_DIR" fetch --depth=1 origin "$AWS_MCP_REV"
git -C "$UPSTREAM_DIR" sparse-checkout init --cone
git -C "$UPSTREAM_DIR" sparse-checkout set src
git -C "$UPSTREAM_DIR" checkout --detach FETCH_HEAD

python scripts/extract_mcp_inventory.py   --root "$UPSTREAM_DIR"   --repository awslabs/mcp   --revision "$AWS_MCP_REV"   --inventory-id awslabs-mcp@5abe7e82ba70   --output "$OUT_DIR/awslabs-mcp.inventory.json"

EXPECTED_DIGEST="$AWS_MCP_DIGEST" INVENTORY_PATH="$OUT_DIR/awslabs-mcp.inventory.json" python - <<'PY'
import os
from decision_lab.inventory import inventory_digest, load_inventory

path = os.environ["INVENTORY_PATH"]
expected = os.environ["EXPECTED_DIGEST"]
inventory = load_inventory(path)
actual = inventory_digest(inventory)

if len(inventory.tools) != 804:
    raise SystemExit(
        f"inventory tool-count drift: expected 804, got {len(inventory.tools)}"
    )
if actual != expected:
    raise SystemExit(
        f"inventory digest drift: expected {expected}, got {actual}"
    )

print(f"inventory_tools=804 sha256={actual}")
PY

python scripts/build_m5_stress_artifacts.py   --inventory "$OUT_DIR/awslabs-mcp.inventory.json"   --calibration benchmarks/mcp_tool_router/v2.real_identity.calibration.jsonl   --test benchmarks/mcp_tool_router/v2.real_identity.test.jsonl   --candidate-counts 5,10,20,50,100   --strategies random   --seed agent-decision-benchmark-v0.2   --output-dir "$OUT_DIR/m5-inputs"

python examples/run_m5_qwen_stress.py   --model "$MODEL"   --inventory "$OUT_DIR/awslabs-mcp.inventory.json"   --calibration benchmarks/mcp_tool_router/v2.real_identity.calibration.jsonl   --base-test benchmarks/mcp_tool_router/v2.real_identity.test.jsonl   --covered "$OUT_DIR/m5-inputs/test.random.covered.jsonl"   --missing "$OUT_DIR/m5-inputs/test.random.missing.jsonl"   --candidate-batch-size "$CANDIDATE_BATCH_SIZE"   --risk-budget "$RISK_BUDGET"   --min-coverage "$MIN_COVERAGE"   --permutation-cases-per-k "$PERMUTATION_CASES_PER_K"   --output "$OUT_DIR/m5-qwen-stress.json"

{
  echo "git_sha=$(git rev-parse HEAD)"
  echo "model=$MODEL"
  echo "candidate_batch_size=$CANDIDATE_BATCH_SIZE"
  echo "risk_budget=$RISK_BUDGET"
  echo "min_coverage=$MIN_COVERAGE"
  echo "permutation_cases_per_k=$PERMUTATION_CASES_PER_K"
  echo "aws_mcp_revision=$AWS_MCP_REV"
  echo "aws_mcp_inventory_sha256=$AWS_MCP_DIGEST"
  echo "--- python ---"
  python --version
  echo "--- uname ---"
  uname -a
  echo "--- packages ---"
  python -m pip freeze
} > "$OUT_DIR/environment.txt"

python - "$OUT_DIR" "$MODEL" "$CANDIDATE_BATCH_SIZE" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
model = sys.argv[2]
batch_size = int(sys.argv[3])

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

gpu_rows = [
    line.strip()
    for line in (out_dir / "gpu.csv").read_text().splitlines()
    if line.strip()
]

manifest = {
    "schema_version": "m5-gpu-replay/v1",
    "git_sha": subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip(),
    "model": model,
    "candidate_batch_size": batch_size,
    "gpu": gpu_rows,
    "artifacts": {},
}

for path in sorted(out_dir.rglob("*")):
    if not path.is_file():
        continue
    if path.name == "run-manifest.json":
        continue
    relative = str(path.relative_to(out_dir))
    manifest["artifacts"][relative] = {
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }

(out_dir / "run-manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

(
  cd "$OUT_DIR"
  find . -type f ! -name sha256sums.txt -print0     | sort -z     | xargs -0 sha256sum     > sha256sums.txt
)

echo "GPU replay complete: $OUT_DIR"
echo "Primary report: $OUT_DIR/m5-qwen-stress.json"
echo "Run manifest: $OUT_DIR/run-manifest.json"
