#!/usr/bin/env bash
# Run AuroraSmall under NVIDIA Nsight Systems with settings closer to steady-state inference:
#   - Many warmup iterations (cuDNN / caches settle)
#   - cudnn.benchmark for fixed-shape paths
#   - cudaProfilerStart/Stop + nsys --capture-range=cudaProfilerApi so the timeline is
#     dominated by forward, not Python import / dlopen
#
# Requires: nsys on PATH, GPU, HF cache for checkpoint.
#
# Usage (from repo root):
#   ./aurora/scripts/profile_aurora_small_nsys.sh
#   ./aurora/scripts/profile_aurora_small_nsys.sh --synthetic
#
# Override defaults:
#   NSYS_WARMUP=24 NSIGHT_OUT_DIR=/tmp/nsight ./aurora/scripts/profile_aurora_small_nsys.sh
#
# Disable CUDA profiler API capture (full process timeline, more import noise):
#   NSYS_CAPTURE_API=0 ./aurora/scripts/profile_aurora_small_nsys.sh
#
# Open the report:
#   nsys-ui profiling/nsight/aurora_small_<timestamp>.nsys-rep

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${NSIGHT_OUT_DIR:-$REPO_ROOT/profiling/nsight}"
mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_BASE="$OUT_DIR/aurora_small_${STAMP}"

# Steady-state oriented defaults (override with env).
WARMUP="${NSYS_WARMUP:-16}"
REPEAT="${NSYS_REPEAT:-1}"
NSYS_CAPTURE_API="${NSYS_CAPTURE_API:-1}"

if ! command -v nsys &>/dev/null; then
  echo "nsys not found. Install NVIDIA Nsight Systems and ensure it is on PATH." >&2
  exit 1
fi

echo "[nsys] output base: ${OUT_BASE}"
echo "[nsys] repo root: ${REPO_ROOT}"
echo "[nsys] warmup=${WARMUP} repeat=${REPEAT} capture_cuda_profiler_api=${NSYS_CAPTURE_API}"

NSYS_ARGS=(
  profile
  --output "$OUT_BASE"
  --force-overwrite true
  --trace=cuda,nvtx,cublas,cudnn,osrt
)

if [[ "${NSYS_CAPTURE_API}" == "1" ]]; then
  NSYS_ARGS+=(--capture-range=cudaProfilerApi)
fi

PY_ARGS=(
  uv run python aurora/profiling.py
  --repeat "${REPEAT}"
  --forward-only
  --warmup "${WARMUP}"
  --no-torch-profiler
  --cudnn-benchmark
)

if [[ "${NSYS_CAPTURE_API}" == "1" ]]; then
  PY_ARGS+=(--cuda-profiler-api)
fi

nsys "${NSYS_ARGS[@]}" "${PY_ARGS[@]}" "$@"

echo ""
echo "Report: ${OUT_BASE}.nsys-rep"
echo "Open with: nsys-ui \"${OUT_BASE}.nsys-rep\""
echo "Export CSV summaries (for review in chat / IDE):"
echo "  ./aurora/scripts/nsys_export_csv.sh \"${OUT_BASE}.nsys-rep\""
