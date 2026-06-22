#!/usr/bin/env bash
# Run AuroraSmall under NVIDIA Nsight Systems with settings closer to steady-state inference:
#   - Many warmup iterations (cuDNN / caches settle)
#   - cudnn.benchmark for fixed-shape paths
#   - cudaProfilerStart/Stop + nsys --capture-range=cudaProfilerApi so the timeline is
#     dominated by forward, not Python import / dlopen
#
# Requires: nsys (PATH or bundled under /opt/nvidia/nsight-compute/...), GPU, checkpoint.
#
# Usage (from repo root):
#   ./aurora/scripts/profile_aurora_small_nsys.sh
#   ./aurora/scripts/profile_aurora_small_nsys.sh --synthetic
#
# Override defaults:
#   NSYS_WARMUP=24 NSIGHT_OUT_DIR=/tmp/nsight ./aurora/scripts/profile_aurora_small_nsys.sh
#   INFERENCE_PRECISION=bf16_mixed CUTE_DSL_ARCH=sm_120a ./aurora/scripts/profile_aurora_small_nsys.sh
#   ./aurora/scripts/profile_aurora_nsys_pair.sh   # bf16_mixed + tf32 with NVTX
#
# Disable CUDA profiler API capture (full process timeline, more import noise):
#   NSYS_CAPTURE_API=0 ./aurora/scripts/profile_aurora_small_nsys.sh
#
# Open the report:
#   nsys-ui profiling/nsight/aurora_small_<timestamp>.nsys-rep

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_nsys_path.sh
source "$SCRIPT_DIR/_nsys_path.sh"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${NSIGHT_OUT_DIR:-$REPO_ROOT/profiling/nsight}"
mkdir -p "$OUT_DIR"

# Steady-state oriented defaults (override with env).
WARMUP="${NSYS_WARMUP:-16}"
REPEAT="${NSYS_REPEAT:-1}"
NSYS_CAPTURE_API="${NSYS_CAPTURE_API:-1}"
INFERENCE_PRECISION="${INFERENCE_PRECISION:-bf16_mixed}"
STAMP="${NSYS_STAMP:-$(date +%Y%m%d_%H%M%S)}"
if [[ -n "${NSYS_OUT_BASE:-}" ]]; then
  OUT_BASE="${NSYS_OUT_BASE}"
else
  OUT_BASE="$OUT_DIR/aurora_${INFERENCE_PRECISION}_${STAMP}"
fi
export AURORA_NVTX="${AURORA_NVTX:-1}"
AURORA_ASSET_ROOT="${AURORA_ASSET_ROOT:-./assets}"
export AURORA_HF_LOCAL_DIR

echo "[nsys] using: ${NSYS_BIN}"
echo "[nsys] output base: ${OUT_BASE}"
echo "[nsys] repo root: ${REPO_ROOT}"
echo "[nsys] warmup=${WARMUP} repeat=${REPEAT} capture_cuda_profiler_api=${NSYS_CAPTURE_API} inference_precision=${INFERENCE_PRECISION} AURORA_NVTX=${AURORA_NVTX}"

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
  --inference-precision "${INFERENCE_PRECISION}"
)

if [[ "${NSYS_CAPTURE_API}" == "1" ]]; then
  PY_ARGS+=(--cuda-profiler-api)
fi

if [[ "${CUDA_GRAPH:-0}" == "1" ]]; then
  PY_ARGS+=(--cuda-graph)
  WARMUP=$((WARMUP + 4))
  echo "[nsys] CUDA_GRAPH=1 → --cuda-graph backbone replay, warmup=${WARMUP}"
fi

"${NSYS_BIN}" "${NSYS_ARGS[@]}" "${PY_ARGS[@]}" "$@"

echo ""
echo "Report: ${OUT_BASE}.nsys-rep"
echo "Open with: nsys-ui \"${OUT_BASE}.nsys-rep\""
echo "Export CSV summaries (for review in chat / IDE):"
echo "  ./aurora/scripts/nsys_export_csv.sh \"${OUT_BASE}.nsys-rep\""
