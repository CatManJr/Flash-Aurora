#!/usr/bin/env bash
# Compare multi-step rollout with/without backbone CUDA graph (bf16_mixed or tf32_1x).
#
# Usage (repo root):
#   export CUTE_DSL_ARCH=sm_120a
#   ./aurora/scripts/bench_rollout_cuda_graph.sh
#   ROLLOUT_STEPS=6 INFERENCE_PRECISION=tf32_1x ./aurora/scripts/bench_rollout_cuda_graph.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_120a}"
export AURORA_HF_LOCAL_DIR="${AURORA_HF_LOCAL_DIR:-/root/autodl-tmp/aurora}"

ROLLOUT_STEPS="${ROLLOUT_STEPS:-4}"
WARMUP="${WARMUP:-6}"
REPEAT="${REPEAT:-8}"
PRESET="${INFERENCE_PRECISION:-bf16_mixed}"

COMMON=(
  uv run python aurora/profiling.py
  --inference-precision "${PRESET}"
  --rollout-steps "${ROLLOUT_STEPS}"
  --warmup "${WARMUP}"
  --repeat "${REPEAT}"
  --no-torch-profiler
  --cudnn-benchmark
)

echo "========== rollout baseline (no CUDA graph) =========="
"${COMMON[@]}"

echo ""
echo "========== rollout + backbone CUDA graph =========="
"${COMMON[@]}" --cuda-graph
