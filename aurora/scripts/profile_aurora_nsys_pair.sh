#!/usr/bin/env bash
# Paired Nsight Systems captures: bf16_mixed vs tf32_1x (same stamp, NVTX ranges on).
#
# Usage (repo root):
#   export CUTE_DSL_ARCH=sm_120a
#   ./aurora/scripts/profile_aurora_nsys_pair.sh
#
# Outputs:
#   profiling/nsight/aurora_bf16_mixed_<stamp>.nsys-rep
#   profiling/nsight/aurora_tf32_1x_<stamp>.nsys-rep
#   profiling/nsight/*_<stamp>_nvtx_sum.csv (after export)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_120a}"
export AURORA_NVTX=1
export AURORA_HF_LOCAL_DIR="${AURORA_HF_LOCAL_DIR:-/root/autodl-tmp/aurora}"

PAIR_STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${NSIGHT_OUT_DIR:-$REPO_ROOT/profiling/nsight}"
mkdir -p "$OUT_DIR"

PRESETS=(bf16_mixed tf32_1x)
REPORTS=()

for preset in "${PRESETS[@]}"; do
  echo ""
  echo "========== nsys: ${preset} =========="
  NSYS_STAMP="${PAIR_STAMP}" \
    NSYS_OUT_BASE="${OUT_DIR}/aurora_${preset}_${PAIR_STAMP}" \
    INFERENCE_PRECISION="${preset}" \
    "$SCRIPT_DIR/profile_aurora_small_nsys.sh" "$@"
  REPORTS+=("${OUT_DIR}/aurora_${preset}_${PAIR_STAMP}.nsys-rep")
done

echo ""
echo "========== export CSV (incl. nvtx_sum) =========="
for rep in "${REPORTS[@]}"; do
  "$SCRIPT_DIR/nsys_export_csv.sh" "$rep"
done

echo ""
echo "[pair] Done. Reports:"
for rep in "${REPORTS[@]}"; do
  echo "  $rep"
done
echo "Compare NVTX: grep aurora:: profiling/nsight/*_${PAIR_STAMP}_nvtx_sum.csv"
