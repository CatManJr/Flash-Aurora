#!/usr/bin/env bash
# Paired Nsight Systems captures: bf16_mixed vs tf32 (same stamp, NVTX ranges on).
#
# Usage (repo root):
#   ./aurora/scripts/profile_aurora_nsys_pair.sh
# Optional: export CUTE_DSL_ARCH=sm_120a  # only for cross-compiling to another GPU arch
#
# Outputs:
#   profiling/nsight/aurora_bf16_mixed_<stamp>.nsys-rep
#   profiling/nsight/aurora_tf32_<stamp>.nsys-rep
#   profiling/nsight/*_<stamp>_nvtx_sum.csv (after export)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# CuTe DSL JIT targets the current GPU unless CUTE_DSL_ARCH is set explicitly.
export AURORA_NVTX=1
if [[ -z "${AURORA_ASSET_ROOT:-}" && -z "${AURORA_HF_LOCAL_DIR:-}" ]]; then
  echo "Set AURORA_ASSET_ROOT to your data-disk asset directory (not ./assets under the repo)." >&2
  exit 1
fi
export AURORA_ASSET_ROOT="${AURORA_ASSET_ROOT:-${AURORA_HF_LOCAL_DIR}}"

PAIR_STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${NSIGHT_OUT_DIR:-$REPO_ROOT/profiling/nsight}"
mkdir -p "$OUT_DIR"

PRESETS=(bf16_mixed tf32)
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
