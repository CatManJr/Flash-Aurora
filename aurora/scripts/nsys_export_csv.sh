#!/usr/bin/env bash
# Export human/LLM-readable CSV summaries from an Nsight Systems .nsys-rep file.
#
# Uses `nsys stats` (not the GUI). Rebuilds SQLite from the .nsys-rep with
# --force-export so stale/empty sidecar .sqlite files do not break the run.
#
# On some hosts (e.g. WSL2), kernel-level reports may be empty; `cuda_api_sum`
# and `cuda_api_gpu_sum` are usually still populated.
#
# Usage (from repo root):
#   ./aurora/scripts/nsys_export_csv.sh profiling/nsight/aurora_small_YYYYMMDD_HHMMSS.nsys-rep
#   ./aurora/scripts/nsys_export_csv.sh path/to/foo.nsys-rep /tmp/csv_out
#
# Requires: nsys on PATH.

set -euo pipefail

if [[ $# -lt 1 ]] || [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
  echo "Usage: $0 <capture.nsys-rep> [output-directory]" >&2
  exit 1
fi

REP="$(realpath "$1")"
if [[ ! -f "$REP" ]]; then
  echo "Not a file: $REP" >&2
  exit 1
fi

OUT_DIR="$(realpath "${2:-$(dirname "$REP")}")"
mkdir -p "$OUT_DIR"

if ! command -v nsys &>/dev/null; then
  echo "nsys not found. Install NVIDIA Nsight Systems." >&2
  exit 1
fi

BASE="$(basename "$REP" .nsys-rep)"
PREFIX="${OUT_DIR}/${BASE}"

echo "[nsys_export_csv] input:  $REP"
echo "[nsys_export_csv] output: $OUT_DIR"

# Summaries suitable for sharing with tools / chat (API + runtime + optional GPU).
# Use cwd + `--output .` so nsys names files from the capture basename.
(
  cd "$OUT_DIR"
  nsys stats --force-export=true \
    --report cuda_api_sum \
    --report cuda_api_gpu_sum \
    --report cuda_gpu_kern_sum \
    --report cuda_kern_exec_sum \
    --report cuda_gpu_mem_time_sum \
    --report osrt_sum \
    --report openmp_sum \
    --report nvtx_sum \
    --format csv \
    --output . \
    "$REP"
)

echo ""
echo "[nsys_export_csv] Done. Look for files named like:"
echo "  ${PREFIX}_cuda_api_sum.csv"
echo "  ${PREFIX}_cuda_api_gpu_sum.csv"
echo "  ${PREFIX}_cuda_gpu_kern_sum.csv   (may be empty on some WSL captures)"
echo "Attach those CSVs or @-mention them in the IDE for analysis."
