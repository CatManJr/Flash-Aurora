# shellcheck shell=bash
# Resolve `nsys` for Nsight Systems CLI (PATH or bundled with Nsight Compute).
# Source from other scripts:  source "$(dirname "$0")/_nsys_path.sh"

_nsys_find() {
  if command -v nsys &>/dev/null; then
    command -v nsys
    return 0
  fi
  local candidate
  for candidate in \
    /opt/nvidia/nsight-compute/*/host/target-linux-x64/nsys \
    /usr/local/cuda/bin/nsys \
    /usr/local/NVIDIA-Nsight-Systems*/bin/nsys; do
    # shellcheck disable=SC2086
    for candidate in $candidate; do
      if [[ -x "$candidate" ]]; then
        echo "$candidate"
        return 0
      fi
    done
  done
  return 1
}

NSYS_BIN="$(_nsys_find || true)"
if [[ -z "${NSYS_BIN:-}" ]]; then
  echo "nsys not found. Install nsight-systems-cli or Nsight Compute (bundles nsys)." >&2
  exit 1
fi
