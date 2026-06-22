#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export AURORA_HF_LOCAL_DIR="${AURORA_HF_LOCAL_DIR:-}"

echo "Running flash-aurora library tests..."
uv run pytest tests/aurora tests/kernels tests/engine -m "not integration" "$@"
