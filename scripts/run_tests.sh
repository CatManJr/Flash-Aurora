#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export AURORA_HF_LOCAL_DIR="${AURORA_HF_LOCAL_DIR:-}"

echo "Running aurora model tests..."
pytest -c aurora/pyproject.toml aurora/tests/

echo "Running engine tests (unit)..."
pytest tests/ -m "not integration"
