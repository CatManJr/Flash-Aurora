"""Pytest helpers for benchmark modules."""

from __future__ import annotations

import sys
from pathlib import Path

_BENCH_DIR = Path(__file__).resolve().parents[2] / "benchmark"
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

import _bootstrap  # noqa: F401
