"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

OpenMP environment fixes (run before importing torch / numpy).
"""
from __future__ import annotations

import os


def sanitize_openmp_env(*, default: str = "1") -> None:
    """Reset invalid ``OMP_NUM_THREADS`` (e.g. ``0``) so libgomp does not warn."""
    raw = os.environ.get("OMP_NUM_THREADS", "").strip()
    try:
        n = int(raw) if raw else 0
    except ValueError:
        n = 0
    if n < 1:
        os.environ["OMP_NUM_THREADS"] = default
