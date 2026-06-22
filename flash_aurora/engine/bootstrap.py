from __future__ import annotations

import os
import sys
from pathlib import Path


def sanitize_openmp_env(*, default: str = "1") -> None:
    """Reset invalid ``OMP_NUM_THREADS`` (e.g. ``0``) before torch/numpy load libgomp."""
    raw = os.environ.get("OMP_NUM_THREADS", "").strip()
    try:
        n = int(raw) if raw else 0
    except ValueError:
        n = 0
    if n < 1:
        os.environ["OMP_NUM_THREADS"] = default


sanitize_openmp_env()


def ensure_repo_paths() -> Path:
    """Return git repo root; add to ``sys.path`` only when package is not installed."""
    root = Path(__file__).resolve().parents[2]
    repo_root = str(root)
    try:
        import flash_aurora  # noqa: F401
    except ImportError:
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
    return root
