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


def _purge_shadow_aurora() -> None:
    """Drop a broken namespace ``aurora`` cached before paths were fixed."""
    cached = sys.modules.get("aurora")
    if cached is None or getattr(cached, "__file__", None) is not None:
        return
    for name in list(sys.modules):
        if name == "aurora" or name.startswith("aurora."):
            del sys.modules[name]


def ensure_repo_paths() -> Path:
    """Ensure vendored ``aurora`` imports resolve in notebooks and scripts."""
    root = Path(__file__).resolve().parent.parent
    repo_root = str(root.resolve())
    aurora_root = str((root / "aurora").resolve())

    if aurora_root in sys.path:
        sys.path.remove(aurora_root)
    sys.path.insert(0, aurora_root)

    if repo_root not in sys.path:
        sys.path.append(repo_root)

    _purge_shadow_aurora()
    return root
