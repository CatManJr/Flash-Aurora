"""Import before ``torch`` so libgomp sees a valid ``OMP_NUM_THREADS``."""
from __future__ import annotations

import importlib.util
import os

_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "aurora", "aurora", "_openmp_env.py")
)
_spec = importlib.util.spec_from_file_location("aurora._openmp_env", _path)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_mod.sanitize_openmp_env()
