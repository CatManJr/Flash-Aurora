"""CuTe DSL compile-target detection (``CUTE_DSL_ARCH``).

Must run before the first ``cutlass`` import.  When ``CUTE_DSL_ARCH`` is unset and
CUDA is available, :func:`ensure_cute_dsl_arch` picks a target that matches the
local GPU so JIT kernels load on the current device.

Cross-compile for another machine by setting ``CUTE_DSL_ARCH`` explicitly, e.g.
``CUTE_DSL_ARCH=sm_80`` on a Blackwell box when building for A100.
"""
from __future__ import annotations

import os
from typing import Optional

_ENV_VAR = "CUTE_DSL_ARCH"


def cute_dsl_arch_for_capability(major: int, minor: int) -> str:
    """Map ``torch.cuda.get_device_capability()`` to a CuTe DSL arch string."""
    sm = major * 10 + minor
    if sm >= 120:
        return "sm_120a" if major == 12 else f"sm_{major}{minor}a"
    if sm >= 100:
        return "sm_100a" if major == 10 else f"sm_{major}{minor}a"
    if sm == 90:
        return "sm_90"
    if sm == 89:
        return "sm_89"
    if sm == 86:
        return "sm_86"
    if sm == 80:
        return "sm_80"
    if sm == 75:
        return "sm_75"
    if sm == 70:
        return "sm_70"
    return f"sm_{major}{minor}"


def detect_cute_dsl_arch(*, device_index: int = 0) -> Optional[str]:
    """Return the recommended ``CUTE_DSL_ARCH`` for ``device_index``, or ``None``."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability(device_index)
    return cute_dsl_arch_for_capability(major, minor)


def ensure_cute_dsl_arch(*, device_index: int = 0) -> Optional[str]:
    """Set ``CUTE_DSL_ARCH`` when unset; return the active target arch string."""
    current = os.environ.get(_ENV_VAR, "").strip()
    if current:
        return current
    detected = detect_cute_dsl_arch(device_index=device_index)
    if detected:
        os.environ[_ENV_VAR] = detected
    return detected


def active_cute_dsl_arch() -> Optional[str]:
    """Return ``CUTE_DSL_ARCH`` if set, otherwise detect without mutating the env."""
    current = os.environ.get(_ENV_VAR, "").strip()
    if current:
        return current
    return detect_cute_dsl_arch()
