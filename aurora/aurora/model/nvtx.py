"""Copyright (c) Catman Jr. Licensed under the MIT license.

Lightweight NVTX ranges for Nsight Systems (``--trace=nvtx``)."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator


def nvtx_enabled() -> bool:
    return os.environ.get("AURORA_NVTX", "0").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


@contextmanager
def nvtx_range(name: str) -> Iterator[None]:
    """Push/pop an NVTX range on the current CUDA device (no-op if disabled)."""
    if not nvtx_enabled():
        yield
        return
    import torch

    if not torch.cuda.is_available():
        yield
        return
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()
