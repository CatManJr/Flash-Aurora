"""Copyright (c) Catman Jr. Licensed under the MIT license.

This file includes modifications and original contributions by Catman Jr.;
those portions are licensed under the MIT License (see LICENSE).

CuTe DSL-based ops for Aurora window attention.

All three precision modes are served by CuTeDSL kernels in this package: BF16
mixed by ``_kernel_bf16.py``, 1xTF32 by ``_kernel_tf32.py``, and 3xTF32 by
``_kernel_tf32x3.py``.  Each keeps QK and PV at the same precision.
See submodules for flash-attn / CUTLASS reference notes.
"""

from flash_aurora.models.ops.cute.window_attn_fwd import (
    _CUTE_KERNEL_VERSION,
    WinAttnPrecision,
    window_attn_dispatch,
    window_attn_fwd_cute,
    window_attn_fwd_cute_qkvpacked,
)

__all__ = [
    "_CUTE_KERNEL_VERSION",
    "WinAttnPrecision",
    "window_attn_dispatch",
    "window_attn_fwd_cute",
    "window_attn_fwd_cute_qkvpacked",
]
