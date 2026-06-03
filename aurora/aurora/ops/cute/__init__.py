"""Copyright (c) Catman Jr. Licensed under the MIT license.

This file includes modifications and original contributions by Catman Jr.;
those portions are licensed under the MIT License (see LICENSE).

CuTe DSL-based ops for Aurora window attention.

BF16 mixed precision uses the CuTeDSL kernel in this package.
FP32 (strict / TF32) is delegated to torch SDPA or PyTorch fallbacks.
See submodules for flash-attn / CUTLASS reference notes.
"""

from aurora.ops.cute.window_attn_fwd import (
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
