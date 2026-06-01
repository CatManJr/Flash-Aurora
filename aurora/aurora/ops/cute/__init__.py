"""CuTe DSL-based ops for Aurora window attention.

BF16 mixed precision uses the CuTeDSL kernel in this package.
FP32 (strict / TF32) is delegated to torch SDPA or the Triton kernel.
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
