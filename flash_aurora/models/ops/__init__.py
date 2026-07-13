"""Copyright (c) Catman Jr. Licensed under the MIT license.

This file includes modifications and original contributions by Catman Jr.;
those portions are licensed under the MIT License (see LICENSE).

Custom Triton and CuTeDSL ops for Aurora inference (see subpackages for third-party references).
"""

from flash_aurora.aurora.ops.triton_adaln import (
    adaptive_layernorm_film_add_residual_forward,
    adaptive_layernorm_film_forward,
)
from flash_aurora.aurora.ops.triton_perceiver_ln import (
    layernorm_affine_add_residual_forward,
    layernorm_affine_forward,
)
from flash_aurora.aurora.ops.cute import WinAttnPrecision, window_attn_fwd_cute
from flash_aurora.aurora.ops.triton_swin3d_layout import (
    crop_roll_unmerge_windows_triton,
    roll_pad_partition_windows_triton,
)

__all__ = [
    "adaptive_layernorm_film_add_residual_forward",
    "adaptive_layernorm_film_forward",
    "layernorm_affine_add_residual_forward",
    "layernorm_affine_forward",
    "WinAttnPrecision",
    "window_attn_fwd_cute",
    "crop_roll_unmerge_windows_triton",
    "roll_pad_partition_windows_triton",
]
