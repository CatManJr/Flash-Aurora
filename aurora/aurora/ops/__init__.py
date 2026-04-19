"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

This file includes modifications and original contributions by Catman Jr.;
those portions are licensed under the MIT License (see LICENSE).

Custom CUDA kernels and Triton ops for Aurora inference.
"""

from aurora.ops.triton_adaln import (
    adaptive_layernorm_film_add_residual_forward,
    adaptive_layernorm_film_forward,
)
from aurora.ops.flash_window_attn3d import flash_window_attn_3d_forward as flash_window_attn_forward
from aurora.ops.triton_swin3d import (
    crop_roll_unmerge_windows_triton,
    roll_pad_partition_windows_triton,
)

__all__ = [
    "adaptive_layernorm_film_add_residual_forward",
    "adaptive_layernorm_film_forward",
    "flash_window_attn_forward",
    "crop_roll_unmerge_windows_triton",
    "roll_pad_partition_windows_triton",
]
