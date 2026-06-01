"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

This file includes modifications and original contributions by Catman Jr.;
those portions are licensed under the MIT License (see LICENSE).

`AdaptiveLayerNorm` was inspired by the following file:

    https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L101
"""

import torch
from torch import nn

__all__ = ["AdaptiveLayerNorm"]


class AdaptiveLayerNorm(nn.Module):
    """Adaptive layer normalisation with scale and shift modulation."""

    def __init__(
        self,
        dim: int,
        context_dim: int,
        scale_bias: float = 0,
        use_triton: bool = False,
    ) -> None:
        """Initialise.

        Args:
            dim (int): Input dimension.
            context_dim (int): Dimension of the conditioning signal.
            scale_bias (float, optional): Scale bias to add to the scaling factor. Defaults to `0`.
            use_triton (bool, optional): Use fused CUDA Triton for LayerNorm + modulation when
                inputs are float32 on CUDA. Defaults to `False`.
        """
        super().__init__()

        self.ln = nn.LayerNorm(dim, elementwise_affine=False)
        self.ln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(context_dim, dim * 2))
        self.scale_bias = scale_bias
        self.use_triton = use_triton

        self.init_weights()

    def init_weights(self) -> None:
        """Initialise the weights."""
        nn.init.zeros_(self.ln_modulation[-1].weight)
        nn.init.zeros_(self.ln_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape `(B, L, D)`.
            c (torch.Tensor): Conditioning tensor of shape `(B, D)`.

        Returns:
            torch.Tensor: Output tensor of shape `(B, L, D)`.
        """
        shift, scale = self.ln_modulation(c).unsqueeze(1).chunk(2, dim=-1)
        if self.use_triton and x.is_cuda and x.dtype in (torch.float32, torch.bfloat16):
            from aurora.ops.triton_adaln import adaptive_layernorm_film_forward

            return adaptive_layernorm_film_forward(
                x, scale, shift, float(self.scale_bias), float(self.ln.eps)
            )
        return self.ln(x) * (self.scale_bias + scale) + shift

    def forward_add_residual(
        self, residual: torch.Tensor, x: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        """Return ``residual + self.forward(x, c)`` with one modulation of ``c``.

        When ``use_triton`` and CUDA float32, uses a fused kernel that avoids writing a
        full intermediate AdaLN tensor before the add. ``residual`` must not alias ``x``.

        Args:
            residual: Tensor of shape ``(B, L, D)``.
            x: Same shape; AdaLN input (e.g. attention or MLP output).
            c: Conditioning ``(B, context_dim)``.

        Returns:
            Tensor of shape ``(B, L, D)``.
        """
        shift, scale = self.ln_modulation(c).unsqueeze(1).chunk(2, dim=-1)
        if self.use_triton and x.is_cuda and x.dtype in (torch.float32, torch.bfloat16):
            from aurora.model.custom_op_paths import align_binary_activations
            from aurora.ops.triton_adaln import adaptive_layernorm_film_add_residual_forward

            residual, activation = align_binary_activations(residual, x)
            return adaptive_layernorm_film_add_residual_forward(
                residual, activation, scale, shift, float(self.scale_bias), float(self.ln.eps)
            )
        return residual + self.ln(x) * (self.scale_bias + scale) + shift
