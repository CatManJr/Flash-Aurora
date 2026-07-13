"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import math
from typing import Literal

import torch
from torch import nn

__all__ = ["LoRA", "LoRARollout", "LoRAMode"]

LoRAMode = Literal["single", "from_second", "all"]


class LoRA(nn.Module):
    """LoRA adaptation for a linear layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 4,
        alpha: int = 1,
        dropout: float = 0.0,
    ) -> None:
        """Initialise.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            r (int, optional): Rank. Defaults to `4`.
            alpha (int, optional): Alpha. Defaults to `1`.
            dropout (float, optional): Drop-out rate. Defaults to `0.0`.
        """
        super().__init__()

        assert r > 0, "The rank must be strictly positive."
        self.lora_alpha = alpha
        self.r = r

        self.lora_dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty((r, in_features)))
        self.lora_B = nn.Parameter(torch.empty((out_features, r)))
        self.scaling = self.lora_alpha / self.r

        self.init_weights()

    def init_weights(self) -> None:
        """Initialise weights."""
        # Initialise A the same way as the default for `nn.Linear` and set B to zero.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the LoRA adaptation.

        Args:
            x (torch.Tensor): Input to the linear layer.

        Returns:
            torch.Tensor: Additive correction for the output of the linear layer.
        """
        lora_a = self.lora_A.transpose(0, 1).to(dtype=x.dtype)
        lora_b = self.lora_B.transpose(0, 1).to(dtype=x.dtype)
        x = self.lora_dropout(x) @ lora_a @ lora_b
        return x * self.scaling

    def delta_weight(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Return additive weight ``scale * (B @ A)`` for merged linear inference."""
        dw = (self.lora_B @ self.lora_A) * self.scaling
        if device is not None or dtype is not None:
            return dw.to(
                device=device if device is not None else dw.device,
                dtype=dtype if dtype is not None else dw.dtype,
            )
        return dw


class LoRARollout(nn.Module):
    """Per-roll-out-step LoRA finetuning."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: int = 8,
        dropout: float = 0.0,
        max_steps: int = 40,
        mode: LoRAMode = "single",
    ) -> None:
        """Initialise.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            r (int, optional): Rank. Defaults to `4`.
            alpha (int, optional): Alpha. Defaults to `1`.
            dropout (float, optional): Drop-out rate. Defaults to `0.0`.
            max_steps (int, optional): Maximum number of roll-out steps. Defaults to `40`.
            mode (str, optional): Mode. `"single"` uses the same LoRA for all roll-out steps,
                `"from_second"` uses the same LoRA from the second roll-out step on, and `"all"`
                uses a different LoRA for every roll-out step. Defaults to `"single"`.
        """
        super().__init__()

        self.mode = mode
        self.max_steps = max_steps
        lora_layers = max_steps if mode == "all" else 1
        self.loras = nn.ModuleList(
            [
                LoRA(in_features, out_features, r=r, alpha=alpha, dropout=dropout)
                for _ in range(lora_layers)
            ]
        )

    def forward(self, x: torch.Tensor, step: int) -> torch.Tensor:
        """Compute the LoRA adaptation.

        Args:
            x (torch.Tensor): Input to the linear layer.
            step (int): Roll-out step, starting at zero.

        Returns:
            torch.Tensor: Additive correction for the output of the linear layer.
        """
        layer = self.layer_for_step(step)
        if layer is None:
            return 0
        return layer(x)

    def layer_for_step(self, step: int) -> LoRA | None:
        """Return active LoRA layer for this rollout step, else ``None``."""
        assert step >= 0, f"Step must be non-negative, found {step}."
        if step >= self.max_steps:
            return None
        if self.mode == "single":
            return self.loras[0]
        if self.mode == "from_second":
            return None if step == 0 else self.loras[0]
        if self.mode == "all":
            return self.loras[step]
        raise ValueError(f"Invalid mode: {self.mode}")
