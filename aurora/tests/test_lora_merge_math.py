"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

This file includes modifications and original contributions by Catman Jr.;
those portions are licensed under the MIT License (see LICENSE).

Unit tests: LoRA math and merged-linear path vs PyTorch reference (Linear + LoRA).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from aurora.model.lora import LoRA, LoRARollout

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for merged F.linear alignment test",
)


def test_lora_delta_weight_matches_b_a_scale() -> None:
    """delta_weight() == (B @ A) * scaling."""
    torch.manual_seed(0)
    lora = LoRA(in_features=16, out_features=32, r=4, alpha=8)
    dw = lora.delta_weight()
    manual = (lora.lora_B @ lora.lora_A) * lora.scaling
    torch.testing.assert_close(dw, manual, rtol=0, atol=0)


def test_lora_forward_matches_matmul_reference() -> None:
    """LoRA(x) matches explicit low-rank matmul (eval, dropout inactive)."""
    torch.manual_seed(1)
    lora = LoRA(in_features=16, out_features=32, r=4, alpha=8).eval()
    x = torch.randn(2, 10, 16)
    with torch.no_grad():
        y_lora = lora(x)
        y_ref = (x @ lora.lora_A.T @ lora.lora_B.T) * lora.scaling
    torch.testing.assert_close(y_lora, y_ref, rtol=0, atol=0)


@requires_cuda
def test_f_linear_merge_matches_linear_plus_lora_rollout() -> None:
    """F.linear(x, W + ΔW, b) == Linear(x) + LoRARollout(x) for active step."""
    torch.manual_seed(2)
    dim, out_features = 64, 192
    linear = nn.Linear(dim, out_features, bias=True).cuda()
    rollout = LoRARollout(
        dim, out_features, r=8, alpha=8, dropout=0.0, max_steps=8, mode="single"
    ).cuda()
    x = torch.randn(4, 32, dim, device="cuda", dtype=torch.float32)
    step = 0
    with torch.no_grad():
        ref = linear(x) + rollout(x, step)
        layer = rollout.layer_for_step(step)
        assert layer is not None
        dw = layer.delta_weight(device=linear.weight.device, dtype=linear.weight.dtype)
        merged_w = linear.weight + dw
        out = F.linear(x, merged_w, linear.bias)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


@requires_cuda
@pytest.mark.parametrize("mode,step", [("single", 0), ("from_second", 1), ("all", 2)])
def test_f_linear_merge_matches_from_second_and_all(
    mode: str,
    step: int,
) -> None:
    """Same merge identity for modes where LoRA is active at ``step``."""
    torch.manual_seed(3 + step)
    dim, out_features = 48, 96
    linear = nn.Linear(dim, out_features, bias=True).cuda()
    rollout = LoRARollout(
        dim, out_features, r=4, alpha=4, dropout=0.0, max_steps=8, mode=mode
    ).cuda()
    x = torch.randn(2, 16, dim, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        ref = linear(x) + rollout(x, step)
        layer = rollout.layer_for_step(step)
        if layer is None:
            pytest.skip("step inactive for this mode")
        dw = layer.delta_weight(device=linear.weight.device, dtype=linear.weight.dtype)
        out = F.linear(x, linear.weight + dw, linear.bias)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
