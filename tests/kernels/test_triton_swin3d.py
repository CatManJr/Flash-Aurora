"""Copyright (c) Catman Jr. Licensed under the MIT license."""

from __future__ import annotations

from datetime import timedelta

import pytest
import torch

from flash_aurora.aurora.model.film import AdaptiveLayerNorm
from flash_aurora.aurora.model.swin3d import (
    Swin3DTransformerBackbone,
    crop_3d,
    pad_3d,
    window_partition_3d,
    window_reverse_3d,
)
from flash_aurora.aurora.model.util import maybe_adjust_windows
from flash_aurora.aurora.ops.triton_adaln import (
    adaptive_layernorm_film_add_residual_forward,
    adaptive_layernorm_film_forward,
)
from flash_aurora.aurora.ops.triton_gelu import gelu_forward_triton
from flash_aurora.aurora.ops.triton_swin3d_layout import (
    crop_roll_unmerge_windows_triton,
    roll_pad_partition_windows_triton,
)


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for Triton Swin3D tests",
)


def _ref_roll_pad_partition(
    x: torch.Tensor,
    res: tuple[int, int, int],
    window_size: tuple[int, int, int],
    shift_size: tuple[int, int, int],
) -> torch.Tensor:
    C, H, W = res
    B, _, _, _, D = x.shape
    ws, ss = maybe_adjust_windows(window_size, shift_size, res)
    if not all(s == 0 for s in ss):
        shifted_x = torch.roll(x, shifts=(-ss[0], -ss[1], -ss[2]), dims=(1, 2, 3))
    else:
        shifted_x = x
    pad_size = ((-C) % ws[0], (-H) % ws[1], (-W) % ws[2])
    shifted_x = pad_3d(shifted_x, pad_size)
    w = window_partition_3d(shifted_x, ws)
    return w.view(-1, ws[0] * ws[1] * ws[2], D)


def _ref_crop_roll_unmerge(
    windows: torch.Tensor,
    res: tuple[int, int, int],
    window_size: tuple[int, int, int],
    shift_size: tuple[int, int, int],
) -> torch.Tensor:
    C, H, W = res
    B_times_nW, _, D = windows.shape
    ws, ss = maybe_adjust_windows(window_size, shift_size, res)
    pad_size = ((-C) % ws[0], (-H) % ws[1], (-W) % ws[2])
    w2 = windows.view(-1, ws[0], ws[1], ws[2], D)
    sx = pad_3d(
        torch.zeros(
            1,
            C,
            H,
            W,
            D,
            device=windows.device,
            dtype=windows.dtype,
        ),
        pad_size,
    )
    _, pad_C, pad_H, pad_W, _ = sx.shape
    merged = window_reverse_3d(w2, ws, pad_C, pad_H, pad_W)
    merged = crop_3d(merged, pad_size)
    if not all(s == 0 for s in ss):
        out = torch.roll(merged, shifts=(ss[0], ss[1], ss[2]), dims=(1, 2, 3))
    else:
        out = merged
    return out


@requires_cuda
def test_roll_pad_partition_matches_reference() -> None:
    torch.manual_seed(0)
    B, C, H, W, D = 2, 4, 8, 16, 32
    x = torch.randn(B, C, H, W, D, device="cuda", dtype=torch.float32)
    ws = (2, 4, 4)
    ss = (1, 2, 2)
    res = (C, H, W)
    ref = _ref_roll_pad_partition(x, res, ws, ss)
    tr = roll_pad_partition_windows_triton(x, res, ws, ss)
    torch.testing.assert_close(tr, ref, rtol=0, atol=0)


@requires_cuda
def test_crop_roll_unmerge_matches_reference() -> None:
    torch.manual_seed(1)
    C, H, W = 4, 8, 16
    D = 32
    ws = (2, 4, 4)
    ss = (1, 2, 2)
    res = (C, H, W)
    ws2, _ = maybe_adjust_windows(ws, ss, res)
    pad_size = ((-C) % ws2[0], (-H) % ws2[1], (-W) % ws2[2])
    sx = pad_3d(torch.zeros(2, C, H, W, D, device="cuda", dtype=torch.float32), pad_size)
    nW = (
        (sx.shape[1] // ws2[0])
        * (sx.shape[2] // ws2[1])
        * (sx.shape[3] // ws2[2])
    )
    windows = torch.randn(2 * nW, ws2[0] * ws2[1] * ws2[2], D, device="cuda", dtype=torch.float32)
    ref = _ref_crop_roll_unmerge(windows, res, ws, ss)
    tr = crop_roll_unmerge_windows_triton(windows, res, ws, ss)
    torch.testing.assert_close(tr, ref, rtol=0, atol=0)


@requires_cuda
def test_partition_unmerge_roundtrip() -> None:
    torch.manual_seed(2)
    B, C, H, W, D = 1, 4, 8, 16, 64
    x = torch.randn(B, C, H, W, D, device="cuda", dtype=torch.float32)
    res = (C, H, W)
    ws = (2, 4, 4)
    ss = (1, 2, 2)
    w = roll_pad_partition_windows_triton(x, res, ws, ss)
    x_back = crop_roll_unmerge_windows_triton(w, res, ws, ss)
    ref = _ref_crop_roll_unmerge(_ref_roll_pad_partition(x, res, ws, ss), res, ws, ss)
    torch.testing.assert_close(x_back, ref, rtol=0, atol=0)


@requires_cuda
def test_adaptive_layernorm_film_matches_module() -> None:
    torch.manual_seed(3)
    dim, ctx = 256, 256
    m = AdaptiveLayerNorm(dim, ctx, use_triton=False).cuda().eval()
    x = torch.randn(2, 128, dim, device="cuda", dtype=torch.float32)
    c = torch.randn(2, ctx, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        shift, scale = m.ln_modulation(c).unsqueeze(1).chunk(2, dim=-1)
        ref = m.ln(x) * (m.scale_bias + scale) + shift
        tr = adaptive_layernorm_film_forward(
            x, scale, shift, float(m.scale_bias), float(m.ln.eps)
        )
    torch.testing.assert_close(tr, ref, rtol=0, atol=0)


@requires_cuda
def test_adaptive_layernorm_film_add_residual_op_matches_torch() -> None:
    """Direct test of :func:`adaptive_layernorm_film_add_residual_forward` vs PyTorch reference."""
    torch.manual_seed(44)
    dim, ctx = 256, 256
    m = AdaptiveLayerNorm(dim, ctx, use_triton=False).cuda().eval()
    residual = torch.randn(2, 96, dim, device="cuda", dtype=torch.float32)
    x = torch.randn(2, 96, dim, device="cuda", dtype=torch.float32)
    c = torch.randn(2, ctx, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        shift, scale = m.ln_modulation(c).unsqueeze(1).chunk(2, dim=-1)
        ref = residual + m.ln(x) * (m.scale_bias + scale) + shift
        tr = adaptive_layernorm_film_add_residual_forward(
            residual,
            x,
            scale,
            shift,
            float(m.scale_bias),
            float(m.ln.eps),
        )
        composed = residual + adaptive_layernorm_film_forward(
            x, scale, shift, float(m.scale_bias), float(m.ln.eps)
        )
    torch.testing.assert_close(tr, ref, rtol=0, atol=0)
    torch.testing.assert_close(tr, composed, rtol=0, atol=0)


@requires_cuda
def test_adaln_forward_add_residual_bf16_activation_output_fp32() -> None:
    """MLP may feed BF16; fused AdaLN must still return FP32 for the next block."""
    torch.manual_seed(40)
    dim, ctx = 128, 128
    m = AdaptiveLayerNorm(dim, ctx, use_triton=True).cuda().eval()
    residual = torch.randn(2, 32, dim, device="cuda", dtype=torch.float32)
    x = torch.randn(2, 32, dim, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(2, ctx, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        shift, scale = m.ln_modulation(c).unsqueeze(1).chunk(2, dim=-1)
        x_fp32 = x.float()
        ref = residual + m.ln(x_fp32) * (m.scale_bias + scale) + shift
        out = m.forward_add_residual(residual, x, c)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


@requires_cuda
def test_adaln_fp32_out_op_matches_explicit_cast() -> None:
    from flash_aurora.aurora.ops.triton_adaln import adaptive_layernorm_film_add_residual_forward

    torch.manual_seed(45)
    dim = 256
    residual = torch.randn(2, 48, dim, device="cuda", dtype=torch.float32)
    x = torch.randn(2, 48, dim, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(2, 1, dim, device="cuda", dtype=torch.bfloat16)
    shift = torch.randn(2, 1, dim, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        ref = adaptive_layernorm_film_add_residual_forward(
            residual,
            x.float(),
            scale.float(),
            shift.float(),
            0.0,
            1e-5,
            output_fp32=False,
        )
        fused = adaptive_layernorm_film_add_residual_forward(
            residual,
            x,
            scale,
            shift,
            0.0,
            1e-5,
            output_fp32=True,
        )
    assert fused.dtype == torch.float32
    torch.testing.assert_close(fused, ref, rtol=0, atol=0)


@requires_cuda
def test_adaln_forward_add_residual_matches_reference_fp32_out() -> None:
    """Fused FP32-out path matches eager residual + forward for all-FP32."""
    torch.manual_seed(41)
    dim, ctx = 256, 256
    m = AdaptiveLayerNorm(dim, ctx, use_triton=True).cuda().eval()
    residual = torch.randn(2, 64, dim, device="cuda", dtype=torch.float32)
    x = torch.randn(2, 64, dim, device="cuda", dtype=torch.float32)
    c = torch.randn(2, ctx, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        ref = residual + m(x, c)
        fused = m.forward_add_residual(residual, x, c)
    torch.testing.assert_close(fused, ref, rtol=1e-5, atol=1e-5)


@requires_cuda
def test_adaln_forward_add_residual_matches_reference() -> None:
    """Fused residual+AdaLN matches residual + forward (D2)."""
    torch.manual_seed(41)
    dim, ctx = 256, 256
    m = AdaptiveLayerNorm(dim, ctx, use_triton=True).cuda().eval()
    residual = torch.randn(2, 64, dim, device="cuda", dtype=torch.float32)
    x = torch.randn(2, 64, dim, device="cuda", dtype=torch.float32)
    c = torch.randn(2, ctx, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        ref = residual + m(x, c)
        fused = m.forward_add_residual(residual, x, c)
    torch.testing.assert_close(fused, ref, rtol=1e-5, atol=1e-5)


@requires_cuda
def test_adaln_forward_add_residual_large_d() -> None:
    torch.manual_seed(42)
    dim, ctx = 2048, 512
    m = AdaptiveLayerNorm(dim, ctx, use_triton=True).cuda().eval()
    residual = torch.randn(1, 4, dim, device="cuda", dtype=torch.float32)
    x = torch.randn(1, 4, dim, device="cuda", dtype=torch.float32)
    c = torch.randn(1, ctx, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        ref = residual + m(x, c)
        fused = m.forward_add_residual(residual, x, c)
    torch.testing.assert_close(fused, ref, rtol=1e-4, atol=1e-4)


@requires_cuda
def test_swin3d_block_d2_matches_pytorch_adaln() -> None:
    """Block with Triton AdaLN + D2 residual fuse vs PyTorch AdaLN."""
    from flash_aurora.aurora.model.swin3d import Swin3DTransformerBlock

    torch.manual_seed(43)
    dim, heads = 128, 4
    B, C, H, W = 1, 4, 8, 16
    L = C * H * W
    kwargs = dict(
        dim=dim,
        num_heads=heads,
        time_dim=dim,
        window_size=(2, 4, 4),
        shift_size=(0, 0, 0),
        drop_path=0.0,
    )
    block_t = Swin3DTransformerBlock(**kwargs, use_triton_adaln=True).cuda().eval()
    block_r = Swin3DTransformerBlock(**kwargs, use_triton_adaln=False).cuda().eval()
    block_r.load_state_dict(block_t.state_dict())
    x = torch.randn(B, L, dim, device="cuda", dtype=torch.float32)
    c = torch.randn(B, dim, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        y_t = block_t(x, c, (C, H, W), 0)
        y_r = block_r(x, c, (C, H, W), 0)
    torch.testing.assert_close(y_t, y_r, rtol=1e-4, atol=1e-4)


@requires_cuda
def test_triton_gelu_matches_torch() -> None:
    torch.manual_seed(5)
    x = torch.randn(2, 128, 512, device="cuda", dtype=torch.float32)
    ref = torch.nn.functional.gelu(x, approximate="tanh")
    tr = gelu_forward_triton(x)
    torch.testing.assert_close(tr, ref, rtol=1e-6, atol=1e-6)


@requires_cuda
def test_swin3d_backbone_triton_matches_reference() -> None:
    """Same weights: Triton layout + AdaLN vs pure PyTorch path."""
    torch.manual_seed(4)
    kwargs = dict(
        embed_dim=256,
        encoder_depths=(2, 6, 2),
        encoder_num_heads=(4, 8, 16),
        decoder_depths=(2, 6, 2),
        decoder_num_heads=(16, 8, 4),
        window_size=(2, 6, 12),
        use_lora=True,
        lora_mode="single",
        use_triton_mlp=False,
    )
    b_triton = Swin3DTransformerBackbone(
        **kwargs,
        use_triton_layout=True,
        use_triton_adaln=True,
    ).cuda().eval()
    b_ref = Swin3DTransformerBackbone(**kwargs).cuda().eval()
    b_ref.load_state_dict(b_triton.state_dict())

    C, H, W = 4, 32, 64
    L = C * H * W
    x = torch.randn(1, L, 256, device="cuda", dtype=torch.float32)
    lead = timedelta(hours=6)
    with torch.no_grad():
        y_t = b_triton(x, lead_time=lead, rollout_step=0, patch_res=(C, H, W))
        y_r = b_ref(x, lead_time=lead, rollout_step=0, patch_res=(C, H, W))
    torch.testing.assert_close(y_t, y_r, rtol=1e-5, atol=1e-5)


@requires_cuda
@pytest.mark.parametrize(
    ("lora_mode", "rollout_step"),
    [
        ("single", 0),
        ("single", 3),
        ("from_second", 0),
        ("from_second", 2),
        ("all", 0),
        ("all", 2),
    ],
)
def test_swin3d_lora_merged_inference_matches_reference(
    lora_mode: str,
    rollout_step: int,
) -> None:
    torch.manual_seed(7)
    kwargs = dict(
        embed_dim=256,
        encoder_depths=(2, 6, 2),
        encoder_num_heads=(4, 8, 16),
        decoder_depths=(2, 6, 2),
        decoder_num_heads=(16, 8, 4),
        window_size=(2, 6, 12),
        use_lora=True,
        lora_mode=lora_mode,
        lora_steps=4,
        use_triton_layout=True,
        use_triton_adaln=True,
        use_triton_mlp=False,
    )
    b_merge = Swin3DTransformerBackbone(
        **kwargs,
        use_lora_merged_inference=True,
    ).cuda().eval()
    b_ref = Swin3DTransformerBackbone(
        **kwargs,
        use_lora_merged_inference=False,
    ).cuda().eval()
    b_ref.load_state_dict(b_merge.state_dict())

    C, H, W = 4, 32, 64
    L = C * H * W
    x = torch.randn(1, L, 256, device="cuda", dtype=torch.float32)
    lead = timedelta(hours=6)
    with torch.no_grad():
        y_m = b_merge(x, lead_time=lead, rollout_step=rollout_step, patch_res=(C, H, W))
        y_r = b_ref(x, lead_time=lead, rollout_step=rollout_step, patch_res=(C, H, W))
    torch.testing.assert_close(y_m, y_r, rtol=1e-5, atol=1e-5)


@requires_cuda
def test_swin3d_lora_merged_inference_cache_reuse() -> None:
    torch.manual_seed(8)
    b = Swin3DTransformerBackbone(
        embed_dim=256,
        encoder_depths=(2, 6, 2),
        encoder_num_heads=(4, 8, 16),
        decoder_depths=(2, 6, 2),
        decoder_num_heads=(16, 8, 4),
        window_size=(2, 6, 12),
        use_lora=True,
        lora_mode="all",
        lora_steps=4,
        use_lora_merged_inference=True,
    ).cuda().eval()
    C, H, W = 4, 32, 64
    L = C * H * W
    x = torch.randn(1, L, 256, device="cuda", dtype=torch.float32)
    lead = timedelta(hours=6)
    with torch.no_grad():
        _ = b(x, lead_time=lead, rollout_step=2, patch_res=(C, H, W))
        first_cache = len(b.encoder_layers[0].blocks[0].attn._merged_linear_cache)
        _ = b(x, lead_time=lead, rollout_step=2, patch_res=(C, H, W))
        second_cache = len(b.encoder_layers[0].blocks[0].attn._merged_linear_cache)
    assert first_cache > 0
    assert second_cache == first_cache


