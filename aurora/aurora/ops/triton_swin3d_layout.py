"""Copyright (c) Catman Jr. Licensed under the MIT license.

Triton kernels for Swin3D window layout only: fused cyclic shift + zero-pad +
3D window partition, and the inverse (crop / inverse shift / merge).

Does **not** implement attention or the full Swin block — those live in
:mod:`aurora.model.swin3d` (and optionally other ``ops`` modules).

Inference-only; numerically matches :mod:`aurora.model.swin3d` for the same ``ws``, ``ss``, and ``res``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from aurora.model.util import maybe_adjust_windows


def _get_two_sided_padding(H_padding: int, W_padding: int) -> tuple[int, int, int, int]:
    if H_padding:
        padding_top = H_padding // 2
        padding_bottom = H_padding - padding_top
    else:
        padding_top = padding_bottom = 0
    if W_padding:
        padding_left = W_padding // 2
        padding_right = W_padding - padding_left
    else:
        padding_left = padding_right = 0
    return padding_left, padding_right, padding_top, padding_bottom


def _get_three_sided_padding(
    C_padding: int,
    H_padding: int,
    W_padding: int,
) -> tuple[int, int, int, int, int, int]:
    if C_padding:
        pad_front = C_padding // 2
        pad_back = C_padding - pad_front
    else:
        pad_front = pad_back = 0
    return (
        *_get_two_sided_padding(H_padding, W_padding),
        pad_front,
        pad_back,
    )


def roll_pad_partition_windows_triton(
    x: torch.Tensor,
    res: tuple[int, int, int],
    window_size: tuple[int, int, int],
    shift_size: tuple[int, int, int],
) -> torch.Tensor:
    """Fused cyclic shift, zero-pad, and 3D window partition."""
    if x.device.type != "cuda" or x.dtype != torch.float32:
        raise ValueError("roll_pad_partition_windows_triton requires CUDA float32 input.")
    B, C, H, W, D = x.shape
    ws, ss = maybe_adjust_windows(window_size, shift_size, res)
    pad_size = ((-C) % ws[0], (-H) % ws[1], (-W) % ws[2])
    pleft, pright, ptop, pbottom, pfront, pback = _get_three_sided_padding(*pad_size)
    Cp = C + pfront + pback
    Hp = H + ptop + pbottom
    Wp = W + pleft + pright
    C1, H1, W1 = Cp // ws[0], Hp // ws[1], Wp // ws[2]
    nW = C1 * H1 * W1
    N = ws[0] * ws[1] * ws[2]
    out = torch.empty((B * nW, N, D), device=x.device, dtype=x.dtype)
    total = B * nW * N * D
    BLOCK = 256
    grid = (triton.cdiv(total, BLOCK),)
    _roll_pad_partition_kernel[grid](
        x,
        out,
        B,
        C,
        H,
        W,
        D,
        ws[0],
        ws[1],
        ws[2],
        ss[0],
        ss[1],
        ss[2],
        pfront,
        ptop,
        pleft,
        C1,
        H1,
        W1,
        nW,
        N,
        total,
        BLOCK=BLOCK,
    )
    return out


def crop_roll_unmerge_windows_triton(
    windows: torch.Tensor,
    res: tuple[int, int, int],
    window_size: tuple[int, int, int],
    shift_size: tuple[int, int, int],
) -> torch.Tensor:
    """Inverse of :func:`roll_pad_partition_windows_triton`."""
    if windows.device.type != "cuda" or windows.dtype != torch.float32:
        raise ValueError("crop_roll_unmerge_windows_triton requires CUDA float32 input.")
    B_times_nW, N, D = windows.shape
    C, H, W = res
    ws, ss = maybe_adjust_windows(window_size, shift_size, res)
    pad_size = ((-C) % ws[0], (-H) % ws[1], (-W) % ws[2])
    pleft, pright, ptop, pbottom, pfront, pback = _get_three_sided_padding(*pad_size)
    Cp = C + pfront + pback
    Hp = H + ptop + pbottom
    Wp = W + pleft + pright
    C1, H1, W1 = Cp // ws[0], Hp // ws[1], Wp // ws[2]
    nW = C1 * H1 * W1
    assert B_times_nW % nW == 0
    B = B_times_nW // nW
    assert N == ws[0] * ws[1] * ws[2]
    out = torch.empty((B, C, H, W, D), device=windows.device, dtype=windows.dtype)
    total = B * C * H * W * D
    BLOCK = 256
    grid = (triton.cdiv(total, BLOCK),)
    _crop_roll_unmerge_kernel[grid](
        windows,
        out,
        B,
        C,
        H,
        W,
        D,
        ws[0],
        ws[1],
        ws[2],
        ss[0],
        ss[1],
        ss[2],
        pfront,
        ptop,
        pleft,
        C1,
        H1,
        W1,
        nW,
        N,
        total,
        BLOCK=BLOCK,
    )
    return out


@triton.jit
def _roll_pad_partition_kernel(
    x_ptr,
    out_ptr,
    B,
    C,
    H,
    W,
    D,
    Wc,
    Wh,
    Ww,
    ss0,
    ss1,
    ss2,
    pfront,
    ptop,
    pleft,
    C1,
    H1,
    W1,
    nW,
    N,
    total,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    idx = offs
    d = idx % D
    idx = idx // D
    local_n = idx % N
    idx = idx // N
    w_global = idx
    b = w_global // nW
    w_win = w_global % nW
    # local_n = wc * (Wh*Ww) + wh * Ww + ww (Ww fastest; matches einops window layout)
    wc = local_n // (Wh * Ww)
    rem = local_n % (Wh * Ww)
    wh = rem // Ww
    ww = rem % Ww
    c1 = w_win // (H1 * W1)
    rem = w_win % (H1 * W1)
    h1 = rem // W1
    w1 = rem % W1
    pc = c1 * Wc + wc
    ph = h1 * Wh + wh
    pw = w1 * Ww + ww
    cs = pc - pfront
    hs = ph - ptop
    ws_ = pw - pleft
    in_shifted = (cs >= 0) & (cs < C) & (hs >= 0) & (hs < H) & (ws_ >= 0) & (ws_ < W)
    xc = (cs + ss0) % C
    xh = (hs + ss1) % H
    xw = (ws_ + ss2) % W
    x_lin = (((b * C + xc) * H + xh) * W + xw) * D + d
    val = tl.load(x_ptr + x_lin, mask=in_shifted & mask, other=0.0)
    out_lin = (w_global * N + local_n) * D + d
    tl.store(out_ptr + out_lin, val, mask=mask)


@triton.jit
def _crop_roll_unmerge_kernel(
    win_ptr,
    out_ptr,
    B,
    C,
    H,
    W,
    D,
    Wc,
    Wh,
    Ww,
    ss0,
    ss1,
    ss2,
    pfront,
    ptop,
    pleft,
    C1,
    H1,
    W1,
    nW,
    N,
    total,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    idx = offs
    d = idx % D
    idx = idx // D
    w = idx % W
    idx = idx // W
    h = idx % H
    idx = idx // H
    c = idx % C
    b = idx // C
    cs = (c - ss0 + C) % C
    hs = (h - ss1 + H) % H
    ws__ = (w - ss2 + W) % W
    pc = cs + pfront
    ph = hs + ptop
    pw = ws__ + pleft
    c1 = pc // Wc
    wc = pc % Wc
    h1 = ph // Wh
    wh = ph % Wh
    w1 = pw // Ww
    ww = pw % Ww
    win_idx = c1 * (H1 * W1) + h1 * W1 + w1
    local_n = wc * (Wh * Ww) + wh * Ww + ww
    win_lin = (b * nW + win_idx) * N + local_n
    v = tl.load(win_ptr + win_lin * D + d, mask=mask)
    out_off = ((b * C + c) * H + h) * W + w
    tl.store(out_ptr + out_off * D + d, v, mask=mask)
