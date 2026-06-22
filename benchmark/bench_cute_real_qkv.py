#!/usr/bin/env python3
"""Compare CuTe window attention against SDPA on real ERA5 QKV tensors.

This is a diagnostic benchmark, not a throughput benchmark.  It monkey-patches
``WindowAttention.forward`` so each selected attention call compares the exact
runtime QKV and Swin mask used by ``AuroraPretrained``.

Examples::

    CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_cute_real_qkv.py --precision tf32
    CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_cute_real_qkv.py --precision bf16_mixed --max-records 48
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
import _bootstrap  # noqa: F401, E402


import torch
import torch.nn.functional as F
from einops import rearrange

from _pretrained_era5 import (  # noqa: E402
    _CHECKPOINT_NAME,
    _DEFAULT_ASSET_ROOT,
    build_model,
    load_era5_batch,
    purge_gpu,
)


@dataclass(frozen=True)
class AttnRecord:
    idx: int
    name: str
    shape: tuple[int, int, int, int]
    dtype: str
    has_mask: bool
    qkvpacked: bool
    qkv_max: float
    sdpa_max: float
    cute_max: float
    cute_vs_sdpa_max: float
    cute_vs_sdpa_mean: float
    split_vs_sdpa_max: float | None
    packed_vs_split_max: float | None
    alt_tile_m: int | None
    alt_vs_sdpa_max: float | None
    packed_vs_alt_max: float | None
    max_bwin: int
    max_mask_window: int | None
    max_token: int
    max_head: int
    max_head_dim: int
    sdpa_at_max: float
    cute_at_max: float
    mask_allowed_at_max: int | None
    manual_correct_at_max: float
    manual_complement_at_max: float | None
    manual_nomask_at_max: float


def _sdpa_bnc_from_qkv(
    qkv: torch.Tensor,
    *,
    num_heads: int,
    mask: torch.Tensor | None,
    dropout_p: float,
) -> torch.Tensor:
    bwin, n_tokens, three_c = qkv.shape
    head_dim = three_c // (3 * num_heads)
    qkv_view = qkv.view(bwin, n_tokens, 3, num_heads, head_dim)
    q = qkv_view[:, :, 0].permute(0, 2, 1, 3)
    k = qkv_view[:, :, 1].permute(0, 2, 1, 3)
    v = qkv_view[:, :, 2].permute(0, 2, 1, 3)

    attn_mask = None
    if mask is not None:
        attn_mask = mask.unsqueeze(1).unsqueeze(0)
        batch = q.shape[0] // attn_mask.shape[1]
        attn_mask = attn_mask.repeat(batch, 1, 1, 1, 1).reshape(-1, *attn_mask.shape[2:])

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)
    return rearrange(out, "B H N D -> B N (H D)")


def _split_cute_bnc_from_qkv(
    qkv: torch.Tensor,
    *,
    num_heads: int,
    mask: torch.Tensor | None,
    precision: Any,
) -> torch.Tensor:
    from flash_aurora.aurora.ops.cute import window_attn_fwd_cute

    bwin, n_tokens, three_c = qkv.shape
    head_dim = three_c // (3 * num_heads)
    qkv_view = qkv.view(bwin, n_tokens, 3, num_heads, head_dim)
    q = qkv_view[:, :, 0].permute(0, 2, 1, 3).contiguous()
    k = qkv_view[:, :, 1].permute(0, 2, 1, 3).contiguous()
    v = qkv_view[:, :, 2].permute(0, 2, 1, 3).contiguous()
    out = window_attn_fwd_cute(q, k, v, bias=mask, precision=precision)
    return rearrange(out, "B H N D -> B N (H D)")


def _format_float(value: float | None) -> str:
    if value is None:
        return "      n/a"
    return f"{value:9.4g}"


def _print_records(records: list[AttnRecord]) -> None:
    print()
    print("Real-QKV CuTe attention diff vs SDPA")
    header = (
        f"{'#':>3} {'module':<36} {'shape(B,H,N,D)':<18} {'dtype':<7} {'mask':>4} "
        f"{'packed':>6} {'qkv_max':>9} {'sdpa_max':>9} {'cute_max':>9} "
        f"{'max_abs':>9} {'mean_abs':>9} {'split':>9} {'alt':>9} {'pack-alt':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in records:
        shape = f"{r.shape[0]},{r.shape[1]},{r.shape[2]},{r.shape[3]}"
        print(
            f"{r.idx:3d} {r.name:<36.36} {shape:<18} {r.dtype:<7} "
            f"{str(r.has_mask):>4} {str(r.qkvpacked):>6} "
            f"{r.qkv_max:9.4g} {r.sdpa_max:9.4g} {r.cute_max:9.4g} "
            f"{r.cute_vs_sdpa_max:9.4g} {r.cute_vs_sdpa_mean:9.4g} "
            f"{_format_float(r.split_vs_sdpa_max)} {_format_float(r.alt_vs_sdpa_max)} "
            f"{_format_float(r.packed_vs_alt_max)}"
        )
    print()
    print("Max-error locations")
    loc_header = (
        f"{'#':>3} {'module':<36} {'bwin':>6} {'mask_w':>6} {'token':>6} {'head':>5} "
        f"{'dh':>4} {'sdpa':>10} {'cute':>10} {'manual':>10} {'comp':>10} "
        f"{'nomask':>10} {'allowed':>7}"
    )
    print(loc_header)
    print("-" * len(loc_header))
    for r in records:
        mask_window = "n/a" if r.max_mask_window is None else str(r.max_mask_window)
        print(
            f"{r.idx:3d} {r.name:<36.36} {r.max_bwin:6d} {mask_window:>6} "
            f"{r.max_token:6d} {r.max_head:5d} {r.max_head_dim:4d} "
            f"{r.sdpa_at_max:10.4g} {r.cute_at_max:10.4g} "
            f"{r.manual_correct_at_max:10.4g} {_format_float(r.manual_complement_at_max)} "
            f"{r.manual_nomask_at_max:10.4g} "
            f"{'n/a' if r.mask_allowed_at_max is None else r.mask_allowed_at_max:>7}"
        )


def _patch_window_attention(
    model: Any,
    *,
    max_records: int,
    compare_split: bool,
    alt_tile_m: int | None,
) -> tuple[Any, list[AttnRecord]]:
    from flash_aurora.aurora.model import swin3d
    from flash_aurora.aurora.model.custom_op_paths import (
        backbone_bf16_matmul_active,
        can_use_cute_qkvpacked,
        can_use_cute_window_attention,
        cast_activation_dtype,
    )
    from flash_aurora.aurora.model.lora import LoRARollout
    from flash_aurora.aurora.ops.cute import WinAttnPrecision, window_attn_fwd_cute, window_attn_fwd_cute_qkvpacked

    original_forward = swin3d.WindowAttention.forward
    names = {
        module: name
        for name, module in model.named_modules()
        if isinstance(module, swin3d.WindowAttention)
    }
    records: list[AttnRecord] = []
    call_idx = 0

    def should_record() -> bool:
        return max_records < 0 or len(records) < max_records

    def patched_forward(self: Any, x: torch.Tensor, mask: torch.Tensor | None = None, rollout_step: int = 0) -> torch.Tensor:
        nonlocal call_idx
        idx = call_idx
        call_idx += 1

        bf16_cute_attn = (
            backbone_bf16_matmul_active()
            and self.cute_window_attn_dtype == torch.bfloat16
            and x.is_cuda
            and not torch.is_grad_enabled()
        )
        if x.dtype == torch.bfloat16 and not bf16_cute_attn:
            x = cast_activation_dtype(x, torch.float32)

        if isinstance(self.lora_qkv, LoRARollout):
            qkv = self._linear_with_optional_lora_merge(
                x,
                self.qkv,
                self.lora_qkv,
                step=rollout_step,
                cache_name="qkv",
            )
        else:
            qkv = self.qkv(x) + self.lora_qkv(x, rollout_step)

        attn_dropout = self.attn_drop if self.training else 0.0
        use_cute = can_use_cute_window_attention(
            qkv,
            enabled=self.use_cute_window_attn,
            training=self.training,
            attn_dropout=attn_dropout,
        )
        use_cute_qkvpacked = can_use_cute_qkvpacked(
            qkv,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            cute_enabled=self.use_cute_window_attn,
            training=self.training,
            attn_dropout=attn_dropout,
        )

        bias = None
        if mask is not None:
            bias = mask.to(dtype=torch.float32, device=qkv.device, non_blocking=True)
            if not bias.is_contiguous():
                bias = bias.contiguous()

        qkv_for_cute = qkv
        if use_cute_qkvpacked and qkv_for_cute.dtype != self.cute_window_attn_dtype:
            qkv_for_cute = qkv_for_cute.to(dtype=self.cute_window_attn_dtype, non_blocking=True)

        if use_cute_qkvpacked:
            x_attn = window_attn_fwd_cute_qkvpacked(
                qkv_for_cute,
                self.num_heads,
                bias=bias,
                output_layout="bnc",
            )
        else:
            qkv_re = rearrange(qkv, "B N (qkv H D) -> qkv B H N D", H=self.num_heads, qkv=3)
            q, k, v = qkv_re[0], qkv_re[1], qkv_re[2]
            if use_cute:
                if self.cute_window_attn_dtype == torch.bfloat16 and q.dtype != torch.bfloat16:
                    q, k, v = (
                        q.to(torch.bfloat16, non_blocking=True),
                        k.to(torch.bfloat16, non_blocking=True),
                        v.to(torch.bfloat16, non_blocking=True),
                    )
                precision = (
                    WinAttnPrecision.BF16_MIXED
                    if q.dtype == torch.bfloat16
                    else WinAttnPrecision.TF32_ACC_FP32
                )
                if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
                    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
                x_attn = window_attn_fwd_cute(q, k, v, bias=bias, precision=precision)
            elif mask is not None:
                attn_mask = mask.unsqueeze(1).unsqueeze(0)
                batch = q.shape[0] // attn_mask.shape[1]
                attn_mask = attn_mask.repeat(batch, 1, 1, 1, 1).reshape(-1, *attn_mask.shape[2:])
                x_attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=attn_dropout)
            else:
                x_attn = F.scaled_dot_product_attention(q, k, v, dropout_p=attn_dropout)
            x_attn = rearrange(x_attn, "B H N D -> B N (H D)")

        if use_cute_qkvpacked and should_record():
            bwin, n_tokens, three_c = qkv_for_cute.shape
            head_dim = three_c // (3 * self.num_heads)
            sdpa = _sdpa_bnc_from_qkv(
                qkv_for_cute,
                num_heads=self.num_heads,
                mask=bias,
                dropout_p=attn_dropout,
            )
            diff = (x_attn.float() - sdpa.float()).abs()
            flat_idx = int(diff.argmax().item())
            _bwin, _token, _channel = torch.unravel_index(
                torch.tensor(flat_idx, device=diff.device),
                diff.shape,
            )
            max_bwin = int(_bwin.item())
            max_token = int(_token.item())
            max_head = int(_channel.item() // head_dim)
            max_head_dim = int(_channel.item() % head_dim)
            max_mask_window = None if bias is None else max_bwin % int(bias.shape[0])
            sdpa_at_max = float(sdpa[_bwin, _token, _channel].float().item())
            cute_at_max = float(x_attn[_bwin, _token, _channel].float().item())
            mask_allowed_at_max = None
            qkv_view_for_row = qkv_for_cute.view(bwin, n_tokens, 3, self.num_heads, head_dim)
            q_row = qkv_view_for_row[max_bwin, max_token, 0, max_head].float()
            k_mat = qkv_view_for_row[max_bwin, :, 1, max_head].float()
            v_col = qkv_view_for_row[max_bwin, :, 2, max_head, max_head_dim].float()
            logits = torch.mv(k_mat, q_row) * (1.0 / math.sqrt(head_dim))
            weights_nomask = torch.softmax(logits, dim=-1)
            manual_nomask_at_max = float(torch.dot(weights_nomask, v_col).item())
            manual_complement_at_max: float | None = None
            if bias is not None and max_mask_window is not None:
                row = bias[max_mask_window, max_token]
                allowed = row >= -50.0
                mask_allowed_at_max = int(allowed.sum().item())
                masked_logits = logits.masked_fill(~allowed, -float("inf"))
                weights = torch.softmax(masked_logits, dim=-1)
                manual_correct_at_max = float(torch.dot(weights, v_col).item())
                if mask_allowed_at_max < row.numel():
                    comp_logits = logits.masked_fill(allowed, -float("inf"))
                    comp_weights = torch.softmax(comp_logits, dim=-1)
                    manual_complement_at_max = float(torch.dot(comp_weights, v_col).item())
            else:
                manual_correct_at_max = manual_nomask_at_max
            split_max: float | None = None
            packed_split_max: float | None = None
            alt_max: float | None = None
            packed_alt_max: float | None = None
            if compare_split:
                precision = (
                    WinAttnPrecision.BF16_MIXED
                    if qkv_for_cute.dtype == torch.bfloat16
                    else WinAttnPrecision.TF32_ACC_FP32
                )
                split = _split_cute_bnc_from_qkv(
                    qkv_for_cute,
                    num_heads=self.num_heads,
                    mask=bias,
                    precision=precision,
                )
                split_diff = (split.float() - sdpa.float()).abs()
                split_max = float(split_diff.max().item())
                packed_split_max = float((x_attn.float() - split.float()).abs().max().item())
            if alt_tile_m is not None:
                alt = window_attn_fwd_cute_qkvpacked(
                    qkv_for_cute,
                    self.num_heads,
                    bias=bias,
                    tile_m=alt_tile_m,
                    output_layout="bnc",
                )
                alt_diff = (alt.float() - sdpa.float()).abs()
                alt_max = float(alt_diff.max().item())
                packed_alt_max = float((x_attn.float() - alt.float()).abs().max().item())
            records.append(
                AttnRecord(
                    idx=idx,
                    name=names.get(self, "<unnamed>"),
                    shape=(bwin, self.num_heads, n_tokens, head_dim),
                    dtype=str(qkv_for_cute.dtype).replace("torch.", ""),
                    has_mask=bias is not None,
                    qkvpacked=use_cute_qkvpacked,
                    qkv_max=float(qkv_for_cute.float().abs().max().item()),
                    sdpa_max=float(sdpa.float().abs().max().item()),
                    cute_max=float(x_attn.float().abs().max().item()),
                    cute_vs_sdpa_max=float(diff.max().item()),
                    cute_vs_sdpa_mean=float(diff.mean().item()),
                    split_vs_sdpa_max=split_max,
                    packed_vs_split_max=packed_split_max,
                    alt_tile_m=alt_tile_m,
                    alt_vs_sdpa_max=alt_max,
                    packed_vs_alt_max=packed_alt_max,
                    max_bwin=max_bwin,
                    max_mask_window=max_mask_window,
                    max_token=max_token,
                    max_head=max_head,
                    max_head_dim=max_head_dim,
                    sdpa_at_max=sdpa_at_max,
                    cute_at_max=cute_at_max,
                    mask_allowed_at_max=mask_allowed_at_max,
                    manual_correct_at_max=manual_correct_at_max,
                    manual_complement_at_max=manual_complement_at_max,
                    manual_nomask_at_max=manual_nomask_at_max,
                )
            )

        if bf16_cute_attn and x_attn.dtype == torch.bfloat16:
            x_attn = cast_activation_dtype(x_attn, torch.float32)
        if isinstance(self.lora_proj, LoRARollout):
            x_out = self._linear_with_optional_lora_merge(
                x_attn,
                self.proj,
                self.lora_proj,
                step=rollout_step,
                cache_name="proj",
            )
        else:
            x_out = self.proj(x_attn) + self.lora_proj(x_attn, rollout_step)
        return self.proj_drop(x_out)

    swin3d.WindowAttention.forward = patched_forward  # type: ignore[method-assign]
    return original_forward, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=_DEFAULT_ASSET_ROOT)
    parser.add_argument("--era5-cache", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--valid-time", type=str, default="2023-01-01T06:00:00")
    parser.add_argument("--time-index", type=int, default=1)
    parser.add_argument("--precision", choices=("tf32", "bf16_mixed", "bf16"), default="tf32")
    parser.add_argument(
        "--max-records",
        type=int,
        default=12,
        help="Number of attention calls to diagnose; use -1 for all calls.",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Skip split-CuTe comparison and only compare production qkvpacked CuTe.",
    )
    parser.add_argument(
        "--alt-tile-m",
        type=int,
        default=None,
        help="Also run qkvpacked CuTe with this tile_m on the same QKV and compare it to SDPA.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    from flash_aurora.aurora.model import swin3d

    device = torch.device("cuda")
    asset_root = args.asset_root.expanduser().resolve()
    checkpoint = (args.checkpoint or asset_root / _CHECKPOINT_NAME).expanduser().resolve()
    valid_time = datetime.fromisoformat(args.valid_time)

    batch = load_era5_batch(
        asset_root,
        era5_cache=args.era5_cache,
        valid_time=valid_time,
        time_index=args.time_index,
    ).to(device)
    model = build_model(args.precision, checkpoint, device)

    original_forward, records = _patch_window_attention(
        model,
        max_records=args.max_records,
        compare_split=not args.no_split,
        alt_tile_m=args.alt_tile_m,
    )
    try:
        with torch.inference_mode():
            _ = model.forward(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    finally:
        swin3d.WindowAttention.forward = original_forward  # type: ignore[method-assign]
        purge_gpu(model, batch)

    print(f"[config] precision={args.precision} records={len(records)} checkpoint={checkpoint}")
    _print_records(records)


if __name__ == "__main__":
    main()
