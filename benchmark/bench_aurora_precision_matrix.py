#!/usr/bin/env python3
"""Copyright (c) Catman Jr. Licensed under the MIT license.

Full-model Aurora benchmark: compare five Swin3D optimization tiers on throughput and VRAM.

Tiers (accuracy reference = ``fp32``):

1. ``fp32``             — PyTorch FP32
2. ``pytorch_autocast`` — PyTorch backbone BF16 autocast
3. ``fast_fp32``        — Triton layout + native Perceiver
4. ``tf32_1x``          — ``fast_fp32`` + TF32 backbone matmuls + CuTe TF32 window attention
5. ``bf16_mixed``       — ``fast_fp32`` + explicit BF16 backbone (CuTe attn + BF16 matmuls)

All tiers use native Perceiver (PyTorch SDPA). Each tier rebuilds
the model and fully purges GPU state before timing.

Examples::

    # Default: Aurora 0.25° pretrained, production grid (721×1440), auto batch @ 90% VRAM
    CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_aurora_precision_matrix.py

    # Quick sanity on debug small model
    uv run python benchmark/bench_aurora_precision_matrix.py --model small --preset medium --no-auto-batch

    # Full model, fixed batch, CUDA graph on supported tiers
    CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_aurora_precision_matrix.py \\
        --batch-size 1 --cuda-graph --verify-paths

Unless ``--batch-size`` is set, batch size is auto-probed to target ``--vram-fraction`` (default
90%) of total GPU memory using a conservative fp32 forward before the tier matrix runs.
On ``--preset production`` (721×1440) each probe forward can take several minutes; use
``--batch-size 1`` to skip probing, or watch ``[auto-vram]`` progress lines.
Use ``--preset smoke`` (32×64) only for quick sanity checks; ``bf16_mixed`` may crash there
(N<32 CuTe limitation).

Default grid is ``--preset production`` (721×1440, ERA5 0.25° global, patch_res 4×180×360).
Default model is ``--model full`` (``AuroraPretrained`` + ``aurora-0.25-pretrained.ckpt``).
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
import _bootstrap  # noqa: F401, E402

os.environ.setdefault("CUTE_DSL_ARCH", "sm_120a")

import torch

_REPO = Path(__file__).resolve().parents[1]
_AURORA_PKG = _REPO / "aurora"
if _AURORA_PKG.is_dir():
    sys.path.insert(0, str(_AURORA_PKG))

_DEFAULT_CHECKPOINT_DIR = "/root/autodl-tmp/aurora"

# ERA5 0.25° global (721×1440) → patch_res (4, 180, 360), window N=144 at all stages.
_GRID_PRESETS: dict[str, tuple[int, int]] = {
    "production": (721, 1440),
    "medium": (128, 256),  # patch_res (4, 32, 64), L=8192
    "smoke": (32, 64),  # tiny; bf16_mixed may hit N<32 CuTe limitation
}

_MODEL_PRESETS: dict[str, dict[str, str]] = {
    "full": {
        "checkpoint": "aurora-0.25-pretrained.ckpt",
        "description": "Aurora 0.25° pretrained (embed=512, 48 Swin blocks)",
    },
    "small": {
        "checkpoint": "aurora-0.25-small-pretrained.ckpt",
        "description": "AuroraSmallPretrained debug model (embed=256, 20 Swin blocks)",
    },
}

_DEFAULT_VRAM_FRACTION = 0.90
_DEFAULT_BATCH_CAP = 512
_PROBE_PRECISION = "fp32"  # conservative vs all tiers (weights + activations)

_BENCH_TIERS: tuple[tuple[str, str, str], ...] = (
    ("fp32", "fp32", "PyTorch FP32"),
    ("pytorch_autocast", "pytorch_autocast", "PyTorch backbone BF16 autocast"),
    ("fast_fp32", "fast_fp32", "Triton Swin + native Perceiver"),
    ("tf32_1x", "tf32_1x", "fast_fp32 + TF32 backbone matmuls + CuTe TF32 attn"),
    ("bf16_mixed", "bf16_mixed", "fast_fp32 + explicit BF16 backbone (CuTe + matmuls)"),
)


@dataclass(frozen=True)
class ModelBuildOptions:
    model_name: str
    use_lora: bool = False
    lora_mode: str = "single"
    use_lora_merged_inference: bool = True


@dataclass
class TierResult:
    key: str
    precision: str
    label: str
    ms_per_forward: float
    forwards_per_sec: float
    peak_alloc_mb: float
    peak_reserved_mb: float
    max_abs_diff_vs_baseline: float | None
    mean_abs_diff_vs_baseline: float | None
    max_rel_diff_vs_baseline: float | None
    cosine_sim_vs_baseline: float | None
    cuda_graph: bool = False


def _get_model_class(model_name: str) -> type:
    from aurora import AuroraPretrained, AuroraSmallPretrained

    if model_name == "full":
        return AuroraPretrained
    if model_name == "small":
        return AuroraSmallPretrained
    raise ValueError(f"Unknown model {model_name!r}; expected one of: {', '.join(_MODEL_PRESETS)}")


def _model_common_kwargs(opts: ModelBuildOptions) -> dict[str, Any]:
    return {
        "use_lora": opts.use_lora,
        "lora_mode": opts.lora_mode,
        "use_lora_merged_inference": opts.use_lora_merged_inference,
    }


def _swin_block_count(model: Any) -> int:
    backbone = model.backbone
    return sum(len(layer.blocks) for layer in backbone.encoder_layers) + sum(
        len(layer.blocks) for layer in backbone.decoder_layers
    )


def _model_summary(model: Any) -> str:
    attn = model.backbone.encoder_layers[0].blocks[0].attn
    return (
        f"embed={model.backbone.embed_dim} swin_blocks={_swin_block_count(model)} "
        f"lora_merged={getattr(attn, 'use_lora_merged_inference', False)}"
    )


def _purge_gpu(*objs: Any) -> None:
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _load_batch_synthetic(
    *,
    batch_size: int,
    h: int,
    w: int,
    history: int,
    levels: tuple[int | float, ...],
    device: torch.device,
) -> Any:
    from aurora import Batch, Metadata

    batch = Batch(
        surf_vars={k: torch.randn(batch_size, history, h, w) for k in ("2t", "10u", "10v", "msl")},
        static_vars={k: torch.randn(h, w) for k in ("lsm", "z", "slt")},
        atmos_vars={
            k: torch.randn(batch_size, history, len(levels), h, w) for k in ("z", "u", "v", "t", "q")
        },
        metadata=Metadata(
            lat=torch.linspace(90, -90, h),
            lon=torch.linspace(0, 360, w + 1)[:-1],
            time=(datetime(2020, 6, 1, 12, 0),),
            atmos_levels=levels,
        ),
    )
    return batch.to(device)


def _cuda_oom_like(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if "outofmemory" in msg.replace(" ", "") or "out of memory" in msg:
        return True
    if "cudaerrormemoryallocation" in msg.replace(" ", ""):
        return True
    if type(exc).__name__ in {"OutOfMemoryError", "AcceleratorError"} and "memory" in msg:
        return True
    return False


def _recover_cuda_after_oom() -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()


def _gpu_total_mb(device: torch.device) -> float:
    props = torch.cuda.get_device_properties(device)
    return props.total_memory / 1e6


def _build_model(
    precision: str,
    state_dict: dict[str, torch.Tensor],
    device: torch.device,
    *,
    build_opts: ModelBuildOptions,
) -> Any:
    model_cls = _get_model_class(build_opts.model_name)
    model = model_cls(
        **_model_common_kwargs(build_opts),
        inference_precision=precision,
    )
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        print(
            f"[warn] load_state_dict missing {len(incompatible.missing_keys)} keys "
            f"(first: {incompatible.missing_keys[:3]})"
        )
    if incompatible.unexpected_keys:
        print(
            f"[warn] load_state_dict unexpected {len(incompatible.unexpected_keys)} keys "
            f"(first: {incompatible.unexpected_keys[:3]})"
        )
    model.eval()
    return model.to(device)


def _maybe_capture_cuda_graph(model: Any, batch: Any, *, precision: str, enabled: bool) -> bool:
    if not enabled:
        return False
    from aurora.model.inference_precision import resolve_inference_config

    cfg = resolve_inference_config(precision)
    if cfg is None or cfg.cuda_graph_scope == "off":
        return False
    try:
        model.capture_inference_cuda_graph(batch, warmup_iters=2)
    except RuntimeError as exc:
        print(
            f"[cuda-graph] capture failed for {precision}: {exc}\n"
            "[cuda-graph] continuing with eager forward for this tier"
        )
        model.clear_inference_cuda_graph()
        return False
    print(f"[cuda-graph] captured scope={cfg.cuda_graph_scope} for {precision}")
    return True


def _forward_peak_mb(
    model: Any,
    batch: Any,
    device: torch.device,
) -> float:
    _purge_gpu()
    with torch.inference_mode():
        _ = model.forward(batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return torch.cuda.max_memory_allocated(device) / 1e6
    return float("nan")


def _forward_peak_fits_target(
    *,
    state_dict: dict[str, torch.Tensor],
    batch: Any,
    device: torch.device,
    target_peak_mb: float,
    precision: str,
    build_opts: ModelBuildOptions,
) -> tuple[bool, float]:
    model = _build_model(precision, state_dict, device, build_opts=build_opts)
    try:
        peak_mb = _forward_peak_mb(model, batch, device)
        return peak_mb <= target_peak_mb, peak_mb
    except Exception as exc:
        if _cuda_oom_like(exc):
            _recover_cuda_after_oom()
            return False, float("inf")
        raise
    finally:
        _purge_gpu(model, batch)


def _probe_batch_peak(
    *,
    state_dict: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
    h: int,
    w: int,
    history: int,
    levels: tuple[int | float, ...],
    target_peak_mb: float,
    build_opts: ModelBuildOptions,
) -> tuple[bool, float]:
    """Run one fp32 probe forward; log progress (production grids can take minutes)."""
    print(
        f"[auto-vram] probing batch={batch_size} grid={h}x{w} precision={_PROBE_PRECISION}...",
        flush=True,
    )
    t0 = time.perf_counter()
    batch = _load_batch_synthetic(
        batch_size=batch_size,
        h=h,
        w=w,
        history=history,
        levels=levels,
        device=device,
    )
    fits, peak_mb = _forward_peak_fits_target(
        state_dict=state_dict,
        batch=batch,
        device=device,
        target_peak_mb=target_peak_mb,
        precision=_PROBE_PRECISION,
        build_opts=build_opts,
    )
    elapsed = time.perf_counter() - t0
    if peak_mb == float("inf"):
        note = "OOM"
    elif fits:
        note = "ok"
    else:
        note = "over budget"
    print(
        f"[auto-vram]   batch={batch_size} peak={peak_mb:.0f} MB "
        f"(target {target_peak_mb:.0f} MB) elapsed={elapsed:.1f}s → {note}",
        flush=True,
    )
    return fits, peak_mb


def _auto_batch_size_for_vram(
    *,
    state_dict: dict[str, torch.Tensor],
    device: torch.device,
    h: int,
    w: int,
    history: int,
    levels: tuple[int | float, ...],
    target_fraction: float,
    cap: int,
    build_opts: ModelBuildOptions,
) -> tuple[int, float, float, float]:
    """Pick the largest batch whose fp32 forward peak stays within ``target_fraction`` of total VRAM."""
    total_mb = _gpu_total_mb(device)
    target_peak_mb = total_mb * target_fraction
    print(
        f"[auto-vram] GPU total={total_mb:.0f} MB, target peak={target_peak_mb:.0f} MB "
        f"({target_fraction:.0%}), cap={cap}",
        flush=True,
    )

    fits, peak_mb = _probe_batch_peak(
        state_dict=state_dict,
        device=device,
        batch_size=1,
        h=h,
        w=w,
        history=history,
        levels=levels,
        target_peak_mb=target_peak_mb,
        build_opts=build_opts,
    )
    if not fits:
        print(
            f"[auto-vram] batch=1 alone exceeds {target_fraction:.0%} of VRAM; using batch_size=1.",
            flush=True,
        )
        return 1, peak_mb if peak_mb != float("inf") else total_mb, total_mb, target_peak_mb

    best_batch = 1
    best_peak = peak_mb
    first_fail: int | None = None
    probe = 2
    while probe <= cap:
        fits, peak_mb = _probe_batch_peak(
            state_dict=state_dict,
            device=device,
            batch_size=probe,
            h=h,
            w=w,
            history=history,
            levels=levels,
            target_peak_mb=target_peak_mb,
            build_opts=build_opts,
        )
        if fits:
            best_batch, best_peak = probe, peak_mb
            probe *= 2
            continue
        first_fail = probe
        break

    if first_fail is not None and best_batch + 1 < first_fail:
        lo, hi = best_batch, first_fail - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            fits, peak_mb = _probe_batch_peak(
                state_dict=state_dict,
                device=device,
                batch_size=mid,
                h=h,
                w=w,
                history=history,
                levels=levels,
                target_peak_mb=target_peak_mb,
                build_opts=build_opts,
            )
            if fits:
                lo = mid
                best_batch, best_peak = mid, peak_mb
            else:
                hi = mid - 1

    return best_batch, best_peak, total_mb, target_peak_mb


def _prediction_tensors(pred: Any) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for group in ("surf_vars", "atmos_vars"):
        for name, tensor in getattr(pred, group).items():
            out[f"{group}.{name}"] = tensor.detach().float().cpu()
    return out


def _diff_vs_reference(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> tuple[float, float, float, float]:
    max_diff = 0.0
    max_rel = 0.0
    total = 0.0
    cos_total = 0.0
    count = 0
    for key, ref in reference.items():
        cand = candidate[key]
        diff = (cand - ref).abs()
        max_diff = max(max_diff, float(diff.max().item()))
        denom = ref.abs().clamp_min(1e-6)
        max_rel = max(max_rel, float((diff / denom).max().item()))
        total += float(diff.mean().item())

        ref_flat = ref.flatten().double()
        cand_flat = cand.flatten().double()
        ref_norm = ref_flat.norm()
        cand_norm = cand_flat.norm()
        if ref_norm.item() == 0.0 and cand_norm.item() == 0.0:
            cos = 1.0
        elif ref_norm.item() == 0.0 or cand_norm.item() == 0.0:
            cos = 0.0
        else:
            cos = float(torch.dot(ref_flat, cand_flat).item() / (ref_norm.item() * cand_norm.item()))
        cos_total += cos
        count += 1
    mean_diff = total / max(count, 1)
    mean_cosine = cos_total / max(count, 1)
    return max_diff, mean_diff, max_rel, mean_cosine


def _checkpoint_has_lora(checkpoint_path: Path) -> bool:
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    return any("lora_" in key for key in payload)


def _resolve_lora_for_checkpoint(
    *,
    checkpoint_path: Path,
    use_lora: bool,
) -> bool:
    """Align ``use_lora`` with checkpoint contents; pretrained ckpts have no LoRA tensors."""
    ckpt_has_lora = _checkpoint_has_lora(checkpoint_path)
    if use_lora and not ckpt_has_lora:
        print(
            "[config] checkpoint has no LoRA weights (pretrained base); "
            "building model with use_lora=False"
        )
        return False
    if not use_lora and ckpt_has_lora:
        print(
            "[warn] checkpoint contains LoRA weights; pass --use-lora to load them "
            "(otherwise strict load may fail or LoRA stays uninitialized)"
        )
    return use_lora


def _load_shared_state_dict(checkpoint_path: Path, *, build_opts: ModelBuildOptions) -> dict[str, torch.Tensor]:
    model_cls = _get_model_class(build_opts.model_name)
    model = model_cls(**_model_common_kwargs(build_opts))
    try:
        model.load_checkpoint_local(str(checkpoint_path), strict=True)
    except RuntimeError as exc:
        if build_opts.use_lora:
            print(
                "[warn] strict load failed with use_lora=True; "
                f"retrying with strict=False ({exc.__class__.__name__})"
            )
        else:
            print(f"[warn] strict checkpoint load failed; retrying with strict=False: {exc}")
        model.load_checkpoint_local(str(checkpoint_path), strict=False)
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def _time_forward(
    model: Any,
    batch: Any,
    *,
    warmup: int,
    repeat: int,
    device: torch.device,
) -> tuple[float, float, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model.forward(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeat):
                _ = model.forward(batch)
            end.record()
            torch.cuda.synchronize(device)
            ms_total = start.elapsed_time(end)
            peak_alloc = torch.cuda.max_memory_allocated(device) / 1e6
            peak_reserved = torch.cuda.max_memory_reserved(device) / 1e6
        else:
            import time

            t0 = time.perf_counter()
            for _ in range(repeat):
                _ = model.forward(batch)
            ms_total = (time.perf_counter() - t0) * 1e3
            peak_alloc = float("nan")
            peak_reserved = float("nan")

    ms_per = ms_total / repeat
    return ms_per, peak_alloc, peak_reserved


def _verify_runtime_paths(model: Any, batch: Any) -> None:
    """One-shot sanity check that custom kernels fire on an optimized preset."""
    from collections import Counter

    from aurora.model.custom_op_paths import can_use_cute_qkvpacked, can_use_cute_window_attention
    from aurora.model import swin3d
    from aurora.model.lora import LoRARollout
    import aurora.ops.triton_swin3d_layout as layout_mod
    import aurora.ops.triton_gelu as gelu_mod
    import aurora.ops.triton_adaln as adaln_mod

    counts: Counter[str] = Counter()
    originals: dict[str, Any] = {}

    def patch(module: Any, name: str, counter_key: str) -> None:
        fn = getattr(module, name)
        originals[counter_key] = fn

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            counts[counter_key] += 1
            return fn(*args, **kwargs)

        setattr(module, name, wrapped)

    patch(layout_mod, "roll_pad_partition_windows_triton", "triton_layout")
    patch(gelu_mod, "gelu_forward_triton", "triton_gelu")
    patch(adaln_mod, "adaptive_layernorm_film_add_residual_forward", "triton_adaln_res")

    orig_attn_fwd = swin3d.WindowAttention.forward

    def attn_fwd(self: Any, x: torch.Tensor, mask: torch.Tensor | None = None, rollout_step: int = 0) -> torch.Tensor:
        if isinstance(self.lora_qkv, LoRARollout):
            qkv = self._linear_with_optional_lora_merge(
                x, self.qkv, self.lora_qkv, step=rollout_step, cache_name="qkv"
            )
        else:
            qkv = self.qkv(x) + self.lora_qkv(x, rollout_step)
        attn_dropout = self.attn_drop if self.training else 0.0
        if can_use_cute_qkvpacked(
            qkv,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            cute_enabled=self.use_cute_window_attn,
            training=self.training,
            attn_dropout=attn_dropout,
        ):
            counts["cute_qkvpacked"] += 1
        elif can_use_cute_window_attention(
            qkv,
            enabled=self.use_cute_window_attn,
            training=self.training,
            attn_dropout=attn_dropout,
        ):
            counts["cute_split"] += 1
        else:
            counts["window_sdpa"] += 1
        return orig_attn_fwd(self, x, mask, rollout_step)

    swin3d.WindowAttention.forward = attn_fwd  # type: ignore[method-assign]

    n_blocks = _swin_block_count(model)
    with torch.inference_mode():
        model.forward(batch)

    swin3d.WindowAttention.forward = orig_attn_fwd
    for counter_key, fn in originals.items():
        if counter_key == "triton_layout":
            layout_mod.roll_pad_partition_windows_triton = fn
        elif counter_key == "triton_gelu":
            gelu_mod.gelu_forward_triton = fn
        elif counter_key == "triton_adaln_res":
            adaln_mod.adaptive_layernorm_film_add_residual_forward = fn

    cfg = model.inference_config
    print("\n[verify-paths] runtime kernel counts (single forward):")
    print(
        f"  preset={cfg.precision.value if cfg else 'n/a'} "
        f"triton_layout={counts['triton_layout']}/{n_blocks} "
        f"triton_adaln_res={counts['triton_adaln_res']}/{2 * n_blocks} "
        f"triton_gelu={counts['triton_gelu']}/{n_blocks}"
    )
    print(
        f"  cute_qkvpacked={counts['cute_qkvpacked']}/{n_blocks} "
        f"cute_split={counts['cute_split']} window_sdpa={counts['window_sdpa']}"
    )
    if cfg and cfg.use_triton_layout and counts["triton_layout"] == 0:
        print("[verify-paths] WARN: Triton layout expected but count is 0")
    if cfg and cfg.use_cute_window_attn and counts["cute_qkvpacked"] + counts["cute_split"] == 0:
        print("[verify-paths] WARN: CuTe window attn expected but count is 0")


def _run_tier(
    *,
    key: str,
    precision: str,
    label: str,
    state_dict: dict[str, torch.Tensor],
    batch: Any,
    device: torch.device,
    warmup: int,
    repeat: int,
    baseline_pred: dict[str, torch.Tensor] | None,
    build_opts: ModelBuildOptions,
    cuda_graph: bool,
) -> TierResult:
    print(f"\n{'=' * 72}\n[tier] {key} ({precision}) — {label}\n{'=' * 72}")
    _purge_gpu()

    model = _build_model(precision, state_dict, device, build_opts=build_opts)
    graph_captured = _maybe_capture_cuda_graph(model, batch, precision=precision, enabled=cuda_graph)
    ms_per, peak_alloc, peak_reserved = _time_forward(
        model, batch, warmup=warmup, repeat=repeat, device=device
    )

    max_diff: float | None = None
    mean_diff: float | None = None
    max_rel: float | None = None
    cosine_sim: float | None = None
    if baseline_pred is not None:
        with torch.inference_mode():
            pred = model.forward(batch)
        cand = _prediction_tensors(pred)
        max_diff, mean_diff, max_rel, cosine_sim = _diff_vs_reference(baseline_pred, cand)
        print(
            f"[accuracy] max_abs_diff={max_diff:.6e} mean_abs_diff={mean_diff:.6e} "
            f"max_rel_diff={max_rel:.6e} cosine_sim={cosine_sim:.6f} vs baseline"
        )

    graph_note = " (cuda-graph)" if graph_captured else ""
    print(
        f"[timing] {ms_per:.3f} ms/forward ({1000.0 / ms_per:.2f} forwards/s){graph_note}\n"
        f"[mem] peak allocated={peak_alloc:.1f} MB, peak reserved={peak_reserved:.1f} MB"
    )

    _purge_gpu(model, batch)
    return TierResult(
        key=key,
        precision=precision,
        label=label,
        ms_per_forward=ms_per,
        forwards_per_sec=1000.0 / ms_per,
        peak_alloc_mb=peak_alloc,
        peak_reserved_mb=peak_reserved,
        max_abs_diff_vs_baseline=max_diff,
        mean_abs_diff_vs_baseline=mean_diff,
        max_rel_diff_vs_baseline=max_rel,
        cosine_sim_vs_baseline=cosine_sim,
        cuda_graph=graph_captured,
    )


def _print_summary(results: list[TierResult]) -> None:
    baseline_ms = results[0].ms_per_forward if results else 1.0
    print(f"\n{'=' * 72}\nSummary (baseline = {results[0].key if results else 'n/a'})\n{'=' * 72}")
    header = (
        f"{'tier':<14}{'ms/fwd':>10}{'thrpt':>10}{'peak MB':>10}{'reserved':>10}"
        f"{'speedup':>9}{'max|Δ|':>12}{'mean|Δ|':>12}{'max rel':>10}{'cos sim':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        speedup = baseline_ms / row.ms_per_forward if row.ms_per_forward > 0 else float("nan")
        max_d = (
            f"{row.max_abs_diff_vs_baseline:.3e}"
            if row.max_abs_diff_vs_baseline is not None
            else "   (ref)"
        )
        mean_d = (
            f"{row.mean_abs_diff_vs_baseline:.3e}"
            if row.mean_abs_diff_vs_baseline is not None
            else "   (ref)"
        )
        rel_d = (
            f"{row.max_rel_diff_vs_baseline:.3e}"
            if row.max_rel_diff_vs_baseline is not None
            else "   (ref)"
        )
        cos_d = (
            f"{row.cosine_sim_vs_baseline:.6f}"
            if row.cosine_sim_vs_baseline is not None
            else "   (ref)"
        )
        print(
            f"{row.key:<14}"
            f"{row.ms_per_forward:>10.3f}"
            f"{row.forwards_per_sec:>10.2f}"
            f"{row.peak_alloc_mb:>10.1f}"
            f"{row.peak_reserved_mb:>10.1f}"
            f"{speedup:>8.2f}x"
            f"{max_d:>12}"
            f"{mean_d:>12}"
            f"{rel_d:>10}"
            f"{cos_d:>10}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Aurora full-model precision/throughput matrix benchmark.")
    p.add_argument(
        "--model",
        choices=tuple(_MODEL_PRESETS),
        default="full",
        help="Model preset: full = Aurora 0.25° pretrained (default), small = debug AuroraSmallPretrained.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Fixed batch size. Default: auto-probe from GPU memory (--vram-fraction).",
    )
    p.add_argument(
        "--vram-fraction",
        type=float,
        default=_DEFAULT_VRAM_FRACTION,
        help="Target peak allocated fraction of total GPU memory for auto batch probe (default 0.90).",
    )
    p.add_argument(
        "--batch-cap",
        type=int,
        default=_DEFAULT_BATCH_CAP,
        help="Upper bound for auto batch-size binary search.",
    )
    p.add_argument(
        "--no-auto-batch",
        action="store_true",
        help="Disable VRAM auto-probe; use batch_size=1.",
    )
    p.add_argument(
        "--preset",
        choices=tuple(_GRID_PRESETS),
        default="production",
        help="Synthetic grid preset (overridden by explicit --synthetic-h/--synthetic-w).",
    )
    p.add_argument(
        "--synthetic-h",
        type=int,
        default=None,
        help="Latitude grid height (default: from --preset).",
    )
    p.add_argument(
        "--synthetic-w",
        type=int,
        default=None,
        help="Longitude grid width (default: from --preset).",
    )
    p.add_argument("--history", type=int, default=2)
    p.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[100, 250, 500, 850],
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeat", type=int, default=30)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--checkpoint-dir", type=str, default=_DEFAULT_CHECKPOINT_DIR)
    p.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Explicit checkpoint path (overrides default for --model).",
    )
    p.add_argument(
        "--use-lora",
        action="store_true",
        help="Build with LoRA adapters (for finetuned checkpoints). Default off for pretrained ckpts.",
    )
    p.add_argument(
        "--no-lora-merge",
        action="store_true",
        help="When --use-lora: disable LoRA weight merge during inference (extra qkv/proj GEMMs).",
    )
    p.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Capture CUDA graph before timing on tiers that support it (tf32_1x, bf16_mixed).",
    )
    p.add_argument(
        "--verify-paths",
        action="store_true",
        help="Run one forward with kernel counters on tf32_1x before the tier matrix.",
    )
    p.add_argument("--report-out", type=str, default="")
    args = p.parse_args()

    preset_h, preset_w = _GRID_PRESETS[args.preset]
    synthetic_h = preset_h if args.synthetic_h is None else args.synthetic_h
    synthetic_w = preset_w if args.synthetic_w is None else args.synthetic_w

    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise SystemExit("This benchmark requires CUDA.")

    if synthetic_w % 4 != 0:
        raise SystemExit("--synthetic-w must be a multiple of 4.")
    if synthetic_h % 4 not in (0, 1):
        raise SystemExit("--synthetic-h must satisfy H%4==0 or H%4==1.")
    if not (0.0 < args.vram_fraction <= 1.0):
        raise SystemExit("--vram-fraction must be in (0, 1].")
    if args.batch_cap < 1:
        raise SystemExit("--batch-cap must be >= 1.")

    model_spec = _MODEL_PRESETS[args.model]

    from aurora.model.checkpoint_local import resolve_checkpoint_path

    device = torch.device(args.device)
    ckpt_filename = model_spec["checkpoint"]
    ckpt_path = resolve_checkpoint_path(
        filename=ckpt_filename,
        checkpoint_dir=args.checkpoint_dir,
        explicit_path=args.checkpoint or None,
        allow_hub_download=False,
    )

    use_lora = _resolve_lora_for_checkpoint(checkpoint_path=ckpt_path, use_lora=args.use_lora)
    build_opts = ModelBuildOptions(
        model_name=args.model,
        use_lora=use_lora,
        use_lora_merged_inference=use_lora and not args.no_lora_merge,
    )

    print(f"[startup] loading checkpoint {ckpt_path} ...", flush=True)
    _purge_gpu()
    state_dict = _load_shared_state_dict(ckpt_path, build_opts=build_opts)
    print(f"[startup] checkpoint loaded ({len(state_dict)} tensors on CPU)", flush=True)
    _purge_gpu()

    levels = tuple(args.levels)
    if args.batch_size is not None:
        batch_size = args.batch_size
        auto_vram_note = ""
    elif args.no_auto_batch:
        batch_size = 1
        auto_vram_note = ""
    else:
        batch_size, probe_peak_mb, total_mb, target_mb = _auto_batch_size_for_vram(
            state_dict=state_dict,
            device=device,
            h=synthetic_h,
            w=synthetic_w,
            history=args.history,
            levels=levels,
            target_fraction=args.vram_fraction,
            cap=args.batch_cap,
            build_opts=build_opts,
        )
        auto_vram_note = (
            f" auto_vram={args.vram_fraction:.0%} target={target_mb:.0f}MB "
            f"probe_peak={probe_peak_mb:.0f}MB/{total_mb:.0f}MB"
        )
        print(
            f"[auto-vram] batch_size={batch_size} "
            f"(target {args.vram_fraction:.0%} of {total_mb:.0f} MB → {target_mb:.0f} MB, "
            f"probe peak {probe_peak_mb:.0f} MB, precision={_PROBE_PRECISION})"
        )

    print(f"[config] device={torch.cuda.get_device_name(device)}")
    print(f"[config] model={args.model} — {model_spec['description']}")
    print(
        f"[config] batch={batch_size} preset={args.preset} "
        f"grid={synthetic_h}x{synthetic_w} warmup={args.warmup} repeat={args.repeat}"
        f"{auto_vram_note}"
    )
    print(
        f"[config] use_lora={build_opts.use_lora} lora_merged={build_opts.use_lora_merged_inference} "
        f"cuda_graph={args.cuda_graph} CUTE_DSL_ARCH={os.environ.get('CUTE_DSL_ARCH', '(unset)')}"
    )
    print(f"[checkpoint] {ckpt_path}")

    batch = _load_batch_synthetic(
        batch_size=batch_size,
        h=synthetic_h,
        w=synthetic_w,
        history=args.history,
        levels=levels,
        device=device,
    )

    probe_model = _build_model("tf32_1x", state_dict, device, build_opts=build_opts)
    print(f"[model] {_model_summary(probe_model)}")
    if args.verify_paths:
        _verify_runtime_paths(probe_model, batch)
    _purge_gpu(probe_model, batch)

    results: list[TierResult] = []
    baseline_pred: dict[str, torch.Tensor] | None = None

    _purge_gpu()
    ref_model = _build_model("fp32", state_dict, device, build_opts=build_opts)
    with torch.inference_mode():
        baseline_pred = _prediction_tensors(ref_model.forward(batch))
    _purge_gpu(ref_model)

    for key, precision, label in _BENCH_TIERS:
        result = _run_tier(
            key=key,
            precision=precision,
            label=label,
            state_dict=state_dict,
            batch=batch,
            device=device,
            warmup=args.warmup,
            repeat=args.repeat,
            baseline_pred=None if key == "fp32" else baseline_pred,
            build_opts=build_opts,
            cuda_graph=args.cuda_graph,
        )
        results.append(result)
        batch = _load_batch_synthetic(
            batch_size=batch_size,
            h=synthetic_h,
            w=synthetic_w,
            history=args.history,
            levels=levels,
            device=device,
        )

    _print_summary(results)

    if args.report_out:
        path = Path(args.report_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Aurora precision matrix benchmark",
            "",
            f"- device: {torch.cuda.get_device_name(device)}",
            f"- model: {args.model} ({model_spec['description']})",
            f"- checkpoint: {ckpt_path}",
            f"- batch_size: {batch_size}",
            f"- vram_fraction: {args.vram_fraction}",
            f"- preset: {args.preset}",
            f"- grid: {synthetic_h}x{synthetic_w}",
            f"- lora_merged: {build_opts.use_lora_merged_inference}",
            f"- cuda_graph_flag: {args.cuda_graph}",
            f"- warmup/repeat: {args.warmup}/{args.repeat}",
            "",
            "| tier | precision | ms/forward | forwards/s | peak alloc MB | peak reserved MB | speedup vs baseline | cuda graph | max abs diff | mean abs diff | cosine sim |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
        ]
        baseline_ms = results[0].ms_per_forward
        for row in results:
            speedup = baseline_ms / row.ms_per_forward
            max_d = "" if row.max_abs_diff_vs_baseline is None else f"{row.max_abs_diff_vs_baseline:.6e}"
            mean_d = "" if row.mean_abs_diff_vs_baseline is None else f"{row.mean_abs_diff_vs_baseline:.6e}"
            cos_d = "" if row.cosine_sim_vs_baseline is None else f"{row.cosine_sim_vs_baseline:.6f}"
            graph = "yes" if row.cuda_graph else "no"
            lines.append(
                f"| {row.key} | {row.precision} | {row.ms_per_forward:.3f} | "
                f"{row.forwards_per_sec:.2f} | {row.peak_alloc_mb:.1f} | {row.peak_reserved_mb:.1f} | "
                f"{speedup:.2f}x | {graph} | {max_d} | {mean_d} | {cos_d} |"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n[report] {path.resolve()}")


if __name__ == "__main__":
    main()
