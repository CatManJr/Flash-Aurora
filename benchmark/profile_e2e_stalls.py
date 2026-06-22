#!/usr/bin/env python3
"""Profile end-to-end Aurora forward for GPU stalls (idle gaps between CUDA work).

Uses PyTorch Kineto (CUDA activity) to estimate:
  - Active GPU time (sum of kernel/runtime event durations on device streams)
  - Idle gaps between consecutive events on the same stream (launch latency, sync, host)
  - Self-CUDA buckets from torch.profiler (attention, GEMM, memcpy, etc.)

For timeline inspection, export a Chrome trace and open chrome://tracing or Perfetto.

Example::

    CUTE_DSL_ARCH=sm_120a uv run python benchmark/profile_e2e_stalls.py
    CUTE_DSL_ARCH=sm_120a uv run python benchmark/profile_e2e_stalls.py \\
        --tier bf16_mixed --trace-out /tmp/aurora_e2e.json
    CUTE_DSL_ARCH=sm_120a uv run python benchmark/profile_e2e_stalls.py --backbone-only

Install NVIDIA Nsight Systems for deeper stall attribution (dependency chains, memcopies):
    ./aurora/scripts/profile_aurora_small_nsys.sh
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

_BENCH_DIR = Path(__file__).resolve().parent
_REPO = _BENCH_DIR.parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))
import _bootstrap  # noqa: F401, E402
from _asset_root import default_asset_root


from bench_small_pretrained import _load_batch, _purge_gpu  # noqa: E402
from profile_precision_tiers import _aggregate_buckets, _bucket_tier_profile  # noqa: E402
from profiling_swin3d import _top_ops_ms  # noqa: E402


@dataclass
class StreamSpan:
    stream_id: int
    start_us: float
    end_us: float
    name: str
    activity: str


def _event_name(e: object) -> str:
    n = getattr(e, "name", None)
    return n() if callable(n) else str(n)


def _analyze_kineto_gaps(
    events: list,
    *,
    gap_threshold_us: float,
    wall_us: float,
) -> tuple[dict[str, float], dict[str, float], list[tuple[float, float, str, int]]]:
    """Per-stream gap analysis; returns category_ms, stream_stats, largest_gaps."""
    by_stream: dict[int, list[StreamSpan]] = defaultdict(list)
    category_us: dict[str, float] = defaultdict(float)

    for e in events:
        act = str(getattr(e, "activity_type", lambda: "unknown")())
        name = _event_name(e)
        dur_us = float(e.duration_ns()) / 1e3
        start_us = float(e.start_ns()) / 1e3
        end_us = start_us + dur_us

        stream = int(getattr(e, "cuda_stream_id", lambda: -1)())
        if stream < 0:
            stream = int(getattr(e, "device_resource_id", lambda: 0)())

        by_stream[stream].append(
            StreamSpan(stream, start_us, end_us, name, act)
        )

        key = act
        nl = name.lower()
        if "memcpy" in nl or act == "memcpy":
            key = "memcpy"
        elif "sync" in nl or "devicesynchronize" in nl:
            key = "sync"
        elif "kernel" in act or act in ("cuda_driver", "cuda_runtime"):
            if "gemm" in nl or "cutlass" in nl or "cublas" in nl or "mma" in nl:
                key = "kernel_gemm"
            elif "windowattn" in nl or "flash" in nl or "fmha" in nl:
                key = "kernel_attn"
            elif "kernel" in act:
                key = "kernel_other"
            else:
                key = "runtime_other"
        elif act == "overhead":
            key = "profiler_overhead"
        category_us[key] += dur_us

    def _is_compute_span(s: StreamSpan) -> bool:
        nl = s.name.lower()
        if "dtoh" in nl or "pinned" in nl or "profiler" in nl:
            return False
        return any(
            k in nl
            for k in (
                "cutlass", "cublas", "gemm", "windowattn", "fmha", "flash",
                "addmm", "mm", "triton", "adaln", "gelu", "layer_norm",
            )
        )

    compute_streams: set[int] = set()
    for sid, spans in by_stream.items():
        compute_us = sum(
            s.end_us - s.start_us for s in spans if _is_compute_span(s)
        )
        if compute_us > 50.0:
            compute_streams.add(sid)

    gap_us_total = 0.0
    gap_us_compute = 0.0
    active_us_total = 0.0
    compute_active_us = 0.0
    largest_gaps: list[tuple[float, float, str, int]] = []

    t_min = min((s.start_us for spans in by_stream.values() for s in spans), default=0.0)
    t_max = max((s.end_us for spans in by_stream.values() for s in spans), default=wall_us)

    for sid, spans in by_stream.items():
        if not spans:
            continue
        spans.sort(key=lambda s: s.start_us)
        on_compute = sid in compute_streams
        for s in spans:
            dur = s.end_us - s.start_us
            active_us_total += dur
            if on_compute:
                compute_active_us += dur
        for prev, nxt in zip(spans, spans[1:]):
            gap = nxt.start_us - prev.end_us
            if gap >= gap_threshold_us:
                gap_us_total += gap
                if on_compute:
                    gap_us_compute += gap
                    largest_gaps.append((gap, prev.end_us - t_min, prev.name[:60], sid))

    largest_gaps.sort(key=lambda x: -x[0])

  # Span of traced GPU work vs wall clock (gaps on compute streams ~ launch bubbles).
    trace_span_us = max(t_max - t_min, wall_us)
    unaccounted_us = max(0.0, wall_us - trace_span_us)
    category_ms = {k: v / 1e3 for k, v in category_us.items()}
    stats = {
        "wall_ms": wall_us / 1e3,
        "active_sum_ms": active_us_total / 1e3,
        "compute_active_ms": compute_active_us / 1e3,
        "compute_streams": sorted(compute_streams),
        "gap_ms": gap_us_total / 1e3,
        "gap_compute_ms": gap_us_compute / 1e3,
        "unaccounted_ms": unaccounted_us / 1e3,
        "gap_pct_of_wall": 100.0 * gap_us_total / wall_us if wall_us > 0 else 0.0,
        "gap_compute_pct_of_wall": 100.0 * gap_us_compute / wall_us if wall_us > 0 else 0.0,
    }
    return category_ms, stats, largest_gaps[:15]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_asset_root())
    parser.add_argument("--tier", type=str, default="bf16_mixed")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--gap-threshold-us", type=float, default=5.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--backbone-only",
        action="store_true",
        help="Time Swin backbone only (encoder run once outside timed loop).",
    )
    parser.add_argument(
        "--trace-out",
        type=Path,
        default=None,
        help="Chrome trace JSON for Perfetto (default: profiling/aurora_perfetto_<scope>.json).",
    )
    parser.add_argument("--report-out", type=Path, default=None)
    parser.add_argument(
        "--with-stack",
        action="store_true",
        help="Record Python stacks in the trace (larger file, better Perfetto drill-down).",
    )
    args = parser.parse_args()

    import torch
    from torch.autograd.profiler import record_function
    from torch.profiler import ProfilerActivity, profile

    from flash_aurora.aurora import AuroraSmallPretrained

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    dev = torch.device("cuda")
    data_dir = args.data_dir.expanduser().resolve()
    batch = _load_batch(data_dir).to(dev)

    model = AuroraSmallPretrained(use_lora=False, inference_precision=args.tier)
    model.load_checkpoint_local(
        str(data_dir / "aurora-0.25-small-pretrained.ckpt"),
        strict=True,
    )
    model.eval()
    model.to(dev)

    patch_res = None
    backbone_x = None
    rollout_step = 0
    if args.backbone_only:
        from flash_aurora.aurora.model.custom_op_paths import run_with_encoder_decoder_autocast

        _, transformed, patch_res = model._prepare_encoder_batch(batch)
        with torch.inference_mode():
            backbone_x = run_with_encoder_decoder_autocast(
                model.encoder,
                transformed,
                enabled=model.autocast_encoder_decoder,
                lead_time=model.timestep,
            )
        rollout_step = batch.metadata.rollout_step

    def run_once() -> None:
        with torch.inference_mode():
            if args.backbone_only:
                assert patch_res is not None and backbone_x is not None
                with record_function("aurora::backbone"):
                    _ = model._run_backbone(
                        backbone_x,
                        lead_time=model.timestep,
                        patch_res=patch_res,
                        rollout_step=rollout_step,
                    )
            else:
                with record_function("aurora::forward"):
                    with record_function("aurora::prepare_batch"):
                        enc_batch, transformed, pr = model._prepare_encoder_batch(batch)
                    with record_function("aurora::encoder"):
                        from flash_aurora.aurora.model.custom_op_paths import (
                            run_with_encoder_decoder_autocast,
                        )

                        x = run_with_encoder_decoder_autocast(
                            model.encoder,
                            transformed,
                            enabled=model.autocast_encoder_decoder,
                            lead_time=model.timestep,
                        )
                    with record_function("aurora::backbone"):
                        x = model._run_backbone(
                            x,
                            lead_time=model.timestep,
                            patch_res=pr,
                            rollout_step=batch.metadata.rollout_step,
                        )
                    with record_function("aurora::decoder"):
                        pred = run_with_encoder_decoder_autocast(
                            model.decoder,
                            x,
                            enc_batch,
                            enabled=model.autocast_encoder_decoder,
                            lead_time=model.timestep,
                            patch_res=pr,
                        )
                    with record_function("aurora::finish_prediction"):
                        _ = model._finish_prediction(enc_batch, pred)

    scope = "backbone" if args.backbone_only else "full_e2e"
    print(f"[config] tier={args.tier} scope={scope} warmup={args.warmup} repeat={args.repeat}")
    print(f"[config] gap_threshold={args.gap_threshold_us} us (idle between CUDA events on same stream)")

    for _ in range(args.warmup):
        run_once()
    torch.cuda.synchronize()

    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(args.repeat):
        run_once()
    ev1.record()
    torch.cuda.synchronize()
    ms_forward = ev0.elapsed_time(ev1) / args.repeat
    wall_us = ms_forward * 1e3

    if args.trace_out is None:
        scope_tag = "backbone" if args.backbone_only else "full"
        args.trace_out = _REPO / "profiling" / f"aurora_perfetto_{scope_tag}.json"

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=args.with_stack,
    ) as prof:
        for _ in range(args.repeat):
            run_once()
        torch.cuda.synchronize()

    args.trace_out.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(args.trace_out))
    trace_path = args.trace_out.resolve()
    size_mb = trace_path.stat().st_size / (1024 * 1024)
    print(f"\n=== Perfetto trace ===")
    print(f"  file: {trace_path}  ({size_mb:.1f} MB)")
    print(f"  1. Open https://ui.perfetto.dev")
    print(f"  2. Open trace file → pick the JSON above")
    print(f"  3. Timeline: search tracks `aurora::` (encoder / backbone / decoder)")
    print(f"  4. GPU: expand `cuda` / `gpu` rows; empty gaps = launch/sync bubbles")
    if size_mb > 80:
        print(f"  tip: trace is large; next run add `--backbone-only` or drop `--with-stack`")

    kineto = prof.profiler.kineto_results
    events = list(kineto.events()) if kineto is not None else []
    cat_ms, stats, largest_gaps = _analyze_kineto_gaps(
        events,
        gap_threshold_us=args.gap_threshold_us,
        wall_us=wall_us,
    )

    buckets, total_kpi_ms = _aggregate_buckets(prof, use_cuda=True)
    top_names, top_ms = _top_ops_ms(prof, use_cuda=True, top_k=args.top_k)

    print(f"\n=== Forward (CUDA events, no profiler in timed loop) ===")
    print(f"  {ms_forward:.2f} ms / forward")

    print(f"\n=== GPU timeline (Kineto, {len(events)} CUDA events) ===")
    print(f"  wall (timed)          {stats['wall_ms']:.2f} ms")
    print(f"  sum event duration    {stats['active_sum_ms']:.2f} ms  (overlapping streams double-count)")
    print(f"  compute streams       {stats['compute_streams']}  (GEMM/attn/Triton; excludes DtoH/profiler)")
    print(f"  compute-active time   {stats['compute_active_ms']:.2f} ms")
    print(
        f"  idle gaps on compute streams (>={args.gap_threshold_us:.0f} us) "
        f"{stats['gap_compute_ms']:.2f} ms  ({stats['gap_compute_pct_of_wall']:.1f}% of wall)"
    )
    print(
        f"  idle gaps all streams {stats['gap_ms']:.2f} ms  ({stats['gap_pct_of_wall']:.1f}% of wall, "
        "includes profiler DtoH)"
    )

    print(f"\n=== Kineto event categories (sum duration, may overlap across streams) ===")
    for k, v in sorted(cat_ms.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<22} {v:8.2f} ms")

    print(f"\n=== Largest inter-kernel gaps (same stream) ===")
    if not largest_gaps:
        print("  (none above threshold)")
    for gap_us, at_us, after_name, sid in largest_gaps[:10]:
        print(f"  {gap_us:8.1f} us @ {at_us/1e3:7.2f} ms  stream={sid}  after `{after_name}`")

    print(f"\n=== Self-CUDA buckets (torch.profiler, {total_kpi_ms:.1f} ms profiled) ===")
    for b, ms in sorted(buckets.items(), key=lambda kv: -kv[1]):
        pct = 100.0 * ms / total_kpi_ms if total_kpi_ms else 0.0
        print(f"  {b:<22} {ms:8.2f} ms  ({pct:5.1f}%)")

    print(f"\n=== Top-{args.top_k} self-CUDA ops ===")
    for name, ms in zip(top_names, top_ms, strict=True):
        print(f"  {ms:8.2f} ms  {_bucket_tier_profile(name):<18}  {name[:70]}")

    if args.report_out is not None:
        lines = [
            "# Aurora E2E GPU stall profile",
            "",
            f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
            f"- Tier: `{args.tier}`",
            f"- Scope: `{scope}`",
            f"- Forward: **{ms_forward:.2f} ms**",
            "",
            "## Timeline",
            "",
            f"| Metric | ms |",
            f"| --- | ---: |",
            f"| Wall | {stats['wall_ms']:.2f} |",
            f"| Idle gaps (same stream, >={args.gap_threshold_us:.0f} µs) | {stats['gap_ms']:.2f} |",
            f"| Gap % of wall | {stats['gap_pct_of_wall']:.1f}% |",
            "",
            "## Self-CUDA buckets",
            "",
            "| Bucket | ms | % |",
            "| --- | ---: | ---: |",
        ]
        for b, ms in sorted(buckets.items(), key=lambda kv: -kv[1]):
            pct = 100.0 * ms / total_kpi_ms if total_kpi_ms else 0.0
            lines.append(f"| {b} | {ms:.2f} | {pct:.1f} |")
        lines.extend(["", "## Top ops", "", "| ms | op |", "| ---: | --- |"])
        for name, ms in zip(top_names, top_ms, strict=True):
            lines.append(f"| {ms:.3f} | `{name}` |")
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n[report] {args.report_out.resolve()}")

    _purge_gpu(model, batch)


if __name__ == "__main__":
    main()
