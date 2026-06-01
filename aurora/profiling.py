#!/usr/bin/env python3
"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

This file includes modifications and original contributions by Catman Jr.;
those portions are licensed under the MIT License (see LICENSE).

End-to-end Aurora profiling tuned for modest GPUs (e.g. ~12GB VRAM).

Runs ``AuroraSmallPretrained`` + optional multi-step ``rollout`` under
:class:`torch.profiler.profile`, with warmup and CUDA memory stats.

Dependencies (same as the Aurora package): ``torch``, ``huggingface-hub``,
``numpy``, ``scipy`` (for HF test batch static regridding).

Usage (from the repository root, with the ``aurora`` package on ``PYTHONPATH``)::

    uv run python aurora/profiling.py
    uv run python aurora/profiling.py --repeat 1
    uv run python aurora/profiling.py --stress --synthetic
    uv run python aurora/profiling.py --batch-size 8 --synthetic
    uv run python aurora/profiling.py --trace-out /tmp/aurora_trace.json

For Nsight Systems (``nsys profile``), use ``--no-torch-profiler`` so the capture is not
mixed with ``torch.profiler``. For steady-state GPU timelines, use
``--cuda-profiler-api`` with ``nsys profile --capture-range=cudaProfilerApi``; see
``aurora/scripts/profile_aurora_small_nsys.sh``.

Use ``--stress`` or ``--batch-size N`` (often with ``--synthetic``) for VRAM/latency
stress tests; HF test tensors repeated to large batch may OOM before synthetic.

Peak VRAM is often only a few GB here because **AuroraSmall**, **batch=1**, **inference**, and
**BF16 backbone autocast** are frugal. To fill the GPU more: increase ``--batch-size``, use
``--synthetic --synthetic-h/--synthetic-w`` for a larger grid, or ``--no-autocast-backbone``.
The Triton window-layout path (``--use-triton-layout``) typically lowers Swin activation footprint
relative to the default layout, so you can often push ``--batch-size`` higher than with the eager
layout before OOM—worth trying when stress-testing VRAM or throughput.

Optional charts (``matplotlib``)::

    uv run python aurora/profiling.py --plot
    uv run python aurora/profiling.py --plot-out profiling/my_ops.png

Write a Markdown report (config, timing, profiler table, paths)::

    uv run python aurora/profiling.py --report
    uv run python aurora/profiling.py --report-out profiling/run.md

If imports fail, install the local package::

    uv pip install -e ./aurora
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import glob
import os
import pickle
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

# Package lives in ``<repo>/aurora/aurora/``; allow ``python aurora/profiling.py``.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_batch_from_hf() -> Any:
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download

    from aurora import Batch, Metadata
    from aurora.batch import interpolate_numpy

    path = hf_hub_download(
        repo_id="microsoft/aurora",
        filename="aurora-0.25-small-pretrained-test-input.pickle",
    )
    # Pickled NumPy arrays may trigger numpy.core deprecation noise on NumPy 2.x.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with open(path, "rb") as f:
            test_input = pickle.load(f)

    path = hf_hub_download(
        repo_id="microsoft/aurora",
        filename="aurora-0.25-static.pickle",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with open(path, "rb") as f:
            static_vars = pickle.load(f)

    if os.name == "nt":

        class PatchedDateTime(datetime):
            def timestamp(self) -> float:
                return -631134000.0

        test_input["metadata"]["time"] = [PatchedDateTime(1950, 1, 1, 6, 0)]

    static_vars = {
        k: interpolate_numpy(
            v,
            np.linspace(90, -90, v.shape[0]),
            np.linspace(0, 360, v.shape[1], endpoint=False),
            test_input["metadata"]["lat"],
            test_input["metadata"]["lon"],
        )
        for k, v in static_vars.items()
    }

    return Batch(
        surf_vars={k: torch.from_numpy(v) for k, v in test_input["surf_vars"].items()},
        static_vars={k: torch.from_numpy(v) for k, v in static_vars.items()},
        atmos_vars={k: torch.from_numpy(v) for k, v in test_input["atmos_vars"].items()},
        metadata=Metadata(
            lat=torch.from_numpy(test_input["metadata"]["lat"]),
            lon=torch.from_numpy(test_input["metadata"]["lon"]),
            atmos_levels=tuple(test_input["metadata"]["atmos_levels"]),
            time=tuple(test_input["metadata"]["time"]),
        ),
    )


def _load_batch_synthetic(
    *,
    batch_size: int,
    h: int,
    w: int,
    history: int,
    levels: tuple[int | float, ...],
) -> Any:
    import torch

    from aurora import Batch, Metadata

    return Batch(
        surf_vars={k: torch.randn(batch_size, history, h, w) for k in ("2t", "10u", "10v", "msl")},
        static_vars={k: torch.randn(h, w) for k in ("lsm", "z", "slt")},
        atmos_vars={k: torch.randn(batch_size, history, len(levels), h, w) for k in ("z", "u", "v", "t", "q")},
        metadata=Metadata(
            lat=torch.linspace(90, -90, h),
            lon=torch.linspace(0, 360, w + 1)[:-1],
            time=(datetime(2020, 6, 1, 12, 0),),
            atmos_levels=levels,
        ),
    )


def _top_ops_ms(
    prof: Any,
    *,
    use_cuda: bool,
    top_k: int = 20,
) -> tuple[list[str], list[float]]:
    """Extract top-K op names by self time from profiler key averages (times in ms)."""
    rows: list[tuple[str, float]] = []
    for e in prof.key_averages():
        if use_cuda:
            t_us = float(
                getattr(e, "self_cuda_time_total", 0)
                or getattr(e, "self_device_time_total", 0)
                or 0
            )
        else:
            t_us = float(getattr(e, "self_cpu_time_total", 0) or 0)
        if t_us <= 0:
            continue
        rows.append((str(e.key), t_us / 1000.0))
    rows.sort(key=lambda x: -x[1])
    top = rows[:top_k]
    names = [n for n, _ in top]
    times = [t for _, t in top]
    return names, times


def _shorten_label(s: str, max_len: int = 72) -> str:
    s = s.replace("\n", " ")
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _save_profiler_plot(
    *,
    names: list[str],
    times_ms: list[float],
    out_path: Path,
    title_suffix: str,
    timing_line: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not names:
        raise ValueError("no profiler rows to plot")

    n = len(names)
    fig_h = max(4.0, 0.38 * n + 1.2)
    fig, ax = plt.subplots(figsize=(10.5, fig_h), layout="constrained")
    y = range(n)
    ax.barh(y, times_ms, color="#2c5f8d", alpha=0.9)
    ax.set_yticks(list(y))
    ax.set_yticklabels([_shorten_label(x) for x in names], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Self time (ms)")
    ax.set_title(f"Aurora profiler — top ops ({title_suffix})")
    fig.text(0.02, 0.02, timing_line, fontsize=8, family="monospace", color="#333333")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _md_escape_pipe(s: str) -> str:
    return s.replace("|", "\\|")


def _write_markdown_report(
    path: Path,
    *,
    generated: datetime,
    torch_version: str,
    cuda_version: str | None,
    gpu_name: str | None,
    config_summary: str,
    warmup_alloc_mb: float | None,
    timing_line: str,
    total_forwards: int,
    repeat: int,
    hint_text: str,
    prof_table_text: str,
    top_op_names: list[str],
    top_op_ms: list[float],
    plot_path: Path | None,
    trace_path: str | None,
    peak_cuda_mb: float | None,
    peak_reserved_mb: float | None,
) -> None:
    lines: list[str] = [
        "# Aurora end-to-end profiling report",
        "",
        f"- Generated: {generated.isoformat(timespec='seconds')}",
        f"- PyTorch: {torch_version}",
    ]
    if cuda_version:
        lines.append(f"- CUDA (PyTorch): {cuda_version}")
    if gpu_name:
        lines.append(f"- GPU: {gpu_name}")
    lines.extend(
        [
            "",
            "## Run configuration",
            "",
            "```text",
            config_summary.rstrip(),
            "```",
            "",
            "## GPU / CPU timer",
            "",
            timing_line,
            "",
            f"Profiler window: `{repeat}× run_once` ≈ **{total_forwards}** model forwards.",
            "",
        ]
    )
    if warmup_alloc_mb is not None:
        lines.extend([f"- CUDA allocated after warmup: **{warmup_alloc_mb:.1f} MB**", ""])
    if peak_cuda_mb is not None:
        pr = (
            f", peak reserved **{peak_reserved_mb:.1f} MB**"
            if peak_reserved_mb is not None
            else ""
        )
        lines.extend(
            [
                f"- Peak CUDA allocated (this process): **{peak_cuda_mb:.1f} MB**{pr}",
                "",
            ]
        )

    lines.extend(
        [
            "## Top operators (Self time)",
            "",
            "| Rank | Operator | Self time (ms) |",
            "| ---: | --- | ---: |",
        ]
    )
    for i, (name, t) in enumerate(zip(top_op_names, top_op_ms, strict=True), start=1):
        lines.append(f"| {i} | {_md_escape_pipe(name)} | {t:.3f} |")
    lines.append("")

    lines.extend(
        [
            "## PyTorch profiler table (full text)",
            "",
            "Same columns as the terminal table (may be wide).",
            "",
            "```text",
            prof_table_text.rstrip(),
            "```",
            "",
            "## Notes",
            "",
            hint_text,
            "",
        ]
    )
    if plot_path is not None:
        rel = plot_path.resolve()
        lines.extend(["## Artifacts", "", f"- Plot: `{rel}`", ""])
    if trace_path:
        lines.extend([f"- Chrome trace: `{Path(trace_path).resolve()}`", ""])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cudart_profiler_api() -> tuple[Callable[[], None], Callable[[], None]]:
    """Load ``cudaProfilerStart`` / ``cudaProfilerStop`` for use with ``nsys --capture-range=cudaProfilerApi``."""
    import torch

    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    candidates = sorted(glob.glob(str(torch_lib / "libcudart.so*")))
    path = candidates[0] if candidates else "libcudart.so"
    lib = ctypes.CDLL(path)
    _start = lib.cudaProfilerStart
    _stop = lib.cudaProfilerStop
    _start.argtypes = []
    _start.restype = ctypes.c_int
    _stop.argtypes = []
    _stop.restype = ctypes.c_int

    def start() -> None:
        err = int(_start())
        if err != 0:
            print(f"[warn] cudaProfilerStart returned {err}")

    def stop() -> None:
        err = int(_stop())
        if err != 0:
            print(f"[warn] cudaProfilerStop returned {err}")

    return start, stop


def _cuda_aggressive_cleanup(device: str) -> None:
    """Best-effort release of CUDA allocator caches between isolated benchmark phases."""
    import gc

    import torch

    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()
    gc.collect()


def _cuda_oom_like(exc: BaseException) -> bool:
    """PyTorch may raise RuntimeError, OutOfMemoryError, or AcceleratorError for CUDA OOM."""
    s = str(exc).lower().replace(" ", "")
    if "outofmemory" in s or "out of memory" in str(exc).lower():
        return True
    if "cudaerrormemoryallocation" in s or "memoryallocation" in s:
        return True
    name = type(exc).__name__
    if name == "OutOfMemoryError":
        return True
    if name == "AcceleratorError" and ("memory" in str(exc).lower() or "oom" in str(exc).lower()):
        return True
    return False


def _recover_cuda_after_oom(device_is_cuda: bool) -> None:
    """Best-effort: each call is wrapped; CUDA may be in a bad state after OOM."""
    import gc

    import torch

    gc.collect()
    if not device_is_cuda:
        return
    for fn in (
        getattr(torch.cuda, "synchronize", None),
        getattr(torch.cuda, "empty_cache", None),
    ):
        if fn is None:
            continue
        try:
            fn()
        except Exception:
            pass
    gc.collect()


def _probe_max_batch(
    args: argparse.Namespace,
    model: Any,
    batch_b1: Any,
    *,
    cap: int,
) -> int:
    """Largest ``n`` in ``[1, cap]`` for which one workload run succeeds (OOM-safe binary search)."""
    import gc

    import torch

    from aurora import rollout

    def attempt(n: int) -> bool:
        try:
            b = _repeat_batch_along_batch_dim(batch_b1, n)
            if args.forward_only:
                with torch.inference_mode():
                    _ = model.forward(b)
            else:
                with torch.inference_mode():
                    for _ in rollout(model, b, args.rollout_steps):
                        pass
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            return True
        except Exception as e:
            if not _cuda_oom_like(e):
                raise
            _recover_cuda_after_oom(args.device.startswith("cuda"))
            return False

    if cap < 1:
        return 0

    if not attempt(1):
        print("[warn] max_batch probe: batch size 1 failed (OOM or error).")
        return 0

    lo, hi = 1, cap
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if attempt(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def _aurora_model_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    from aurora.model.inference_precision import apply_inference_config, resolve_inference_config
    from aurora.model.workspace_pool import InferenceWorkspacePool

    pool_kw = {"workspace_pool": InferenceWorkspacePool()} if args.use_workspace_pool else {}
    base = {
        "use_lora": True,
        "lora_mode": "single",
        "use_lora_merged_inference": args.use_lora_merged_inference,
        **pool_kw,
    }
    if args.inference_precision:
        cfg = resolve_inference_config(args.inference_precision)
        if args.no_autocast_backbone and cfg is not None and cfg.autocast_backbone:
            raise SystemExit(
                f"--no-autocast-backbone conflicts with inference_precision={args.inference_precision!r}."
            )
        if args.use_triton_layout or args.use_triton_adaln or args.use_triton_mlp:
            print(
                "[warn] --inference-precision overrides scattered --use-triton-* flags."
            )
        preset = apply_inference_config(args.inference_precision)
        return {
            **base,
            **preset,
            "inference_precision": args.inference_precision,
        }
    return {
        **base,
        "autocast": not args.no_autocast_backbone,
        "use_triton_layout": args.use_triton_layout,
        "use_triton_adaln": args.use_triton_adaln,
        "use_triton_mlp": args.use_triton_mlp,
    }


def _timed_e2e_config(
    args: argparse.Namespace,
    batch_b1: Any,
    *,
    timing_batch_size: int,
    label: str,
    use_triton_layout: bool,
    use_triton_adaln: bool,
    use_triton_mlp: bool,
    use_lora_merged_inference: bool,
    use_workspace_pool: bool,
) -> tuple[str, float | None, float | None, int | None]:
    """Build AuroraSmall once, load checkpoint, GPU timer + peak mem (no torch.profiler)."""
    import gc

    import torch

    from aurora import AuroraSmallPretrained, rollout
    from aurora.model.workspace_pool import InferenceWorkspacePool

    dev = torch.device(args.device)
    if dev.type == "cuda":
        print(f"[mem] CUDA cleanup before «{label}» (gc, sync, empty_cache, ipc_collect)")
        _cuda_aggressive_cleanup(args.device)

    if (use_triton_layout or use_triton_adaln or use_triton_mlp) and not args.no_autocast_backbone:
        print(
            f"[warn] [{label}] BF16 autocast is ON; Triton Swin paths need float32 inside the "
            "backbone — use --no-autocast-backbone to enable them."
        )

    pool = InferenceWorkspacePool() if use_workspace_pool else None
    model = AuroraSmallPretrained(
        use_lora=True,
        lora_mode="single",
        autocast=not args.no_autocast_backbone,
        use_triton_layout=use_triton_layout,
        use_triton_adaln=use_triton_adaln,
        use_triton_mlp=use_triton_mlp,
        use_lora_merged_inference=use_lora_merged_inference,
        workspace_pool=pool,
    )
    model.load_checkpoint_prefer_local(
        checkpoint_dir=args.checkpoint_dir,
        path=args.checkpoint or None,
        strict=False,
        allow_hub_download=args.hub_download,
    )
    model.eval()
    model.to(args.device)

    max_batch: int | None = None
    effective_bs = timing_batch_size
    if (
        getattr(args, "compare_find_max_batch", False)
        and getattr(args, "compare_e2e_swin", False)
        and dev.type == "cuda"
    ):
        cap = int(getattr(args, "compare_batch_cap", 128))
        max_batch = _probe_max_batch(args, model, batch_b1, cap=cap)
        print(f"[batch] probed max_batch={max_batch} (cap={cap})")
        if max_batch < 1:
            del model
            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            return "[skip] max_batch probe failed at B>=1", None, None, 0
        if not getattr(args, "compare_time_at_cli_batch", False):
            effective_bs = max_batch
        else:
            effective_bs = timing_batch_size

    batch = _repeat_batch_along_batch_dim(batch_b1, effective_bs).to("cpu")
    if effective_bs != timing_batch_size and getattr(args, "compare_find_max_batch", False):
        print(
            f"[batch] warmup/timing use batch_size={effective_bs} "
            f"(CLI --batch-size={timing_batch_size})"
        )

    def run_once() -> None:
        if args.forward_only:
            with torch.inference_mode():
                _ = model.forward(batch)
        else:
            with torch.inference_mode():
                for _ in rollout(model, batch, args.rollout_steps):
                    pass


    for _ in range(args.warmup):
        run_once()
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

    timing_line = ""
    peak_alloc: float | None = None
    peak_reserved: float | None = None
    n_forwards = 1 if args.forward_only else args.rollout_steps
    total_forwards = args.repeat * n_forwards

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        for _ in range(args.repeat):
            run_once()
        ev1.record()
        torch.cuda.synchronize()
        ms_gpu = ev0.elapsed_time(ev1)
        timing_line = (
            f"GPU: {ms_gpu:.2f} ms for {args.repeat} run_once "
            f"({total_forwards} forwards) → {ms_gpu / total_forwards:.2f} ms/forward"
        )
        peak_alloc = torch.cuda.max_memory_allocated() / 1e6
        peak_reserved = torch.cuda.max_memory_reserved() / 1e6
        print(f"[timing] {timing_line}")
        print(
            f"[mem] peak CUDA allocated: {peak_alloc:.1f} MB, "
            f"peak reserved: {peak_reserved:.1f} MB"
        )
    else:
        import time as time_module

        t0 = time_module.perf_counter()
        for _ in range(args.repeat):
            run_once()
        ms_wall = (time_module.perf_counter() - t0) * 1e3
        timing_line = f"CPU: {ms_wall:.2f} ms for {args.repeat} run_once ({total_forwards} forwards)"

    del model
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
        gc.collect()

    return timing_line, peak_alloc, peak_reserved, max_batch


def _repeat_batch_along_batch_dim(batch: Any, n: int) -> Any:
    """Repeat batch along batch dimension (same as test_model)."""
    import torch

    from aurora import Batch

    if n == 1:
        return batch
    return dataclasses.replace(
        batch,
        surf_vars={k: v.repeat(n, 1, 1, 1) for k, v in batch.surf_vars.items()},
        atmos_vars={k: v.repeat(n, 1, 1, 1, 1) for k, v in batch.atmos_vars.items()},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end Aurora profiling (small GPU friendly).")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use a tiny random batch (minimal VRAM); less representative than HF test input.",
    )
    parser.add_argument(
        "--stress",
        action="store_true",
        help="Shorthand for larger batch: effective batch size 4 if --batch-size is omitted.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Batch dimension. Default 1, or 4 when --stress is set without this flag. "
        "Try 4–8+ with --synthetic for stress tests; HF inputs repeated may OOM sooner. "
        "With --use-triton-layout, Swin-side memory is often lower—retry a larger N if baseline OOMs.",
    )
    parser.add_argument(
        "--synthetic-h",
        type=int,
        default=17,
        metavar="H",
        help="Synthetic batch latitude height (only with --synthetic). Must work with patch size 4 "
        "(e.g. multiple of 4, or 1 mod 4). Raise to increase activation VRAM.",
    )
    parser.add_argument(
        "--synthetic-w",
        type=int,
        default=32,
        metavar="W",
        help="Synthetic batch longitude width (only with --synthetic). Must be a multiple of 4.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=2,
        help="Rollout length for profiling (each step runs the full model).",
    )
    parser.add_argument(
        "--forward-only",
        action="store_true",
        help="Profile a single forward instead of rollout (lower activation memory).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warmup iterations before profiling.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=4,
        help="Iterations per profiler run and GPU timer. Use 1 for easier totals to read by hand.",
    )
    parser.add_argument(
        "--trace-out",
        type=str,
        default="",
        help="If set, export Chrome trace JSON to this path.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save a matplotlib bar chart under profiling/aurora_profile_<timestamp>.png.",
    )
    parser.add_argument(
        "--plot-out",
        type=str,
        default="",
        metavar="PATH",
        help="Save matplotlib bar chart of top ops to this PNG (overrides --plot default path).",
    )
    parser.add_argument(
        "--plot-top",
        type=int,
        default=20,
        help="Number of ops to show in the bar chart (default 20).",
    )
    parser.add_argument(
        "--no-autocast-backbone",
        action="store_true",
        help="Disable BF16 autocast in the backbone (higher VRAM; not recommended on 12GB).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for model and tensors (default cuda).",
    )
    parser.add_argument(
        "--no-torch-profiler",
        action="store_true",
        help="Skip torch.profiler (use when wrapping with nsys profile / ncu so traces are not doubled).",
    )
    parser.add_argument(
        "--cuda-profiler-api",
        action="store_true",
        help="Wrap the timed run_once loop with cudaProfilerStart/Stop. Use with "
        "`nsys profile --capture-range=cudaProfilerApi` so the report focuses on inference, "
        "not import/dlopen. Implies --no-torch-profiler. CUDA only.",
    )
    parser.add_argument(
        "--cudnn-benchmark",
        action="store_true",
        help="Set torch.backends.cudnn.benchmark=True (fixed shapes; warmer cuDNN heuristics).",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Write results to profiling/aurora_e2e_<timestamp>.md.",
    )
    parser.add_argument(
        "--report-out",
        type=str,
        default="",
        metavar="PATH",
        help="Write Markdown report to this path (implies --report content; overrides --report path).",
    )
    parser.add_argument(
        "--use-triton-layout",
        action="store_true",
        help=(
            "Swin backbone: Triton window layout (needs FP32 inside backbone; often "
            "--no-autocast-backbone). Usually reduces activation VRAM vs default layout—"
            "stress tests may tolerate a larger --batch-size."
        ),
    )
    parser.add_argument(
        "--use-triton-adaln",
        action="store_true",
        help="Swin backbone: Triton fused AdaptiveLayerNorm.",
    )
    parser.add_argument(
        "--use-triton-mlp",
        action="store_true",
        help="Swin backbone: Triton GELU in MLP (eval, FP32 path).",
    )
    parser.add_argument(
        "--use-lora-merged-inference",
        action="store_true",
        help="Swin backbone: merge LoRA into linears for inference (Stage-C).",
    )
    parser.add_argument(
        "--use-workspace-pool",
        action="store_true",
        help="Swin backbone: InferenceWorkspacePool for decoder concat scratch (D3).",
    )
    parser.add_argument(
        "--inference-precision",
        choices=("fp32", "fast_fp32", "bf16_mixed"),
        default=None,
        help=(
            "Named Swin3D inference preset. Overrides scattered Triton/CuTe/autocast flags. "
            "Perceiver encoder/decoder stay PyTorch naive."
        ),
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help=(
            "Capture a fixed-shape CUDA graph after warmup. Requires fast_fp32 or bf16_mixed "
            "(or pass --inference-precision with a graph-capable preset)."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="/root/autodl-tmp/aurora",
        metavar="DIR",
        help="Directory for Aurora .ckpt files (default: /root/autodl-tmp/aurora).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        metavar="PATH",
        help="Explicit checkpoint file path (overrides --checkpoint-dir).",
    )
    parser.add_argument(
        "--hub-download",
        action="store_true",
        help="If the checkpoint is missing locally, download from Hugging Face Hub.",
    )
    parser.add_argument(
        "--compare-e2e-swin",
        action="store_true",
        help=(
            "Run four timed end-to-end configs: baseline; Stage-A/B (layout+AdaLN+MLP); "
            "+ Stage-C (LoRA merge); + D3 (workspace pool). Timing only; use --report to save Markdown. "
            "Triton layout lowers Swin memory—try a larger --batch-size than for baseline-only runs."
        ),
    )
    parser.add_argument(
        "--compare-find-max-batch",
        action="store_true",
        help=(
            "With --compare-e2e-swin on CUDA: binary-search the largest batch size per phase "
            "(upper bound --compare-batch-cap) using real forwards/OOM; then warm up and time "
            "at that batch unless --compare-time-at-cli-batch."
        ),
    )
    parser.add_argument(
        "--compare-batch-cap",
        type=int,
        default=128,
        metavar="N",
        help="Upper bound for --compare-find-max-batch (default 128).",
    )
    parser.add_argument(
        "--compare-time-at-cli-batch",
        action="store_true",
        help=(
            "With --compare-find-max-batch: still report probed max_batch but run warmup/timing "
            "at --batch-size instead of the maximum."
        ),
    )
    args = parser.parse_args()

    if args.cuda_profiler_api:
        args.no_torch_profiler = True

    if args.cuda_profiler_api and not str(args.device).startswith("cuda"):
        raise SystemExit("--cuda-profiler-api requires CUDA (--device cuda).")

    if args.compare_e2e_swin and args.cuda_profiler_api:
        raise SystemExit("--compare-e2e-swin cannot be combined with --cuda-profiler-api.")

    if args.compare_e2e_swin and (args.trace_out or args.plot or args.plot_out):
        raise SystemExit("--compare-e2e-swin does not support --trace-out or --plot; run a single config instead.")

    if args.compare_find_max_batch:
        if not args.compare_e2e_swin:
            raise SystemExit("--compare-find-max-batch requires --compare-e2e-swin.")
        if not str(args.device).startswith("cuda"):
            raise SystemExit("--compare-find-max-batch requires CUDA (--device cuda).")
        if args.compare_batch_cap < 1:
            raise SystemExit("--compare-batch-cap must be >= 1.")

    if args.cuda_graph:
        if not str(args.device).startswith("cuda"):
            raise SystemExit("--cuda-graph requires CUDA (--device cuda).")
        if args.inference_precision in (None, "fp32"):
            raise SystemExit(
                "--cuda-graph requires --inference-precision fast_fp32 or bf16_mixed."
            )
    if args.inference_precision and (
        args.use_triton_layout or args.use_triton_adaln or args.use_triton_mlp
    ):
        print("[warn] --inference-precision overrides scattered --use-triton-* flags.")

    if args.no_torch_profiler and (
        args.trace_out
        or args.plot
        or args.plot_out
        or ((args.report or args.report_out) and not args.compare_e2e_swin)
    ):
        raise SystemExit(
            "--no-torch-profiler cannot be combined with --trace-out, --plot, or --report "
            "(torch.profiler is required for those outputs). "
            "Exception: --compare-e2e-swin --report writes a timing-only Markdown summary."
        )

    batch_size = args.batch_size if args.batch_size is not None else (4 if args.stress else 1)
    if batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    if args.synthetic:
        # Aurora default patch_size=4; Batch.crop requires W % 4 == 0 and H % 4 in {0, 1}.
        pw, ph = args.synthetic_w, args.synthetic_h
        if pw % 4 != 0:
            raise SystemExit("--synthetic-w must be a multiple of 4 (patch size 4).")
        if ph % 4 not in (0, 1):
            raise SystemExit("--synthetic-h must satisfy H%%4==0 or H%%4==1 (patch size 4).")

    import torch
    from torch.profiler import ProfilerActivity, profile

    from aurora import AuroraSmallPretrained, rollout

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Use --device cpu (slow) or run on a GPU machine.")

    if args.cudnn_benchmark and args.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    stress_note = " (--stress default)" if args.stress and args.batch_size is None else ""
    print(
        f"[config] batch_size={batch_size}{stress_note}, synthetic={args.synthetic}, "
        f"repeat={args.repeat}, rollout_steps={args.rollout_steps}, forward_only={args.forward_only}, "
        f"cuda_profiler_api={args.cuda_profiler_api}, cudnn_benchmark={args.cudnn_benchmark}"
        + (
            f", synthetic_h={args.synthetic_h}, synthetic_w={args.synthetic_w}"
            if args.synthetic
            else ""
        )
    )

    if args.synthetic:
        if args.compare_e2e_swin:
            batch_one = _load_batch_synthetic(
                batch_size=1,
                h=args.synthetic_h,
                w=args.synthetic_w,
                history=2,
                levels=(100, 250, 500, 850),
            )
            batch = None
        else:
            batch_one = None  # unused
            batch = _load_batch_synthetic(
                batch_size=batch_size,
                h=args.synthetic_h,
                w=args.synthetic_w,
                history=2,
                levels=(100, 250, 500, 850),
            )
    else:
        batch_hf = _load_batch_from_hf()
        batch_one = _repeat_batch_along_batch_dim(batch_hf, 1)
        batch = None if args.compare_e2e_swin else _repeat_batch_along_batch_dim(batch_hf, batch_size)

    if args.compare_e2e_swin:
        configs = [
            ("Baseline", False, False, False, False, False),
            ("Stage-A/B (Triton layout+AdaLN+MLP)", True, True, True, False, False),
            ("+ Stage-C (LoRA merge)", True, True, True, True, False),
            ("+ D3 (workspace pool)", True, True, True, True, True),
        ]
        print(
            "\n=== Aurora end-to-end: baseline → A/B → +C → +D3 ===\n"
            f"[compare] Torch {torch.__version__}, repeat={args.repeat}, warmup={args.warmup}, "
            f"forward_only={args.forward_only}, rollout_steps={args.rollout_steps}, "
            f"no_autocast_backbone={args.no_autocast_backbone}\n"
            "[compare] Hint: A/B onward use Triton layout (lower Swin VRAM)—if baseline was VRAM-bound, "
            "try sweeping a larger --batch-size than without layout."
        )
        if args.compare_find_max_batch:
            print(
                f"[compare] Max-batch probe: cap={args.compare_batch_cap}, "
                f"time_at_cli_batch={args.compare_time_at_cli_batch}"
            )

        hdr = "| Config | Timing | peak alloc (MB) | peak reserved (MB) |"
        sep = "| --- | --- | ---: | ---: |"
        if args.compare_find_max_batch:
            hdr += " max_batch |"
            sep += " ---: |"

        md_lines = [
            "# Aurora end-to-end Swin compare",
            "",
            f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
            f"- Torch: {torch.__version__}",
            f"- repeat={args.repeat}, warmup={args.warmup}, CLI batch_size={batch_size}, "
            f"forward_only={args.forward_only}, rollout_steps={args.rollout_steps}, "
            f"no_autocast_backbone={args.no_autocast_backbone}",
        ]
        if args.compare_find_max_batch:
            md_lines.extend(
                [
                    f"- compare_find_max_batch: cap={args.compare_batch_cap}, "
                    f"time_at_cli_batch={args.compare_time_at_cli_batch}",
                ]
            )
        md_lines.extend(["", "## Results", "", hdr, sep])

        for label, tl, ta, tm, lm, wp in configs:
            print(f"\n--- {label} ---")
            line, pa, pr, mx = _timed_e2e_config(
                args,
                batch_one,
                timing_batch_size=batch_size,
                label=label,
                use_triton_layout=tl,
                use_triton_adaln=ta,
                use_triton_mlp=tm,
                use_lora_merged_inference=lm,
                use_workspace_pool=wp,
            )
            pa_s = f"{pa:.1f}" if pa is not None else "—"
            pr_s = f"{pr:.1f}" if pr is not None else "—"
            row = f"| {label} | {line} | {pa_s} | {pr_s} |"
            if args.compare_find_max_batch:
                mx_s = str(mx) if mx is not None else "—"
                row += f" {mx_s} |"
            md_lines.append(row)

        print("\n=== Summary (Markdown table) ===\n")
        print("\n".join(md_lines[md_lines.index("## Results") + 2 :]))

        report_path: Path | None = None
        if args.report_out:
            report_path = Path(args.report_out)
        elif args.report:
            report_path = Path("profiling") / f"aurora_e2e_swin_compare_{datetime.now():%Y%m%d_%H%M%S}.md"

        if report_path is not None:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
            print(f"\n[report] Markdown written: {report_path.resolve()}")

        return

    from aurora.model.workspace_pool import InferenceWorkspacePool

    pool = InferenceWorkspacePool() if args.use_workspace_pool else None
    model_kwargs = _aurora_model_kwargs(args)
    if pool is not None and "workspace_pool" not in model_kwargs:
        model_kwargs["workspace_pool"] = pool
    model = AuroraSmallPretrained(**model_kwargs)
    ckpt_path = model.load_checkpoint_prefer_local(
        checkpoint_dir=args.checkpoint_dir,
        path=args.checkpoint or None,
        strict=False,
        allow_hub_download=args.hub_download,
    )
    print(f"[checkpoint] {ckpt_path}")
    model.eval()
    model.to(args.device)

    def run_once() -> None:
        if args.forward_only:
            with torch.inference_mode():
                _ = model.forward(batch)
        else:
            with torch.inference_mode():
                for _ in rollout(model, batch, args.rollout_steps):
                    pass

    batch = batch.to("cpu")

    for _ in range(args.warmup):
        run_once()
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

    if args.cuda_graph:
        capture_batch = batch.to(args.device)
        model.capture_inference_cuda_graph(capture_batch)
        print(
            f"[cuda-graph] captured scope="
            f"{getattr(model, '_cuda_graph_scope', 'unknown')} for inference replay"
        )
        for _ in range(max(1, args.warmup // 2)):
            run_once()
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()

    warmup_alloc_mb: float | None = None
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        warmup_alloc_mb = torch.cuda.memory_allocated() / 1e6
        print(f"[mem] CUDA allocated after warmup (approx): {warmup_alloc_mb:.1f} MB")

    n_forwards_per_run = 1 if args.forward_only else args.rollout_steps
    total_forwards = args.repeat * n_forwards_per_run

    timing_line = ""
    if args.device.startswith("cuda"):
        prof_start: Callable[[], None] | None = None
        prof_stop: Callable[[], None] | None = None
        if args.cuda_profiler_api:
            prof_start, prof_stop = _cudart_profiler_api()
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        ev_start.record()
        if prof_start is not None:
            prof_start()
        for _ in range(args.repeat):
            run_once()
        ev_end.record()
        torch.cuda.synchronize()
        if prof_stop is not None:
            prof_stop()
        ms_gpu = ev_start.elapsed_time(ev_end)
        print(
            f"[timing] GPU {ms_gpu:.2f} ms for {args.repeat}× run_once "
            f"({total_forwards} forwards) → {ms_gpu / total_forwards:.2f} ms/forward, "
            f"{ms_gpu / args.repeat:.2f} ms/run_once"
        )
        timing_line = (
            f"GPU timer: {ms_gpu:.2f} ms / {total_forwards} forwards "
            f"= {ms_gpu / total_forwards:.2f} ms/forward"
        )
    else:
        t0 = time.perf_counter()
        for _ in range(args.repeat):
            run_once()
        wall_s = time.perf_counter() - t0
        ms_wall = wall_s * 1e3
        print(
            f"[timing] CPU {ms_wall:.2f} ms for {args.repeat}× run_once "
            f"({total_forwards} forwards) → {ms_wall / total_forwards:.2f} ms/forward"
        )
        timing_line = (
            f"CPU timer: {ms_wall:.2f} ms / {total_forwards} forwards "
            f"= {ms_wall / total_forwards:.2f} ms/forward"
        )

    if args.no_torch_profiler:
        peak_cuda_mb: float | None = None
        peak_reserved_mb: float | None = None
        if args.device.startswith("cuda"):
            peak_cuda_mb = torch.cuda.max_memory_allocated() / 1e6
            peak_reserved_mb = torch.cuda.max_memory_reserved() / 1e6
            print(
                f"\n[mem] Peak CUDA allocated (process): {peak_cuda_mb:.1f} MB, "
                f"peak reserved: {peak_reserved_mb:.1f} MB"
            )
        print(
            "\n[nsys] Skipped torch.profiler (--no-torch-profiler). "
            "Open the .nsys-rep from `nsys profile` in Nsight Systems (nsys-ui)."
        )
        if args.cuda_profiler_api:
            print(
                "[nsys] CUDA capture used cudaProfilerStart/Stop; ensure `nsys profile` "
                "was run with `--capture-range=cudaProfilerApi`."
            )
        return

    activities = [ProfilerActivity.CPU]
    if args.device.startswith("cuda"):
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        acc_events=True,
    ) as prof:
        for _ in range(args.repeat):
            run_once()
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

    sort_by = "self_cuda_time_total" if args.device.startswith("cuda") else "self_cpu_time_total"
    prof_table_text = prof.key_averages().table(sort_by=sort_by, row_limit=25)
    print("\n" + prof_table_text)
    hint_text = (
        "Profiler Self CPU/CUDA totals sum over **all** recorded ops in this window "
        f"({args.repeat}× run_once ≈ {total_forwards} forwards). "
        "Divide by that forward count for a rough per-forward share, or use `--repeat 1`."
    )
    print("\n[hint] " + hint_text)

    plot_path: Path | None = None
    if args.plot_out:
        plot_path = Path(args.plot_out)
    elif args.plot:
        plot_path = Path("profiling") / f"aurora_profile_{datetime.now():%Y%m%d_%H%M%S}.png"

    if plot_path is not None:
        try:
            use_cuda_plot = args.device.startswith("cuda")
            pnames, ptimes = _top_ops_ms(prof, use_cuda=use_cuda_plot, top_k=args.plot_top)
            suffix = "Self CUDA" if use_cuda_plot else "Self CPU"
            _save_profiler_plot(
                names=pnames,
                times_ms=ptimes,
                out_path=plot_path,
                title_suffix=suffix,
                timing_line=timing_line,
            )
            print(f"\n[plot] Saved operator bar chart: {plot_path.resolve()}")
        except ImportError:
            print("\n[plot] matplotlib not installed; skip chart. Install: uv pip install matplotlib")

    if args.trace_out:
        prof.export_chrome_trace(args.trace_out)
        print(f"\nChrome trace written to: {args.trace_out}")

    peak_cuda_mb: float | None = None
    peak_reserved_mb: float | None = None
    if args.device.startswith("cuda"):
        peak_cuda_mb = torch.cuda.max_memory_allocated() / 1e6
        peak_reserved_mb = torch.cuda.max_memory_reserved() / 1e6
        print(
            f"\n[mem] Peak CUDA allocated (process): {peak_cuda_mb:.1f} MB, "
            f"peak reserved: {peak_reserved_mb:.1f} MB"
        )

    report_path: Path | None = None
    if args.report_out:
        report_path = Path(args.report_out)
    elif args.report:
        report_path = Path("profiling") / f"aurora_e2e_{datetime.now():%Y%m%d_%H%M%S}.md"

    if report_path is not None:
        use_cuda = args.device.startswith("cuda")
        top_names, top_ms = _top_ops_ms(prof, use_cuda=use_cuda, top_k=25)
        cuda_ver = (
            torch.version.cuda
            if torch.cuda.is_available() and args.device.startswith("cuda")
            else None
        )
        gpu_name = (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available() and args.device.startswith("cuda")
            else None
        )
        config_summary = (
            f"batch_size={batch_size}{stress_note}\n"
            f"synthetic={args.synthetic}\n"
            + (
                f"synthetic_h={args.synthetic_h}\n"
                f"synthetic_w={args.synthetic_w}\n"
                if args.synthetic
                else ""
            )
            + f"repeat={args.repeat}\n"
            f"rollout_steps={args.rollout_steps}\n"
            f"forward_only={args.forward_only}\n"
            f"autocast_backbone={not args.no_autocast_backbone}\n"
            f"device={args.device}"
        )
        _write_markdown_report(
            report_path,
            generated=datetime.now(),
            torch_version=torch.__version__,
            cuda_version=cuda_ver,
            gpu_name=gpu_name,
            config_summary=config_summary,
            warmup_alloc_mb=warmup_alloc_mb,
            timing_line=timing_line,
            total_forwards=total_forwards,
            repeat=args.repeat,
            hint_text=hint_text,
            prof_table_text=prof_table_text,
            top_op_names=top_names,
            top_op_ms=top_ms,
            plot_path=plot_path,
            trace_path=args.trace_out or None,
            peak_cuda_mb=peak_cuda_mb,
            peak_reserved_mb=peak_reserved_mb,
        )
        print(f"\n[report] Markdown written: {report_path.resolve()}")


if __name__ == "__main__":
    main()
