#!/usr/bin/env python3
"""Copyright (c) Catman Jr. Licensed under the MIT license.

End-to-end maximum batch probe for **AuroraSmallPretrained**: compare ``use_triton_perceiver_ln_fusion``
off vs on using the **same weights**, and report the largest batch size in ``[1, cap]`` for which one
forward (or rollout) completes without CUDA OOM.

Synthetic grid constraints (patch size 4): ``--synthetic-w % 4 == 0``; ``--synthetic-h % 4`` in
``{0, 1}`` (matches :func:`profiling._load_batch_synthetic` assumptions).

Examples::

    PYTHONPATH=aurora uv run python benchmark/bench_aurora_e2e_max_batch.py
    PYTHONPATH=aurora uv run python benchmark/bench_aurora_e2e_max_batch.py --cap 512 --synthetic-h 32 --synthetic-w 64

By default the small-pretrained ``.ckpt`` is stored under ``/root/autodl-tmp`` (AutoDL data disk). Use
``--checkpoint-dir`` to change that directory, or ``--checkpoint /path/to/aurora-0.25-small-pretrained.ckpt``
to load a file you already have.

Use a Hugging Face **mirror** when ``huggingface.co`` is unreachable (parsed before Hub import)::

    uv run benchmark/bench_aurora_e2e_max_batch.py --hf-mirror
    uv run benchmark/bench_aurora_e2e_max_batch.py --hf-endpoint https://hf-mirror.com

Equivalent: ``export HF_ENDPOINT=https://hf-mirror.com`` in the shell *before* starting Python.

Requires CUDA.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import os
import sys
from datetime import datetime
from pathlib import Path

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
import _bootstrap  # noqa: F401, E402

import torch

_REPO = Path(__file__).resolve().parents[1]
_AURORA_PKG = _REPO / "aurora"
if _AURORA_PKG.is_dir():
    sys.path.insert(0, str(_AURORA_PKG))

_DEFAULT_HF_ENDPOINT = "https://huggingface.co"
# AutoDL: large fast volume, standard mount
_DEFAULT_CHECKPOINT_DIR = "/root/autodl-tmp/aurora"


def _apply_hf_hub_endpoint_from_argv(argv: list[str]) -> None:
    """Set ``HF_ENDPOINT`` before ``huggingface_hub`` is imported (mirrors must be applied first)."""
    chosen: str | None = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--hf-mirror":
            chosen = "https://hf-mirror.com"
            i += 1
            continue
        if a.startswith("--hf-endpoint=") and len(a) > len("--hf-endpoint="):
            chosen = a.split("=", 1)[1].strip().rstrip("/")
            i += 1
            continue
        if a == "--hf-endpoint":
            if i + 1 < len(argv):
                nxt = argv[i + 1]
                if not nxt.startswith("-") or nxt.startswith("https://") or nxt.startswith("http://"):
                    chosen = nxt.strip().rstrip("/")
                    i += 2
                    continue
            i += 1
            continue
        i += 1
    if chosen:
        os.environ["HF_ENDPOINT"] = chosen


def _cuda_oom_like(exc: BaseException) -> bool:
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


def _repeat_batch_along_batch_dim(batch: object, n: int) -> object:
    from aurora import Batch

    if n == 1:
        return batch
    assert isinstance(batch, Batch)
    return dataclasses.replace(
        batch,
        surf_vars={k: v.repeat(n, 1, 1, 1) for k, v in batch.surf_vars.items()},
        atmos_vars={k: v.repeat(n, 1, 1, 1, 1) for k, v in batch.atmos_vars.items()},
    )


def _load_batch_synthetic(
    *,
    batch_size: int,
    h: int,
    w: int,
    history: int,
    levels: tuple[int | float, ...],
) -> object:
    from aurora import Batch, Metadata

    return Batch(
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


def _clean_cuda(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    gc.collect()


def probe_max_batch(
    *,
    model: torch.nn.Module,
    batch_b1: object,
    cap: int,
    device: torch.device,
    forward_only: bool,
    rollout_steps: int,
) -> int:
    """Largest n in [1, cap] for which the workload succeeds (OOM-safe binary search)."""
    from aurora import rollout

    use_cuda = device.type == "cuda"

    def attempt(n: int) -> bool:
        try:
            b = _repeat_batch_along_batch_dim(batch_b1, n)
            with torch.inference_mode():
                if forward_only:
                    _ = model.forward(b)
                else:
                    for _ in rollout(model, b, rollout_steps):
                        pass
            if use_cuda:
                torch.cuda.synchronize(device)
            return True
        except Exception as e:
            if not _cuda_oom_like(e):
                raise
            _recover_cuda_after_oom(use_cuda)
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


def peak_mb_one_forward(model: torch.nn.Module, batch: object, device: torch.device) -> float:
    if device.type != "cuda":
        return float("nan")
    _clean_cuda(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        _ = model.forward(batch)
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / (1024**2)


def main() -> None:
    _apply_hf_hub_endpoint_from_argv(sys.argv[1:])
    from aurora import AuroraSmallPretrained

    p = argparse.ArgumentParser(
        description="E2E max batch: Perceiver LN fusion off vs on (AuroraSmallPretrained, same weights)"
    )
    p.add_argument("--cap", type=int, default=256, metavar="N", help="Upper bound for binary search.")
    p.add_argument("--synthetic-h", type=int, default=17)
    p.add_argument("--synthetic-w", type=int, default=32)
    p.add_argument("--history", type=int, default=2)
    p.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[100, 250, 500, 850],
        help="Atmospheric level axis (length must match checkpoint).",
    )
    p.add_argument(
        "--rollout",
        action="store_true",
        help="Measure rollout (multiple forwards) instead of a single forward (uses more VRAM).",
    )
    p.add_argument("--rollout-steps", type=int, default=2)
    p.add_argument(
        "--report-peak-mb",
        action="store_true",
        help="After probing, run once at max batch and print peak CUDA allocated (MB).",
    )
    p.add_argument("--use-triton-layout", action="store_true")
    p.add_argument("--use-triton-adaln", action="store_true")
    p.add_argument("--use-triton-mlp", action="store_true")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="",
        metavar="PATH",
        help=(
            "Explicit path to the small-pretrained .ckpt. If omitted, the file is taken from "
            "--checkpoint-dir (downloading from the Hub there if missing)."
        ),
    )
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default=_DEFAULT_CHECKPOINT_DIR,
        metavar="DIR",
        help=(
            f"Directory for ``{AuroraSmallPretrained.default_checkpoint_name}`` (default: {_DEFAULT_CHECKPOINT_DIR}). "
            "Hub downloads use this as local_dir so the .ckpt lives on the data disk (e.g. AutoDL autodl-tmp)."
        ),
    )
    p.add_argument(
        "--hf-mirror",
        action="store_true",
        help="Use https://hf-mirror.com as the Hub endpoint (sets HF_ENDPOINT; must precede Hub import).",
    )
    p.add_argument(
        "--hf-endpoint",
        type=str,
        default="",
        metavar="URL",
        help=(
            "Custom Hugging Face Hub API base URL (sets HF_ENDPOINT), e.g. https://hf-mirror.com. "
            "Alternatively export HF_ENDPOINT before starting Python."
        ),
    )
    args = p.parse_args()
    forward_only = not args.rollout

    if not torch.cuda.is_available():
        print("CUDA required.", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda")
    h, w = args.synthetic_h, args.synthetic_w
    levels = tuple(args.levels)

    batch_1 = _load_batch_synthetic(batch_size=1, h=h, w=w, history=args.history, levels=levels)

    common_kw: dict[str, object] = {
        "use_lora": True,
        "lora_mode": "single",
        "autocast": True,
        "use_triton_layout": args.use_triton_layout,
        "use_triton_adaln": args.use_triton_adaln,
        "use_triton_mlp": args.use_triton_mlp,
    }

    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"synthetic H×W={h}×{w} history={args.history} levels={levels}")
    print(f"cap={args.cap} forward_only={forward_only} rollout_steps={args.rollout_steps}")
    print(f"Swin triton: layout={args.use_triton_layout} adaln={args.use_triton_adaln} mlp={args.use_triton_mlp}")

    # Load checkpoint once; share weights between eager and fused runs.
    eager = AuroraSmallPretrained(use_triton_perceiver_ln_fusion=False, **common_kw)
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint).expanduser().resolve()
        if not ckpt_path.is_file():
            print(f"error: --checkpoint file not found: {ckpt_path}", file=sys.stderr)
            sys.exit(2)
        print(f"checkpoint={ckpt_path} (explicit --checkpoint)")
    else:
        from huggingface_hub import hf_hub_download

        dest_dir = Path(args.checkpoint_dir).expanduser().resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = Path(
            hf_hub_download(
                repo_id=AuroraSmallPretrained.default_checkpoint_repo,
                filename=AuroraSmallPretrained.default_checkpoint_name,
                revision=AuroraSmallPretrained.default_checkpoint_revision,
                local_dir=str(dest_dir),
            )
        )
        hub_base = os.environ.get("HF_ENDPOINT", _DEFAULT_HF_ENDPOINT).rstrip("/")
        print(
            f"checkpoint={ckpt_path.name} under {dest_dir} "
            f"(HF Hub repo={AuroraSmallPretrained.default_checkpoint_repo}, endpoint={hub_base})"
        )
    eager.load_checkpoint_local(str(ckpt_path), strict=False)
    eager = eager.to(device=device, dtype=torch.bfloat16)
    sd = eager.state_dict()

    _clean_cuda(device)
    mb_eager = probe_max_batch(
        model=eager,
        batch_b1=batch_1,
        cap=args.cap,
        device=device,
        forward_only=forward_only,
        rollout_steps=args.rollout_steps,
    )
    peak_eager: float | None = None
    if args.report_peak_mb and mb_eager >= 1:
        peak_eager = peak_mb_one_forward(eager, _repeat_batch_along_batch_dim(batch_1, mb_eager), device)
    print(f"max_batch [eager (fusion off)] = {mb_eager}")
    if peak_eager is not None:
        print(f"  peak_cuda_alloc_mb @ max_batch = {peak_eager:.2f}")
    del eager
    _clean_cuda(device)

    fused = AuroraSmallPretrained(use_triton_perceiver_ln_fusion=True, **common_kw)
    fused.load_state_dict(sd, strict=False)
    fused = fused.to(device=device, dtype=torch.bfloat16)

    _clean_cuda(device)
    mb_fused = probe_max_batch(
        model=fused,
        batch_b1=batch_1,
        cap=args.cap,
        device=device,
        forward_only=forward_only,
        rollout_steps=args.rollout_steps,
    )
    peak_fused: float | None = None
    if args.report_peak_mb and mb_fused >= 1:
        peak_fused = peak_mb_one_forward(fused, _repeat_batch_along_batch_dim(batch_1, mb_fused), device)
    print(f"max_batch [fused (fusion on)] = {mb_fused}")
    if peak_fused is not None:
        print(f"  peak_cuda_alloc_mb @ max_batch = {peak_fused:.2f}")
    del fused
    _clean_cuda(device)

    eager_b, fused_b = mb_eager, mb_fused
    delta = fused_b - eager_b
    sign = "+" if delta >= 0 else ""
    print("---")
    print(f"max_batch eager={eager_b} fused={fused_b} (fused − eager = {sign}{delta})")


if __name__ == "__main__":
    main()
