#!/usr/bin/env python3
"""Benchmark LoRA inference path on AuroraHighRes (0.1°) across custom precision tiers.

Finetuned checkpoints **require** ``use_lora=True`` (same as ``CheckpointLoader`` /
``hres_0.1`` preset). This script compares the two LoRA inference strategies only:

- ``lora_eager``: ``use_lora_merged_inference=False`` — ``Linear(x) + LoRA(x, step)``
- ``lora_merged``: ``use_lora_merged_inference=True`` — ``F.linear(x, W + ΔW)`` (**engine default**)

``no_lora`` is optional (``--include-no-lora``) for kernel isolation only; not a valid
production configuration for finetuned weights.

Examples::

    CUTE_DSL_ARCH=sm_120a uv run python benchmark/bench_aurora_hres_lora.py \\
        --asset-root /root/autodl-tmp/aurora

For synthetic HRES-only timing see ``bench_aurora_hres_oom_probe.py``. For all finetuned
presets on **real ingress** see ``bench_aurora_finetuned_lora.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
import _bootstrap  # noqa: F401, E402

from bench_aurora_hres_oom_probe import (  # noqa: E402
    _CHECKPOINT_NAME,
    make_synthetic_hres_batch,
)
from _asset_root import default_asset_root  # noqa: E402
from _pretrained_era5 import (  # noqa: E402
    custom_matmul_combo_tiers,
    purge_gpu,
    time_forward,
)

import torch

_DEFAULT_TIERS = (
    "bf16_mixed@fp32",
    "bf16_mixed@tf32",
    "bf16@fp32",
    "tf32@tf32",
)

_LORA_MODES: tuple[tuple[str, bool, bool], ...] = (
    ("lora_eager", True, False),
    ("lora_merged", True, True),
)


def build_hres(
    checkpoint: Path,
    *,
    precision: str,
    use_lora: bool,
    use_lora_merged_inference: bool,
    device: torch.device,
):
    from flash_aurora.aurora.model.aurora import AuroraHighRes

    model = AuroraHighRes(
        use_lora=use_lora,
        use_lora_merged_inference=use_lora_merged_inference,
        inference_precision=precision,
    )
    model.load_checkpoint_local(str(checkpoint), strict=False)
    model.eval()
    return model.to(device)


def run_case(
    *,
    tier_label: str,
    precision: str,
    lora_name: str,
    use_lora: bool,
    merged: bool,
    checkpoint: Path,
    batch,
    device: torch.device,
    warmup: int,
    repeat: int,
) -> tuple[float, float, float]:
    model = build_hres(
        checkpoint,
        precision=precision,
        use_lora=use_lora,
        use_lora_merged_inference=merged,
        device=device,
    )
    dev_batch = batch.to(device)
    try:
        _, ms, peak_alloc, peak_reserved = time_forward(
            model,
            dev_batch,
            warmup=warmup,
            repeat=repeat,
            device=device,
        )
        return ms, peak_alloc, peak_reserved
    finally:
        purge_gpu(model, dev_batch)


def print_table(rows: list[tuple[str, str, float, float, float]]) -> None:
    print(f"\n{'tier':<22} {'lora_mode':<14} {'ms/fwd':>10} {'peak_alloc':>12} {'peak_rsv':>12}")
    print("-" * 74)
    for tier, lora, ms, alloc, rsv in rows:
        print(f"{tier:<22} {lora:<14} {ms:10.1f} {alloc:12.0f} {rsv:12.0f}")


def print_speedup_summary(rows: list[tuple[str, str, float, float, float]]) -> None:
    by_tier: dict[str, dict[str, float]] = {}
    for tier, lora, ms, _a, _r in rows:
        by_tier.setdefault(tier, {})[lora] = ms

    print("\nMerged vs eager (ratio = eager_ms / merged_ms; >1 means merge wins):")
    print(f"  {'tier':<22} {'eager/merged':>14} {'eager ms':>10} {'merged ms':>10}")
    print("  " + "-" * 58)
    for tier in sorted(by_tier):
        eager = by_tier[tier].get("lora_eager", float("nan"))
        merged = by_tier[tier].get("lora_merged", float("nan"))
        if merged <= 0:
            continue
        ratio = eager / merged
        print(f"  {tier:<22} {ratio:13.2f}x {eager:10.1f} {merged:10.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=default_asset_root())
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=list(_DEFAULT_TIERS),
        help="Custom combo tiers (default: main production set)",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--include-no-lora",
        action="store_true",
        help="Also run use_lora=False (invalid for finetuned ckpt semantics; debug only)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    device = torch.device("cuda")
    asset_root = args.asset_root.expanduser().resolve()
    checkpoint = (args.checkpoint or asset_root / _CHECKPOINT_NAME).expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")

    tier_entries = custom_matmul_combo_tiers(
        backbone_levels=tuple({t.split("@")[0] for t in args.tiers}),
        encoder_decoder_levels=tuple({t.split("@")[1] for t in args.tiers}),
    )
    tier_map = {spec: (label, spec) for label, spec, _desc in tier_entries}
    selected: list[tuple[str, str]] = []
    for t in args.tiers:
        if t in tier_map:
            selected.append(tier_map[t])
        else:
            selected.append((t, t))

    print(f"[gpu] {torch.cuda.get_device_name(device)}")
    print(f"[ckpt] {checkpoint}")
    print(f"[warmup] {args.warmup}  [repeat] {args.repeat}")

    batch = make_synthetic_hres_batch(asset_root)
    h, w = batch.spatial_shape
    print(f"[ic] synthetic {h}x{w} batch=1")

    lora_modes = _LORA_MODES
    if args.include_no_lora:
        lora_modes = (("no_lora", False, False),) + lora_modes

    rows: list[tuple[str, str, float, float, float]] = []
    for label, precision in selected:
        for lora_name, use_lora, merged in lora_modes:
            tag = f"{label} ({precision})"
            print(f"[run] {tag}  {lora_name}...", flush=True)
            ms, alloc, rsv = run_case(
                tier_label=label,
                precision=precision,
                lora_name=lora_name,
                use_lora=use_lora,
                merged=merged,
                checkpoint=checkpoint,
                batch=batch,
                device=device,
                warmup=args.warmup,
                repeat=args.repeat,
            )
            rows.append((label, lora_name, ms, alloc, rsv))
            print(f"       -> {ms:.1f} ms/fwd  peak_alloc={alloc:.0f} MiB  peak_rsv={rsv:.0f} MiB")

    print_table(rows)
    print_speedup_summary(rows)


if __name__ == "__main__":
    main()
