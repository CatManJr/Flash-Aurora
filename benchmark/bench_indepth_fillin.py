#!/usr/bin/env python3
"""Fill-in measurements for the in-depth benchmark suite: full one-step mean relative error
and torch.compile after checkpoint load (the in-constructor flag remaps keys).
"""

from __future__ import annotations

import gc
import json
import sys
from datetime import datetime
from pathlib import Path

_BENCH_DIR = Path(__file__).resolve().parent
_REPO = _BENCH_DIR.parent
sys.path.insert(0, str(_BENCH_DIR))
sys.path.insert(0, str(_REPO))
import _bootstrap  # noqa: F401, E402

from _asset_root import default_asset_root  # noqa: E402
from _preset_ic import checkpoint_path, load_preset_batch  # noqa: E402
from _pretrained_era5 import purge_gpu  # noqa: E402
from bench_indepth_eval import (  # noqa: E402
    PHYS_VARS,
    build_model,
    compare_phys,
    one_step_accuracy,
    stats,
    time_samples,
)

import torch

ASSET = Path("/root/autodl-tmp/aurora")
OUT = _BENCH_DIR / "indepth_eval_fillin.json"


def compile_after_load(model) -> None:
    model.backbone = torch.compile(model.backbone, dynamic=False)


def main() -> None:
    device = torch.device("cuda")
    payload: dict = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
    }

    print("=== one-step mean relative error both ICs ===", flush=True)
    batch, config = load_preset_batch("era5_pretrained", ASSET)
    ckpt = checkpoint_path(config, ASSET)
    payload["one_step_2023"] = one_step_accuracy(
        config,
        ckpt,
        batch,
        device,
        ["fp32", "bf16_mixed@fp32", "tf32@fp32", "pytorch_autocast"],
    )
    batch2, config2 = load_preset_batch(
        "era5_pretrained", ASSET, valid_time=datetime(2026, 7, 1, 6)
    )
    payload["one_step_2026"] = one_step_accuracy(
        config2,
        ckpt,
        batch2,
        device,
        ["fp32", "bf16_mixed@fp32", "tf32@fp32", "pytorch_autocast"],
    )

    print("=== torch.compile after load ===", flush=True)
    compile_rows = {}
    for name, precision in (
        ("unfused_fp32_compile", "fp32"),
        ("autocast_compile", "pytorch_autocast"),
        ("bf16_mixed_compile", "bf16_mixed@fp32"),
    ):
        print(f"  [compile] {name}", flush=True)
        purge_gpu()
        model = None
        try:
            model = build_model(config, ckpt, precision=precision, device=device)
            compile_after_load(model)
            samples, peak = time_samples(
                model, batch.to(device), warmup=11, repeat=8, device=device
            )
            row = {"ok": True, **stats(samples), "peak_gib": peak}
            print(
                f"    mean={row['mean']:.1f} std={row['std']:.2f} p95={row['p95']:.1f} "
                f"peak={peak:.1f} GiB",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            row = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            print(f"    FAIL {row['error'][:400]}", flush=True)
        compile_rows[name] = row
        del model
        purge_gpu()
        gc.collect()
    payload["compile_after_load"] = compile_rows

    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
