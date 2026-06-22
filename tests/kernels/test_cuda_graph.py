"""Copyright (c) Catman Jr. Licensed under the MIT license.

CUDA graph capture tests.
"""

from __future__ import annotations

from datetime import datetime

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _smoke_batch(*, batch_size: int = 1, h: int = 32, w: int = 64) -> "Batch":
    from flash_aurora.aurora import Batch, Metadata

    levels = (100, 250, 500, 850)
    return Batch(
        surf_vars={k: torch.randn(batch_size, 2, h, w) for k in ("2t", "10u", "10v", "msl")},
        static_vars={k: torch.randn(h, w) for k in ("lsm", "z", "slt")},
        atmos_vars={
            k: torch.randn(batch_size, 2, len(levels), h, w) for k in ("z", "u", "v", "t", "q")
        },
        metadata=Metadata(
            lat=torch.linspace(90, -90, h),
            lon=torch.linspace(0, 360, w + 1)[:-1],
            time=(datetime(2020, 6, 1, 12, 0),),
            atmos_levels=levels,
        ),
    ).to("cuda")


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="CuTe tf32 preset targets recent NVIDIA GPUs",
)
def test_backbone_cuda_graph_capture_tf32() -> None:
    import os

    from flash_aurora.aurora.model.aurora import AuroraSmallPretrained
    from flash_aurora.aurora.model.checkpoint_local import resolve_checkpoint_path
    from flash_aurora.aurora.model.inference_tensors import clear_constant_tensor_cache

    asset_root = os.environ.get("AURORA_ASSET_ROOT") or os.environ.get("AURORA_HF_LOCAL_DIR")
    if not asset_root:
        pytest.skip("Set AURORA_ASSET_ROOT to a directory with aurora-0.25-small-pretrained.ckpt")

    clear_constant_tensor_cache()
    ckpt = resolve_checkpoint_path(
        filename="aurora-0.25-small-pretrained.ckpt",
        checkpoint_dir=asset_root,
        allow_hub_download=False,
    )
    batch = _smoke_batch()
    model = AuroraSmallPretrained(use_lora=False, inference_precision="tf32")
    model.load_checkpoint_local(str(ckpt), strict=False)
    model.eval().cuda()

    model.capture_inference_cuda_graph(batch, warmup_iters=2)
    assert model._cuda_graph_runner is not None
    assert model._cuda_graph_scope == "backbone"

    with torch.inference_mode():
        out = model.forward(batch)
    assert out.surf_vars["2t"].shape == (1, 1, 32, 64)

    clear_constant_tensor_cache()
