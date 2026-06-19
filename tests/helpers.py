from __future__ import annotations

from datetime import datetime

import torch

from aurora import Batch, Metadata


def smoke_batch() -> Batch:
    return Batch(
        surf_vars={k: torch.randn(1, 2, 17, 32) for k in ("2t", "10u", "10v", "msl")},
        static_vars={k: torch.randn(17, 32) for k in ("lsm", "z", "slt")},
        atmos_vars={k: torch.randn(1, 2, 4, 17, 32) for k in ("z", "u", "v", "t", "q")},
        metadata=Metadata(
            lat=torch.linspace(90, -90, 17),
            lon=torch.linspace(0, 360, 32 + 1)[:-1],
            time=(datetime(2020, 6, 1, 12, 0),),
            atmos_levels=(100, 250, 500, 850),
        ),
    )


def assert_batches_close(left: Batch, right: Batch, *, atol: float = 1e-5) -> None:
    for name in left.surf_vars:
        assert torch.allclose(left.surf_vars[name].cpu(), right.surf_vars[name].cpu(), atol=atol)
    for name in left.atmos_vars:
        assert torch.allclose(left.atmos_vars[name].cpu(), right.atmos_vars[name].cpu(), atol=atol)
