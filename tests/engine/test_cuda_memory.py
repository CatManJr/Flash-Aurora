"""Tests for explicit CUDA storage release helpers."""

from __future__ import annotations

import pytest
import torch

from flash_aurora.engine.runtime.cuda_memory import (
    configure_pytorch_cuda_allocator,
    release_batch_gpu_storage,
    release_tensor_storage,
)


def test_configure_pytorch_cuda_allocator_sets_default(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    configure_pytorch_cuda_allocator()
    assert "expandable_segments" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")


@pytest.mark.gpu
def test_release_tensor_storage_frees_cuda_allocation() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    tensor = torch.zeros(1024, 1024, device="cuda")
    ptr = tensor.data_ptr()
    release_tensor_storage(tensor)
    assert tensor.numel() == 0
    assert tensor.data_ptr() != ptr


@pytest.mark.gpu
def test_release_batch_gpu_storage() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from flash_aurora.aurora.batch import Batch, Metadata

    batch = Batch(
        surf_vars={"2t": torch.zeros(1, 1, 32, 32, device="cuda")},
        static_vars={},
        atmos_vars={"t": torch.zeros(1, 1, 4, 32, 32, device="cuda")},
        metadata=Metadata(
            lat=torch.linspace(90, -90, 32),
            lon=torch.linspace(0, 359, 32),
            time=(),
            atmos_levels=(850,),
        ),
    )
    release_batch_gpu_storage(batch)
    assert batch.surf_vars["2t"].numel() == 0
    assert batch.atmos_vars["t"].numel() == 0
