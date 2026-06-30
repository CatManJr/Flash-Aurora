"""CUDA allocator tuning and explicit tensor release helpers."""

from __future__ import annotations

import os

import torch
from flash_aurora.aurora import Batch


def configure_pytorch_cuda_allocator(*, expandable_segments: bool = True) -> None:
    """Set allocator options before the first CUDA allocation in this process."""
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").strip():
        return
    if expandable_segments:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def release_tensor_storage(tensor: torch.Tensor) -> None:
    """Drop backing storage for a CUDA tensor while keeping a harmless placeholder."""
    if tensor.is_cuda:
        tensor.data = torch.empty(0, device=tensor.device)


def release_batch_gpu_storage(batch: Batch) -> None:
    """Release CUDA storage held by rollout/export batches."""
    for mapping in (batch.surf_vars, batch.atmos_vars, batch.static_vars):
        for name, tensor in mapping.items():
            if tensor.is_cuda:
                release_tensor_storage(tensor)
                mapping[name] = tensor
    if batch.metadata.lat.is_cuda:
        release_tensor_storage(batch.metadata.lat)
        batch.metadata.lat = batch.metadata.lat
    if batch.metadata.lon.is_cuda:
        release_tensor_storage(batch.metadata.lon)
        batch.metadata.lon = batch.metadata.lon


def trim_cuda_cache(*devices: torch.device | str) -> None:
    """Return cached blocks to the driver after large tensors are released."""
    if not torch.cuda.is_available():
        return
    for device in devices:
        dev = torch.device(device)
        if dev.type != "cuda":
            continue
        with torch.cuda.device(dev):
            torch.cuda.empty_cache()
