from __future__ import annotations

import torch
from flash_aurora.aurora import Batch, Metadata


def _own_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().clone()


def owned_cpu_copy(batch: Batch) -> Batch:
    """Return an owned CPU copy safe for async IO (Earth-2 non-reallocation contract)."""
    return Batch(
        surf_vars={k: _own_tensor(v) for k, v in batch.surf_vars.items()},
        static_vars={k: _own_tensor(v) for k, v in batch.static_vars.items()},
        atmos_vars={k: _own_tensor(v) for k, v in batch.atmos_vars.items()},
        metadata=Metadata(
            lat=_own_tensor(batch.metadata.lat),
            lon=_own_tensor(batch.metadata.lon),
            atmos_levels=batch.metadata.atmos_levels,
            time=batch.metadata.time,
            rollout_step=batch.metadata.rollout_step,
        ),
    )


class EgressOffloader:
    """GPU→CPU handoff for export, optionally on a dedicated CUDA stream."""

    def __init__(
        self,
        *,
        device: torch.device | None = None,
        use_stream: bool = True,
    ) -> None:
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._device = device
        self._use_stream = use_stream and device.type == "cuda"
        self._stream: torch.cuda.Stream | None = None
        if self._use_stream:
            self._stream = torch.cuda.Stream(device=device)

    def to_owned_cpu(self, batch: Batch) -> Batch:
        if not self._use_stream or self._stream is None:
            return owned_cpu_copy(batch)

        compute_stream = torch.cuda.current_stream(self._device)
        with torch.cuda.stream(self._stream):
            self._stream.wait_stream(compute_stream)
            for tensor in _iter_tensors(batch):
                if tensor.is_cuda:
                    tensor.record_stream(self._stream)
            cpu_batch = batch.to("cpu")
        compute_stream.wait_stream(self._stream)
        return owned_cpu_copy(cpu_batch)


def _iter_tensors(batch: Batch):
    for group in (batch.surf_vars, batch.static_vars, batch.atmos_vars):
        yield from group.values()
    yield batch.metadata.lat
    yield batch.metadata.lon


class CpuOffloader:
    @staticmethod
    def to_cpu(batch: Batch) -> Batch:
        return batch.to("cpu")
