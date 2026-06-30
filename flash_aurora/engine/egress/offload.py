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
        self._pending_cpu: Batch | None = None
        if self._use_stream:
            self._stream = torch.cuda.Stream(device=device)

    def to_owned_cpu(self, batch: Batch) -> Batch:
        if not self._use_stream or self._stream is None:
            return owned_cpu_copy(batch)

        self.begin_async_d2h(batch)
        return self.finish_async_d2h()

    def begin_async_d2h(self, batch: Batch) -> None:
        """Start GPU→CPU copy on the egress stream without blocking compute."""
        if not self._use_stream or self._stream is None:
            self._pending_cpu = owned_cpu_copy(batch)
            return

        compute_stream = torch.cuda.current_stream(self._device)
        with torch.cuda.stream(self._stream):
            self._stream.wait_stream(compute_stream)
            for tensor in _iter_tensors(batch):
                if tensor.is_cuda:
                    tensor.record_stream(self._stream)
            self._pending_cpu = batch.to("cpu")

    def finish_async_d2h(self) -> Batch:
        """Wait for :meth:`begin_async_d2h` and return an owned CPU batch."""
        if not self._use_stream or self._stream is None:
            pending = getattr(self, "_pending_cpu", None)
            if pending is None:
                raise RuntimeError("no async D2H in progress")
            return pending

        compute_stream = torch.cuda.current_stream(self._device)
        compute_stream.wait_stream(self._stream)
        pending = getattr(self, "_pending_cpu", None)
        if pending is None:
            raise RuntimeError("no async D2H in progress")
        return owned_cpu_copy(pending)

    @property
    def egress_stream(self) -> torch.cuda.Stream | None:
        return self._stream


def _iter_tensors(batch: Batch):
    for group in (batch.surf_vars, batch.static_vars, batch.atmos_vars):
        yield from group.values()
    yield batch.metadata.lat
    yield batch.metadata.lon


class CpuOffloader:
    @staticmethod
    def to_cpu(batch: Batch) -> Batch:
        return batch.to("cpu")
