from __future__ import annotations

import atexit
import fcntl
import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

from flash_aurora.engine.core.config import ModelVariantSpec
from flash_aurora.engine.runtime.gpu_budget import estimate_vram_gib, is_exclusive_variant
from flash_aurora.engine.runtime.gpu_memory import cuda_memory_snapshot, format_cuda_memory_snapshot

_HEARTBEAT_SECONDS = 30.0
_STALE_SECONDS = 120.0
_POLL_SECONDS = 2.0
_DEFAULT_TIMEOUT_SECONDS = 3600.0
_RESERVED_FRACTION = 0.96


def gpu_guard_enabled() -> bool:
    value = os.environ.get("FLASH_AURORA_GPU_GUARD", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def resolve_guard_dir(asset_root: Path | str | None) -> Path:
    override = os.environ.get("FLASH_AURORA_GPU_GUARD_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if asset_root is not None:
        return Path(asset_root).expanduser().resolve() / ".flash-aurora" / "gpu_guard"
    return (Path.cwd() / ".flash-aurora" / "gpu_guard").resolve()


def try_local_cuda_cleanup(*, device_index: int = 0) -> None:
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()
    try:
        torch.cuda.synchronize(device_index)
    except Exception:
        pass


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


@dataclass
class GpuLeaseRecord:
    lease_id: str
    pid: int
    preset: str
    variant: str
    device_index: int
    reserved_gib: float
    exclusive: bool
    rollout_steps: int
    heartbeat: float = field(default_factory=time.time)


@dataclass
class GpuQueueRecord:
    pid: int
    preset: str
    variant: str
    needed_gib: float
    exclusive: bool
    rollout_steps: int
    enqueued_at: float = field(default_factory=time.time)


@dataclass
class GpuDeviceState:
    leases: list[GpuLeaseRecord] = field(default_factory=list)
    queue: list[GpuQueueRecord] = field(default_factory=list)


@dataclass
class GpuGuardTicket:
    lease_id: str
    device_index: int
    reserved_gib: float
    exclusive: bool
    _registry: GpuGuardRegistry
    _released: bool = False

    def heartbeat(self) -> None:
        self._registry.heartbeat(self.lease_id, device_index=self.device_index)

    def release(self) -> None:
        if self._released:
            return
        self._registry.release(self.lease_id, device_index=self.device_index)
        self._released = True


class GpuGuardRegistry:
    """Cross-process CUDA lease registry backed by a JSON file per device."""

    def __init__(self, guard_dir: Path) -> None:
        self._guard_dir = guard_dir
        self._guard_dir.mkdir(parents=True, exist_ok=True)

    def _state_path(self, device_index: int) -> Path:
        return self._guard_dir / f"device_{device_index}.json"

    def _lock_path(self, device_index: int) -> Path:
        return self._guard_dir / f"device_{device_index}.lock"

    @contextmanager
    def _locked(self, device_index: int) -> Iterator[None]:
        lock_path = self._lock_path(device_index)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _load_state(self, device_index: int) -> GpuDeviceState:
        path = self._state_path(device_index)
        if not path.is_file():
            return GpuDeviceState()
        payload = json.loads(path.read_text(encoding="utf-8"))
        return GpuDeviceState(
            leases=[GpuLeaseRecord(**item) for item in payload.get("leases", [])],
            queue=[GpuQueueRecord(**item) for item in payload.get("queue", [])],
        )

    def _save_state(self, device_index: int, state: GpuDeviceState) -> None:
        path = self._state_path(device_index)
        payload = {
            "leases": [asdict(item) for item in state.leases],
            "queue": [asdict(item) for item in state.queue],
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    def _purge_stale(self, state: GpuDeviceState, *, now: float) -> None:
        state.leases = [
            lease
            for lease in state.leases
            if _pid_alive(lease.pid) and (now - lease.heartbeat) <= _STALE_SECONDS
        ]
        state.queue = [item for item in state.queue if _pid_alive(item.pid)]

    def _reserved_gib(self, state: GpuDeviceState, *, exclude_pid: int | None = None) -> float:
        total = 0.0
        for lease in state.leases:
            if exclude_pid is not None and lease.pid == exclude_pid:
                continue
            total += lease.reserved_gib
        return total

    def _has_exclusive_lease(self, state: GpuDeviceState, *, exclude_pid: int | None = None) -> bool:
        for lease in state.leases:
            if exclude_pid is not None and lease.pid == exclude_pid:
                continue
            if lease.exclusive:
                return True
        return False

    def _can_grant(
        self,
        state: GpuDeviceState,
        *,
        needed_gib: float,
        exclusive: bool,
        pid: int,
        device_index: int,
    ) -> tuple[bool, str]:
        snapshot = cuda_memory_snapshot(device_index=device_index)
        reserved = self._reserved_gib(state, exclude_pid=pid)
        projected = reserved + needed_gib

        if exclusive and self._has_exclusive_lease(state, exclude_pid=pid):
            return False, "another exclusive Aurora job holds this GPU"

        if projected > snapshot.total_gib * _RESERVED_FRACTION:
            return (
                False,
                f"projected reservations {projected:.1f} GiB exceed GPU budget "
                f"({snapshot.total_gib * _RESERVED_FRACTION:.1f} GiB)",
            )

        min_free = needed_gib if exclusive else needed_gib * 0.75
        if snapshot.free_gib < min_free:
            return (
                False,
                f"only {snapshot.free_gib:.1f} GiB free, need ~{min_free:.1f} GiB",
            )

        if exclusive and reserved > 0.5 and snapshot.free_gib < needed_gib:
            return False, "exclusive job needs an empty GPU"

        return True, "ok"

    def _queue_position(self, state: GpuDeviceState, pid: int) -> int | None:
        for index, item in enumerate(state.queue):
            if item.pid == pid:
                return index + 1
        return None

    def acquire(
        self,
        *,
        device_index: int,
        preset: str,
        variant: ModelVariantSpec,
        rollout_steps: int = 1,
        inference_precision: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        queue: bool = True,
    ) -> GpuGuardTicket:
        pid = os.getpid()
        needed = estimate_vram_gib(
            variant,
            rollout_steps=rollout_steps,
            inference_precision=inference_precision,
        )
        exclusive = is_exclusive_variant(
            variant,
            rollout_steps=rollout_steps,
            inference_precision=inference_precision,
        )
        deadline = time.time() + timeout
        last_message = ""

        while time.time() < deadline:
            try_local_cuda_cleanup(device_index=device_index)
            with self._locked(device_index):
                now = time.time()
                state = self._load_state(device_index)
                self._purge_stale(state, now=now)

                for lease in state.leases:
                    if lease.pid == pid:
                        lease.heartbeat = now
                        self._save_state(device_index, state)
                        ticket = GpuGuardTicket(
                            lease_id=lease.lease_id,
                            device_index=device_index,
                            reserved_gib=lease.reserved_gib,
                            exclusive=lease.exclusive,
                            _registry=self,
                        )
                        _register_ticket(ticket)
                        return ticket

                position = self._queue_position(state, pid)
                if position is None and queue:
                    state.queue.append(
                        GpuQueueRecord(
                            pid=pid,
                            preset=preset,
                            variant=variant.name,
                            needed_gib=needed,
                            exclusive=exclusive,
                            rollout_steps=rollout_steps,
                        )
                    )
                    position = len(state.queue)

                can_grant, reason = self._can_grant(
                    state,
                    needed_gib=needed,
                    exclusive=exclusive,
                    pid=pid,
                    device_index=device_index,
                )
                if position is not None and position > 1:
                    can_grant = False
                    reason = f"queued at position {position}"

                if can_grant and (position is None or position == 1):
                    state.queue = [item for item in state.queue if item.pid != pid]
                    lease = GpuLeaseRecord(
                        lease_id=str(uuid.uuid4()),
                        pid=pid,
                        preset=preset,
                        variant=variant.name,
                        device_index=device_index,
                        reserved_gib=needed,
                        exclusive=exclusive,
                        rollout_steps=rollout_steps,
                        heartbeat=now,
                    )
                    state.leases.append(lease)
                    self._save_state(device_index, state)
                    ticket = GpuGuardTicket(
                        lease_id=lease.lease_id,
                        device_index=device_index,
                        reserved_gib=needed,
                        exclusive=exclusive,
                        _registry=self,
                    )
                    _register_ticket(ticket)
                    return ticket

                last_message = reason
                self._save_state(device_index, state)

            if not queue:
                break
            time.sleep(_POLL_SECONDS)

        snapshot = cuda_memory_snapshot(device_index=device_index)
        raise TimeoutError(
            "Timed out waiting for GPU "
            f"{device_index} ({preset}, ~{needed:.0f} GiB, "
            f"{'exclusive' if exclusive else 'shareable'}).\n"
            f"{format_cuda_memory_snapshot(snapshot)}\n"
            f"Last blocker: {last_message}"
        )

    def heartbeat(self, lease_id: str, *, device_index: int) -> None:
        with self._locked(device_index):
            state = self._load_state(device_index)
            now = time.time()
            for lease in state.leases:
                if lease.lease_id == lease_id:
                    lease.heartbeat = now
                    self._save_state(device_index, state)
                    return

    def release(self, lease_id: str, *, device_index: int) -> None:
        with self._locked(device_index):
            state = self._load_state(device_index)
            state.leases = [lease for lease in state.leases if lease.lease_id != lease_id]
            state.queue = [item for item in state.queue if item.pid != os.getpid()]
            self._save_state(device_index, state)

    def status(self, *, device_index: int = 0) -> GpuDeviceState:
        with self._locked(device_index):
            state = self._load_state(device_index)
            self._purge_stale(state, now=time.time())
            self._save_state(device_index, state)
            return state


_ACTIVE_TICKETS: list[GpuGuardTicket] = []


def _register_ticket(ticket: GpuGuardTicket) -> None:
    _ACTIVE_TICKETS[:] = [item for item in _ACTIVE_TICKETS if not item._released]
    _ACTIVE_TICKETS.append(ticket)


def _release_active_tickets() -> None:
    for ticket in list(_ACTIVE_TICKETS):
        ticket.release()


atexit.register(_release_active_tickets)


@contextmanager
def gpu_guard_session(
    *,
    asset_root: Path | str | None,
    device: str,
    preset: str,
    variant: ModelVariantSpec,
    rollout_steps: int = 1,
    inference_precision: str | None = None,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    enabled: bool | None = None,
) -> Iterator[GpuGuardTicket | None]:
    """Acquire a cross-process GPU lease for the duration of a context block."""
    if enabled is None:
        enabled = gpu_guard_enabled()
    if not enabled:
        yield None
        return

    device_index = 0
    if device.startswith("cuda:"):
        device_index = int(device.split(":", 1)[1])

    try:
        import torch
    except ImportError:
        yield None
        return

    if not torch.cuda.is_available():
        yield None
        return

    registry = GpuGuardRegistry(resolve_guard_dir(asset_root))
    ticket = registry.acquire(
        device_index=device_index,
        preset=preset,
        variant=variant,
        rollout_steps=rollout_steps,
        inference_precision=inference_precision,
        timeout=timeout,
    )
    try:
        yield ticket
    finally:
        ticket.release()
