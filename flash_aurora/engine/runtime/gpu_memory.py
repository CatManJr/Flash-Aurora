from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class CudaMemorySnapshot:
    device_index: int
    free_gib: float
    total_gib: float
    torch_allocated_gib: float | None
    torch_reserved_gib: float | None
    other_processes_gib: float | None

    @property
    def used_gib(self) -> float:
        return self.total_gib - self.free_gib


def _gib(bytes_value: int | float) -> float:
    return float(bytes_value) / (1024**3)


def _nvidia_smi_used_by_others(device_index: int, current_pid: int) -> float | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None

    total_mib = 0.0
    for line in output.strip().splitlines():
        if not line.strip():
            continue
        pid_text, used_text = (part.strip() for part in line.split(",", maxsplit=1))
        if int(pid_text) == current_pid:
            continue
        total_mib += float(used_text)
    return _gib(total_mib * 1024 * 1024)


def cuda_memory_snapshot(*, device_index: int = 0) -> CudaMemorySnapshot:
    """Return current CUDA memory stats for ``device_index``."""
    import os

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    torch_allocated: float | None
    torch_reserved: float | None
    try:
        torch_allocated = _gib(torch.cuda.memory_allocated(device_index))
        torch_reserved = _gib(torch.cuda.memory_reserved(device_index))
    except Exception:
        torch_allocated = None
        torch_reserved = None

    return CudaMemorySnapshot(
        device_index=device_index,
        free_gib=_gib(free_bytes),
        total_gib=_gib(total_bytes),
        torch_allocated_gib=torch_allocated,
        torch_reserved_gib=torch_reserved,
        other_processes_gib=_nvidia_smi_used_by_others(device_index, os.getpid()),
    )


def format_cuda_memory_snapshot(snapshot: CudaMemorySnapshot) -> str:
    lines = [
        f"GPU {snapshot.device_index}: {snapshot.free_gib:.1f} GiB free / {snapshot.total_gib:.1f} GiB total",
    ]
    if snapshot.torch_allocated_gib is not None:
        lines.append(
            f"  this process (PyTorch): {snapshot.torch_allocated_gib:.1f} GiB allocated, "
            f"{snapshot.torch_reserved_gib:.1f} GiB reserved"
        )
    if snapshot.other_processes_gib is not None and snapshot.other_processes_gib > 0.05:
        lines.append(f"  other GPU processes: ~{snapshot.other_processes_gib:.1f} GiB")
    return "\n".join(lines)


def print_cuda_memory_summary(*, device_index: int = 0) -> CudaMemorySnapshot:
    snapshot = cuda_memory_snapshot(device_index=device_index)
    print(format_cuda_memory_snapshot(snapshot))
    return snapshot


def require_cuda_free_memory(
    min_free_gib: float,
    *,
    device_index: int = 0,
    context: str = "CUDA inference",
) -> CudaMemorySnapshot:
    """Fail fast with an actionable message when GPU memory is too tight."""
    snapshot = cuda_memory_snapshot(device_index=device_index)
    if snapshot.free_gib >= min_free_gib:
        return snapshot

    hints = [
        format_cuda_memory_snapshot(snapshot),
        f"{context} needs at least {min_free_gib:.0f} GiB free on GPU {device_index}.",
    ]
    if snapshot.other_processes_gib is not None and snapshot.other_processes_gib > 1.0:
        hints.append(
            "Other Python/Jupyter kernels are using this GPU. Shut down extra notebooks "
            "or run: nvidia-smi  # then kill stale PIDs"
        )
    else:
        hints.append(
            "Restart this Jupyter kernel after a failed run, then retry with one kernel per GPU."
        )
    raise RuntimeError("\n".join(hints))
