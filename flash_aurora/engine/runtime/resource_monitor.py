from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GpuResourceSample:
    index: int
    name: str | None
    utilization_percent: float | None
    memory_used_mib: float | None
    memory_total_mib: float | None

    @property
    def memory_used_gib(self) -> float | None:
        if self.memory_used_mib is None:
            return None
        return self.memory_used_mib / 1024.0

    @property
    def memory_utilization_percent(self) -> float | None:
        if self.memory_used_mib is None or self.memory_total_mib in (None, 0.0):
            return None
        return 100.0 * self.memory_used_mib / self.memory_total_mib


@dataclass(frozen=True)
class ResourceSample:
    time_s: float
    cpu_percent: float
    dram_used_gib: float
    dram_utilization_percent: float
    gpus: dict[int, GpuResourceSample]


def device_index_from_name(device: str) -> int:
    if not device.startswith("cuda:"):
        raise ValueError(f"expected CUDA device name like 'cuda:0', got {device!r}")
    return int(device.split(":", 1)[1])


def _parse_float(value: str | None) -> float | None:
    if value in (None, "?"):
        return None
    return float(value)


def _read_cpu_snapshot() -> tuple[float, float]:
    cpu_line = Path("/proc/stat").read_text().splitlines()[0]
    values = [float(value) for value in cpu_line.split()[1:]]
    idle_time = values[3] + values[4]
    total_time = sum(values)
    return idle_time, total_time


def _calculate_cpu_percent(
    previous_snapshot: tuple[float, float],
    current_snapshot: tuple[float, float],
) -> float:
    previous_idle, previous_total = previous_snapshot
    current_idle, current_total = current_snapshot
    total_delta = current_total - previous_total
    if total_delta <= 0:
        return 0.0
    idle_delta = current_idle - previous_idle
    return max(0.0, min(100.0, 100.0 * (1.0 - idle_delta / total_delta)))


def _read_dram_status() -> tuple[float, float]:
    meminfo: dict[str, float] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value, *_ = line.replace(":", "").split()
        meminfo[key] = float(value)
    total_gib = meminfo["MemTotal"] / 1024.0 / 1024.0
    available_gib = meminfo["MemAvailable"] / 1024.0 / 1024.0
    used_gib = total_gib - available_gib
    utilization_percent = 100.0 * used_gib / total_gib
    return used_gib, utilization_percent


def query_gpu_status() -> dict[int, GpuResourceSample]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return {}

    status: dict[int, GpuResourceSample] = {}
    for line in result.stdout.strip().splitlines():
        index, name, memory_used, memory_total, utilization = [
            part.strip() for part in line.split(",")
        ]
        device_index = int(index)
        status[device_index] = GpuResourceSample(
            index=device_index,
            name=name,
            utilization_percent=_parse_float(utilization),
            memory_used_mib=_parse_float(memory_used),
            memory_total_mib=_parse_float(memory_total),
        )
    return status


class ResourceMonitor:
    def __init__(self, device_indices: list[int], interval_s: float = 1.0) -> None:
        self.device_indices = device_indices
        self.interval_s = interval_s
        self.samples: list[ResourceSample] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at: float | None = None
        self._previous_cpu_snapshot: tuple[float, float] | None = None

    def start(self) -> None:
        self.samples.clear()
        self._stop_event.clear()
        self._started_at = time.perf_counter()
        self._previous_cpu_snapshot = _read_cpu_snapshot()
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list[ResourceSample]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 1.0)
        self._sample()
        return list(self.samples)

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_s):
            self._sample()

    def _sample(self) -> None:
        if self._started_at is None or self._previous_cpu_snapshot is None:
            return

        current_cpu_snapshot = _read_cpu_snapshot()
        cpu_percent = _calculate_cpu_percent(
            self._previous_cpu_snapshot,
            current_cpu_snapshot,
        )
        self._previous_cpu_snapshot = current_cpu_snapshot

        dram_used_gib, dram_utilization_percent = _read_dram_status()
        gpu_status = query_gpu_status()
        gpu_samples = {
            device_index: gpu_status[device_index]
            for device_index in self.device_indices
            if device_index in gpu_status
        }
        sample = ResourceSample(
            time_s=time.perf_counter() - self._started_at,
            cpu_percent=cpu_percent,
            dram_used_gib=dram_used_gib,
            dram_utilization_percent=dram_utilization_percent,
            gpus=gpu_samples,
        )
        with self._lock:
            self.samples.append(sample)


def _gpu_utilization_series(
    samples: list[ResourceSample],
    device_index: int,
) -> list[float | None]:
    return [
        sample.gpus.get(device_index).utilization_percent
        if device_index in sample.gpus
        else None
        for sample in samples
    ]


def _vram_utilization_series(
    samples: list[ResourceSample],
    device_index: int,
) -> list[float | None]:
    return [
        sample.gpus.get(device_index).memory_utilization_percent
        if device_index in sample.gpus
        else None
        for sample in samples
    ]


def resource_samples_to_dict(samples: list[ResourceSample]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sample in samples:
        gpus: dict[str, dict[str, object]] = {}
        for index, gpu in sample.gpus.items():
            gpus[str(index)] = {
                "index": gpu.index,
                "name": gpu.name,
                "utilization_percent": gpu.utilization_percent,
                "memory_used_mib": gpu.memory_used_mib,
                "memory_total_mib": gpu.memory_total_mib,
            }
        rows.append(
            {
                "time_s": sample.time_s,
                "cpu_percent": sample.cpu_percent,
                "dram_used_gib": sample.dram_used_gib,
                "dram_utilization_percent": sample.dram_utilization_percent,
                "gpus": gpus,
            }
        )
    return rows


def _apply_resource_plot_style() -> None:
    """Match ``docs/example_scheduler_distributed_workers.ipynb`` plotting settings."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "figure.facecolor": "white",
        }
    )


def _plot_resource_grid(
    axes,
    samples: list[ResourceSample],
    device_indices: list[int],
    device_labels: dict[int, str],
) -> None:
    """Fill a 2×2 axis grid: CPU, DRAM, GPU util, VRAM util."""
    times = [sample.time_s for sample in samples]

    axes[0, 0].plot(times, [sample.cpu_percent for sample in samples], color="tab:blue")
    axes[0, 0].set_title("CPU utilization")
    axes[0, 0].set_ylabel("percent")
    axes[0, 0].set_ylim(0, 100)

    axes[0, 1].plot(
        times,
        [sample.dram_utilization_percent for sample in samples],
        color="tab:green",
    )
    axes[0, 1].set_title("DRAM utilization")
    axes[0, 1].set_ylabel("percent")
    axes[0, 1].set_ylim(0, 100)

    for device_index in device_indices:
        label = device_labels.get(device_index, f"cuda:{device_index}")
        axes[1, 0].plot(
            times,
            _gpu_utilization_series(samples, device_index),
            label=label,
        )
        axes[1, 1].plot(
            times,
            _vram_utilization_series(samples, device_index),
            label=label,
        )

    axes[1, 0].set_title("GPU utilization")
    axes[1, 0].set_xlabel("seconds")
    axes[1, 0].set_ylabel("percent")
    axes[1, 0].set_ylim(0, 100)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].set_title("VRAM utilization")
    axes[1, 1].set_xlabel("seconds")
    axes[1, 1].set_ylabel("percent")
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].legend(fontsize=8)

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)


def plot_resource_usage_figure(
    samples: list[ResourceSample],
    *,
    output_path: Path | str,
    device_indices: list[int],
    device_labels: dict[int, str] | None = None,
    title: str = "Resource usage",
) -> Path:
    """Save a single notebook-style 2×2 utilization figure."""
    import matplotlib.pyplot as plt

    _apply_resource_plot_style()
    device_labels = device_labels or {}

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    fig.suptitle(title)
    _plot_resource_grid(axes, samples, device_indices, device_labels)
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor="white")
    plt.close(fig)
    return output


def _utilization_output_paths(
    output_path: Path | str,
    mode_names: list[str],
) -> list[Path]:
    """Map each rollout mode to its own image path."""
    output = Path(output_path)
    if output.suffix:
        return [output.parent / f"{output.stem}_{mode}{output.suffix}" for mode in mode_names]
    output.mkdir(parents=True, exist_ok=True)
    return [output / f"{mode}.png" for mode in mode_names]


def plot_distributed_rollout_utilization(
    traces: dict[str, list[ResourceSample]],
    *,
    device_indices: list[int],
    output_path: Path | str,
    title: str | None = None,
    device_labels: dict[int, str] | None = None,
) -> list[Path]:
    """Save one notebook-style 2×2 figure per rollout mode.

    *output_path* may be a directory (writes ``{mode}.png``) or a file stem such as
    ``docs/image/distributed_rollout_utilization_4090_era5_pretrained.png`` (writes
    ``distributed_rollout_utilization_4090_era5_pretrained_{mode}.png``).
    """
    if not traces:
        raise ValueError("traces must not be empty")
    for name, samples in traces.items():
        if not samples:
            raise ValueError(f"no samples for trace {name!r}")

    device_labels = device_labels or {}
    mode_names = list(traces.keys())
    paths = _utilization_output_paths(output_path, mode_names)

    written: list[Path] = []
    for mode, path in zip(mode_names, paths, strict=True):
        mode_title = mode.replace("_", " ")
        if title:
            mode_title = f"{title} ({mode_title})"
        written.append(
            plot_resource_usage_figure(
                traces[mode],
                output_path=path,
                device_indices=device_indices,
                device_labels=device_labels,
                title=mode_title,
            )
        )
    return written


def plot_resource_usage(
    samples: list[ResourceSample],
    *,
    device_labels: dict[int, str] | None = None,
    title: str = "Resource usage",
) -> None:
    if not samples:
        print("no resource samples collected")
        return

    import matplotlib.pyplot as plt

    _apply_resource_plot_style()
    device_labels = device_labels or {}
    device_indices = sorted({device_index for sample in samples for device_index in sample.gpus})

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    fig.suptitle(title)
    _plot_resource_grid(axes, samples, device_indices, device_labels)
    fig.tight_layout()
    plt.show()
