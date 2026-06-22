from __future__ import annotations

from unittest.mock import patch

import pytest

from flash_aurora.engine.runtime.gpu_memory import (
    CudaMemorySnapshot,
    format_cuda_memory_snapshot,
    require_cuda_free_memory,
)


def test_format_cuda_memory_snapshot_includes_other_processes() -> None:
    snapshot = CudaMemorySnapshot(
        device_index=0,
        free_gib=1.2,
        total_gib=95.0,
        torch_allocated_gib=17.0,
        torch_reserved_gib=26.0,
        other_processes_gib=67.0,
    )
    text = format_cuda_memory_snapshot(snapshot)
    assert "1.2 GiB free" in text
    assert "other GPU processes" in text


def test_require_cuda_free_memory_raises_with_actionable_message() -> None:
    snapshot = CudaMemorySnapshot(
        device_index=0,
        free_gib=2.0,
        total_gib=95.0,
        torch_allocated_gib=20.0,
        torch_reserved_gib=25.0,
        other_processes_gib=70.0,
    )
    with patch(
        "flash_aurora.engine.runtime.gpu_memory.cuda_memory_snapshot",
        return_value=snapshot,
    ):
        with pytest.raises(RuntimeError, match="Other Python/Jupyter kernels"):
            require_cuda_free_memory(40.0, context="HRES 0.1 rollout")
