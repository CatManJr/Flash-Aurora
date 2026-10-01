"""Run a test body on several ranks in fresh processes.

Each rank builds its own copy of the module under test from the same seed, so a
worker can compare its parallel output with a single-process reference locally.
An assertion error on any rank fails the spawning test. The default backend is gloo
(CPU); the NCCL variant binds rank ``r`` to ``cuda:r``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

GLOO_BACKEND = "gloo"
NCCL_BACKEND = "nccl"


def _rank_entry(
    rank: int,
    world_size: int,
    init_file: str,
    backend: str,
    worker: Callable[..., None],
    worker_args: tuple[Any, ...],
) -> None:
    if backend == NCCL_BACKEND:
        torch.cuda.set_device(rank)
    dist.init_process_group(
        backend, init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        worker(*worker_args)
    finally:
        dist.destroy_process_group()


def _run_on_ranks(
    backend: str,
    world_size: int,
    worker: Callable[..., None],
    tmp_path: Path,
    worker_args: tuple[Any, ...],
) -> None:
    init_file = (tmp_path / f"{backend}_init").as_posix()
    mp.spawn(
        _rank_entry,
        args=(world_size, init_file, backend, worker, worker_args),
        nprocs=world_size,
        join=True,
    )


def run_on_gloo_ranks(
    world_size: int,
    worker: Callable[..., None],
    tmp_path: Path,
    *worker_args: Any,
) -> None:
    """Call ``worker(*worker_args)`` on ``world_size`` gloo ranks; ``worker`` must be top-level."""
    _run_on_ranks(GLOO_BACKEND, world_size, worker, tmp_path, worker_args)


def run_on_nccl_ranks(
    world_size: int,
    worker: Callable[..., None],
    tmp_path: Path,
    *worker_args: Any,
) -> None:
    """Call ``worker(*worker_args)`` on ``world_size`` NCCL ranks, one GPU per rank."""
    _run_on_ranks(NCCL_BACKEND, world_size, worker, tmp_path, worker_args)
