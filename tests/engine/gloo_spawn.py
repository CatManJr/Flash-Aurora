"""Run a test body on several gloo ranks in fresh processes.

Each rank builds its own copy of the module under test from the same seed, so a
worker can compare its parallel output with a single-process reference locally.
An assertion error on any rank fails the spawning test.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch.distributed as dist
import torch.multiprocessing as mp


def _rank_entry(
    rank: int,
    world_size: int,
    init_file: str,
    worker: Callable[..., None],
    worker_args: tuple[Any, ...],
) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        worker(*worker_args)
    finally:
        dist.destroy_process_group()


def run_on_gloo_ranks(
    world_size: int,
    worker: Callable[..., None],
    tmp_path: Path,
    *worker_args: Any,
) -> None:
    """Call ``worker(*worker_args)`` on ``world_size`` gloo ranks; ``worker`` must be top-level."""
    init_file = (tmp_path / "gloo_init").as_posix()
    mp.spawn(
        _rank_entry,
        args=(world_size, init_file, worker, worker_args),
        nprocs=world_size,
        join=True,
    )
