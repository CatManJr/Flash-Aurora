from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypeVar

from flash_aurora.engine.ingress.download.credentials import (
    active_download_credentials,
    run_with_credentials,
)

T = TypeVar("T")


def run_labeled_tasks(
    tasks: Sequence[tuple[str, Callable[[], T]]],
    *,
    workers: int,
    description: str | None = None,
    show_progress: bool = False,
) -> dict[str, T]:
    """Run independent callables, optionally in parallel.

    When ``workers <= 1`` or only one task is provided, tasks run sequentially in
    declaration order. The first raised exception is propagated after cancelling
    outstanding futures.

    Parallel workers inherit the caller's ``contextvars`` (e.g. download credentials).
    """
    if not tasks:
        return {}

    if workers <= 1 or len(tasks) == 1:
        return {label: fn() for label, fn in tasks}

    max_workers = min(workers, len(tasks))
    results: dict[str, T] = {}
    captured_credentials = active_download_credentials()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(run_with_credentials, captured_credentials, fn): label
            for label, fn in tasks
        }
        completed = as_completed(futures)
        if show_progress:
            from tqdm.auto import tqdm

            completed = tqdm(
                completed,
                total=len(futures),
                desc=description or "download",
                unit="task",
            )

        try:
            for future in completed:
                label = futures[future]
                results[label] = future.result()
        except Exception:
            for pending in futures:
                pending.cancel()
            raise

    return results
