from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from flash_aurora.scheduler.client import ForecastClient


@dataclass(frozen=True)
class SchedulerProcess:
    pid: int
    command: str


def find_stale_scheduler_processes(socket_dir: Path | str) -> list[SchedulerProcess]:
    """Find scheduler subprocesses that belong to one tutorial socket directory."""

    marker = str(Path(socket_dir))
    result = subprocess.run(
        ["ps", "-eo", "pid=,cmd="],
        check=False,
        capture_output=True,
        text=True,
    )
    processes: list[SchedulerProcess] = []
    for line in result.stdout.splitlines():
        pid_text, _, command = line.strip().partition(" ")
        if not pid_text.isdigit():
            continue
        pid = int(pid_text)
        if pid == os.getpid():
            continue
        if marker in command and "flash_aurora.scheduler" in command:
            processes.append(SchedulerProcess(pid=pid, command=command))
    return processes


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def cleanup_scheduler_ipc_files(socket_dir: Path | str, *, log: Callable[[str], None] = print) -> list[Path]:
    """Remove stale tutorial IPC socket files after matching scheduler processes stop."""

    directory = Path(socket_dir)
    if not directory.exists():
        return []
    removed: list[Path] = []
    for path in sorted(directory.glob("*.ipc")):
        try:
            path.unlink()
            removed.append(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            log(f"could not remove stale IPC file {path}: {exc}")
    return removed


def cleanup_stale_scheduler_processes(
    socket_dir: Path | str,
    *,
    timeout_s: float = 10.0,
    remove_ipc_files: bool = True,
    log: Callable[[str], None] = print,
) -> list[int]:
    """Terminate stale scheduler subprocesses scoped to one tutorial socket directory."""

    processes = find_stale_scheduler_processes(socket_dir)
    if not processes:
        log("no stale scheduler processes found")
        if remove_ipc_files:
            removed = cleanup_scheduler_ipc_files(socket_dir, log=log)
            if removed:
                log(f"removed {len(removed)} stale IPC files")
        return []

    log("stale scheduler processes:")
    for process in processes:
        log(f"{process.pid} {process.command}")
        try:
            os.kill(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.time() + timeout_s
    remaining = {process.pid for process in processes}
    while remaining and time.time() < deadline:
        time.sleep(0.5)
        for pid in list(remaining):
            if not _process_exists(pid):
                remaining.remove(pid)

    for pid in sorted(remaining):
        log(f"force killing stale scheduler process: {pid}")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    log("stale scheduler cleanup completed")
    if remove_ipc_files:
        removed = cleanup_scheduler_ipc_files(socket_dir, log=log)
        if removed:
            log(f"removed {len(removed)} stale IPC files")
    return [process.pid for process in processes]


def _terminate_pid(pid: int, *, timeout_s: float) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not _process_exists(pid):
            return
        time.sleep(0.2)

    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def terminate_process_tree(proc: subprocess.Popen[bytes] | None, *, timeout_s: float = 10.0) -> None:
    """Terminate a subprocess and its process group when it was started with a new session."""

    if proc is None or proc.poll() is not None:
        return
    _terminate_pid(proc.pid, timeout_s=timeout_s)
    try:
        proc.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=1.0)


def _wait_for_processes(
    procs: list[subprocess.Popen[bytes] | None],
    *,
    timeout_s: float,
) -> list[subprocess.Popen[bytes]]:
    deadline = time.time() + timeout_s
    alive = [proc for proc in procs if proc is not None and proc.poll() is None]
    while alive and time.time() < deadline:
        time.sleep(0.2)
        alive = [proc for proc in alive if proc.poll() is None]
    return alive


def shutdown_scheduler_subprocess(
    client: ForecastClient | None,
    proc: subprocess.Popen[bytes] | None,
    *,
    grace_timeout_s: float = 30.0,
    force_timeout_s: float = 10.0,
) -> None:
    """Shut down one scheduler subprocess through the client shutdown lifecycle."""

    if client is not None:
        try:
            client.shutdown_worker()
        finally:
            client.close()

    for remaining in _wait_for_processes([proc], timeout_s=grace_timeout_s):
        terminate_process_tree(remaining, timeout_s=force_timeout_s)


def shutdown_scheduler_subprocesses(
    client: ForecastClient | None,
    procs: list[subprocess.Popen[bytes] | None],
    *,
    grace_timeout_s: float = 30.0,
    force_timeout_s: float = 10.0,
) -> None:
    """Shut down scheduler subprocesses through the client shutdown lifecycle."""

    if client is not None:
        try:
            client.shutdown_worker()
        finally:
            client.close()

    for remaining in _wait_for_processes(procs, timeout_s=grace_timeout_s):
        terminate_process_tree(remaining, timeout_s=force_timeout_s)
