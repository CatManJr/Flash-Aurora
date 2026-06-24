import os
import signal
import subprocess
from pathlib import Path

from flash_aurora.scheduler.processes import (
    cleanup_scheduler_ipc_files,
    cleanup_stale_scheduler_processes,
    find_stale_scheduler_processes,
    shutdown_scheduler_subprocess,
    shutdown_scheduler_subprocesses,
)


def test_find_stale_scheduler_processes_scopes_to_socket_dir(
    monkeypatch,
    tmp_path: Path,
) -> None:
    socket_dir = tmp_path / "scheduler_distributed_workers"
    current_pid = os.getpid()
    stdout = "\n".join(
        [
            f"{current_pid} python -m flash_aurora.scheduler --command-addr ipc://{socket_dir}/self.ipc",
            f"123 python -m flash_aurora.scheduler --command-addr ipc://{socket_dir}/worker.ipc",
            "124 python -m flash_aurora.scheduler --command-addr ipc:///other/worker.ipc",
            f"125 python train.py --path {socket_dir}",
        ]
    )

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    processes = find_stale_scheduler_processes(socket_dir)

    assert [process.pid for process in processes] == [123]


def test_cleanup_stale_scheduler_processes_uses_term_before_kill(
    monkeypatch,
    tmp_path: Path,
) -> None:
    socket_dir = tmp_path / "scheduler_single_worker"
    stdout = f"123 python -m flash_aurora.scheduler --command-addr ipc://{socket_dir}/worker.ipc"
    killed: list[tuple[int, int]] = []
    terminated = set()

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")

    def fake_kill(pid: int, sig: int) -> None:
        killed.append((pid, sig))
        if sig == signal.SIGTERM:
            terminated.add(pid)
        elif sig == 0 and pid in terminated:
            raise ProcessLookupError

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(os, "kill", fake_kill)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    cleaned = cleanup_stale_scheduler_processes(socket_dir, log=lambda _message: None)

    assert cleaned == [123]
    assert (123, signal.SIGTERM) in killed
    assert (123, signal.SIGKILL) not in killed


def test_cleanup_scheduler_ipc_files_removes_ipc_files(tmp_path: Path) -> None:
    socket_dir = tmp_path / "scheduler"
    socket_dir.mkdir()
    ipc_file = socket_dir / "worker.ipc"
    other_file = socket_dir / "notes.txt"
    ipc_file.write_text("")
    other_file.write_text("")

    removed = cleanup_scheduler_ipc_files(socket_dir, log=lambda _message: None)

    assert removed == [ipc_file]
    assert not ipc_file.exists()
    assert other_file.exists()


def test_shutdown_scheduler_subprocess_waits_for_graceful_exit(monkeypatch) -> None:
    shutdown_calls: list[str] = []
    close_calls: list[str] = []
    terminate_calls: list[int] = []

    class FakeClient:
        def shutdown_worker(self) -> None:
            shutdown_calls.append("shutdown")

        def close(self) -> None:
            close_calls.append("close")

    class FakeProc:
        def __init__(self) -> None:
            self._alive = True

        def poll(self) -> int | None:
            return None if self._alive else 0

    proc = FakeProc()

    def fake_wait(procs, *, timeout_s: float):
        proc._alive = False
        return []

    monkeypatch.setattr(
        "flash_aurora.scheduler.processes._wait_for_processes",
        fake_wait,
    )
    monkeypatch.setattr(
        "flash_aurora.scheduler.processes.terminate_process_tree",
        lambda proc, timeout_s=10.0: terminate_calls.append(proc.pid if hasattr(proc, "pid") else 0),
    )

    shutdown_scheduler_subprocess(FakeClient(), proc)  # type: ignore[arg-type]

    assert shutdown_calls == ["shutdown"]
    assert close_calls == ["close"]
    assert terminate_calls == []


def test_shutdown_scheduler_subprocesses_force_kills_stragglers(monkeypatch) -> None:
    killed: list[str] = []

    class FakeClient:
        def shutdown_worker(self) -> None:
            pass

        def close(self) -> None:
            pass

    class FakeProc:
        pid = 42

        def poll(self) -> int | None:
            return None

    proc = FakeProc()

    monkeypatch.setattr(
        "flash_aurora.scheduler.processes._wait_for_processes",
        lambda _procs, timeout_s: [proc],
    )
    monkeypatch.setattr(
        "flash_aurora.scheduler.processes.terminate_process_tree",
        lambda remaining, timeout_s=10.0: killed.append("force"),
    )

    shutdown_scheduler_subprocesses(FakeClient(), [proc])  # type: ignore[arg-type]

    assert killed == ["force"]
