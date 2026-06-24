"""Tests for the SchedulerSupervisor orphan detection and forced-kill logic."""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from flash_aurora.scheduler.supervisor import (
    OrphanProcess,
    SchedulerSupervisor,
    SupervisorReport,
    find_orphan_scheduler_processes,
    find_stale_ipc_files,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_process_table(rows: list[tuple[int, int, str]]) -> str:
    return "\n".join(f"{pid} {ppid} {cmd}" for pid, ppid, cmd in rows)


def _fake_run_factory(stdout: str):
    def _fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")

    return _fake_run


# ---------------------------------------------------------------------------
# find_orphan_scheduler_processes
# ---------------------------------------------------------------------------


class TestFindOrphanSchedulerProcesses:
    def test_detects_ppid_one_as_orphan(self, monkeypatch) -> None:
        table = _fake_process_table(
            [
                (1, 0, "init"),
                (999, 1, "python -m flash_aurora.scheduler --command-addr ipc:///tmp/a/w.ipc"),
            ]
        )
        monkeypatch.setattr(subprocess, "run", _fake_run_factory(table))
        monkeypatch.setattr(os, "getpid", lambda: 1000)

        orphans = find_orphan_scheduler_processes()

        assert [o.pid for o in orphans] == [999]

    def test_detects_dead_parent_as_orphan(self, monkeypatch) -> None:
        # PID 888 claims PPID 700, but 700 does not appear in the table.
        table = _fake_process_table(
            [
                (1, 0, "init"),
                (888, 700, "python -m flash_aurora.scheduler --command-addr ipc:///tmp/b/w.ipc"),
            ]
        )
        monkeypatch.setattr(subprocess, "run", _fake_run_factory(table))
        monkeypatch.setattr(os, "getpid", lambda: 1000)

        orphans = find_orphan_scheduler_processes()

        assert [o.pid for o in orphans] == [888]

    def test_ignores_healthy_process_with_live_parent(self, monkeypatch) -> None:
        table = _fake_process_table(
            [
                (1, 0, "init"),
                (500, 1, "jupyter-notebook"),
                (501, 500, "python -m flash_aurora.scheduler --command-addr ipc:///tmp/c/w.ipc"),
            ]
        )
        monkeypatch.setattr(subprocess, "run", _fake_run_factory(table))
        monkeypatch.setattr(os, "getpid", lambda: 1000)

        orphans = find_orphan_scheduler_processes()

        assert orphans == []

    def test_ignores_own_pid(self, monkeypatch) -> None:
        own_pid = 999
        table = _fake_process_table(
            [
                (1, 0, "init"),
                (own_pid, 1, "python -m flash_aurora.scheduler.supervisor"),
            ]
        )
        monkeypatch.setattr(subprocess, "run", _fake_run_factory(table))
        monkeypatch.setattr(os, "getpid", lambda: own_pid)

        orphans = find_orphan_scheduler_processes()

        assert orphans == []

    def test_ignores_non_scheduler_processes(self, monkeypatch) -> None:
        table = _fake_process_table(
            [
                (1, 0, "init"),
                (600, 1, "python train.py flash_aurora"),
            ]
        )
        monkeypatch.setattr(subprocess, "run", _fake_run_factory(table))
        monkeypatch.setattr(os, "getpid", lambda: 1000)

        orphans = find_orphan_scheduler_processes()

        assert orphans == []

    def test_matches_coordinator_shim(self, monkeypatch) -> None:
        table = _fake_process_table(
            [
                (1, 0, "init"),
                (701, 1, "python -m flash_aurora.scheduler.coordinator --worker-addrs ..."),
            ]
        )
        monkeypatch.setattr(subprocess, "run", _fake_run_factory(table))
        monkeypatch.setattr(os, "getpid", lambda: 1000)

        orphans = find_orphan_scheduler_processes()

        assert [o.pid for o in orphans] == [701]


# ---------------------------------------------------------------------------
# find_stale_ipc_files
# ---------------------------------------------------------------------------


class TestFindStaleIpcFiles:
    def test_returns_ipc_files_not_referenced_by_any_live_process(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        session_dir = tmp_path / "aurora_scheduler_abc"
        session_dir.mkdir()
        stale_ipc = session_dir / "worker.ipc"
        stale_ipc.write_text("")

        live_ipc = session_dir / "live.ipc"
        live_ipc.write_text("")

        table = _fake_process_table(
            [
                (1, 0, "init"),
                (200, 1, f"python -m flash_aurora.scheduler --command-addr ipc://{live_ipc}"),
            ]
        )
        monkeypatch.setattr(subprocess, "run", _fake_run_factory(table))
        monkeypatch.setattr(os, "getpid", lambda: 1000)

        stale = find_stale_ipc_files(tmp_path)

        assert stale_ipc in stale
        assert live_ipc not in stale

    def test_returns_empty_when_no_ipc_dirs_present(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(subprocess, "run", _fake_run_factory(""))
        monkeypatch.setattr(os, "getpid", lambda: 1000)

        stale = find_stale_ipc_files(tmp_path)

        assert stale == []


# ---------------------------------------------------------------------------
# SupervisorReport
# ---------------------------------------------------------------------------


class TestSupervisorReport:
    def test_gpu_memory_freed_computes_positive_delta(self) -> None:
        from flash_aurora.engine.runtime.resource_monitor import GpuResourceSample

        before = GpuResourceSample(
            index=0, name="Test GPU", utilization_percent=50.0,
            memory_used_mib=20000.0, memory_total_mib=24000.0,
        )
        after = GpuResourceSample(
            index=0, name="Test GPU", utilization_percent=10.0,
            memory_used_mib=400.0, memory_total_mib=24000.0,
        )
        report = SupervisorReport(gpu_before={0: before}, gpu_after={0: after})

        freed = report.gpu_memory_freed_mib()

        assert freed[0] == pytest.approx(19600.0)

    def test_gpu_memory_freed_ignores_increase(self) -> None:
        from flash_aurora.engine.runtime.resource_monitor import GpuResourceSample

        before = GpuResourceSample(
            index=0, name="GPU", utilization_percent=None,
            memory_used_mib=100.0, memory_total_mib=24000.0,
        )
        after = GpuResourceSample(
            index=0, name="GPU", utilization_percent=None,
            memory_used_mib=200.0, memory_total_mib=24000.0,
        )
        report = SupervisorReport(gpu_before={0: before}, gpu_after={0: after})

        freed = report.gpu_memory_freed_mib()

        assert 0 not in freed

    def test_summary_no_orphans(self) -> None:
        report = SupervisorReport()
        assert "no orphaned" in report.summary()


# ---------------------------------------------------------------------------
# SchedulerSupervisor.scan
# ---------------------------------------------------------------------------


class TestSchedulerSupervisorScan:
    def _make_supervisor(self, *, dry_run: bool = False, tmp_path: Path) -> SchedulerSupervisor:
        return SchedulerSupervisor(dry_run=dry_run, ipc_scan_root=tmp_path)

    def test_scan_kills_orphan_with_term_then_kill(self, monkeypatch, tmp_path: Path) -> None:
        orphan = OrphanProcess(pid=333, ppid=1, command="python -m flash_aurora.scheduler")
        monkeypatch.setattr(
            "flash_aurora.scheduler.supervisor.find_orphan_scheduler_processes",
            lambda: [orphan],
        )
        monkeypatch.setattr(
            "flash_aurora.scheduler.supervisor.find_stale_ipc_files",
            lambda _root: [],
        )
        monkeypatch.setattr(
            "flash_aurora.scheduler.supervisor.query_gpu_status",
            lambda: {},
        )

        kill_log: list[tuple[int, int]] = []
        alive = {333}

        def fake_killpg(pgid: int, sig: int) -> None:
            kill_log.append((pgid, sig))
            if sig == signal.SIGTERM:
                alive.discard(pgid)

        def fake_kill_check(pid: int, sig: int) -> None:
            if sig == 0 and pid not in alive:
                raise ProcessLookupError

        monkeypatch.setattr(os, "killpg", fake_killpg)
        monkeypatch.setattr(os, "kill", fake_kill_check)
        monkeypatch.setattr("time.sleep", lambda _: None)
        monkeypatch.setattr("time.monotonic", lambda: 0.0)

        supervisor = self._make_supervisor(tmp_path=tmp_path)
        report = supervisor.scan()

        assert report.orphans_killed == [333]
        assert any(sig == signal.SIGTERM for _, sig in kill_log)
        assert not any(sig == signal.SIGKILL for _, sig in kill_log)

    def test_scan_dry_run_does_not_kill(self, monkeypatch, tmp_path: Path) -> None:
        orphan = OrphanProcess(pid=444, ppid=1, command="python -m flash_aurora.scheduler")
        monkeypatch.setattr(
            "flash_aurora.scheduler.supervisor.find_orphan_scheduler_processes",
            lambda: [orphan],
        )
        monkeypatch.setattr(
            "flash_aurora.scheduler.supervisor.find_stale_ipc_files",
            lambda _root: [],
        )
        monkeypatch.setattr(
            "flash_aurora.scheduler.supervisor.query_gpu_status",
            lambda: {},
        )

        kill_called = []
        monkeypatch.setattr(os, "killpg", lambda *a: kill_called.append(a))
        monkeypatch.setattr(os, "kill", lambda *a: kill_called.append(a))

        supervisor = self._make_supervisor(dry_run=True, tmp_path=tmp_path)
        report = supervisor.scan()

        assert report.orphans_found == [orphan]
        assert report.orphans_killed == []
        assert kill_called == []

    def test_scan_removes_stale_ipc_files(self, monkeypatch, tmp_path: Path) -> None:
        session_dir = tmp_path / "aurora_scheduler_xyz"
        session_dir.mkdir()
        stale = session_dir / "w0.ipc"
        stale.write_text("")

        monkeypatch.setattr(
            "flash_aurora.scheduler.supervisor.find_orphan_scheduler_processes",
            lambda: [],
        )
        monkeypatch.setattr(
            "flash_aurora.scheduler.supervisor.find_stale_ipc_files",
            lambda _root: [stale],
        )
        monkeypatch.setattr(
            "flash_aurora.scheduler.supervisor.query_gpu_status",
            lambda: {},
        )

        supervisor = self._make_supervisor(tmp_path=tmp_path)
        report = supervisor.scan()

        assert stale in report.ipc_files_removed
        assert not stale.exists()

    def test_scan_no_orphans_returns_empty_report(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setattr(
            "flash_aurora.scheduler.supervisor.find_orphan_scheduler_processes",
            lambda: [],
        )
        monkeypatch.setattr(
            "flash_aurora.scheduler.supervisor.find_stale_ipc_files",
            lambda _root: [],
        )
        monkeypatch.setattr(
            "flash_aurora.scheduler.supervisor.query_gpu_status",
            lambda: {},
        )

        supervisor = self._make_supervisor(tmp_path=tmp_path)
        report = supervisor.scan()

        assert report.orphans_found == []
        assert report.orphans_killed == []
        assert report.ipc_files_removed == []
