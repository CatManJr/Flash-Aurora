"""System-wide supervisor that force-kills orphaned flash_aurora scheduler processes."""

from __future__ import annotations

import logging
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from flash_aurora.engine.runtime.resource_monitor import GpuResourceSample, query_gpu_status

logger = logging.getLogger(__name__)

_SCHEDULER_PATTERN = re.compile(r"flash_aurora\.scheduler(?:\.coordinator)?")
_TERM_WAIT_S = 5.0
_KILL_WAIT_S = 3.0


@dataclass(frozen=True)
class OrphanProcess:
    pid: int
    ppid: int
    command: str


@dataclass
class SupervisorReport:
    orphans_found: list[OrphanProcess] = field(default_factory=list)
    orphans_killed: list[int] = field(default_factory=list)
    ipc_files_removed: list[Path] = field(default_factory=list)
    gpu_before: dict[int, GpuResourceSample] = field(default_factory=dict)
    gpu_after: dict[int, GpuResourceSample] = field(default_factory=dict)

    def gpu_memory_freed_mib(self) -> dict[int, float]:
        freed: dict[int, float] = {}
        for index, before in self.gpu_before.items():
            after = self.gpu_after.get(index)
            if before.memory_used_mib is None or after is None or after.memory_used_mib is None:
                continue
            delta = before.memory_used_mib - after.memory_used_mib
            if delta > 0:
                freed[index] = delta
        return freed

    def summary(self) -> str:
        parts: list[str] = []
        if not self.orphans_found:
            parts.append("no orphaned scheduler processes found")
        else:
            parts.append(f"orphans found: {len(self.orphans_found)}")
            parts.append(f"orphans killed: {len(self.orphans_killed)}")
        if self.ipc_files_removed:
            parts.append(f"IPC files removed: {len(self.ipc_files_removed)}")
        freed = self.gpu_memory_freed_mib()
        if freed:
            gpu_parts = ", ".join(f"GPU{i}: {mib:.0f} MiB" for i, mib in sorted(freed.items()))
            parts.append(f"GPU memory freed: {gpu_parts}")
        return " | ".join(parts)


def _read_process_table() -> list[tuple[int, int, str]]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,cmd="],
        check=False,
        capture_output=True,
        text=True,
    )
    rows: list[tuple[int, int, str]] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        pid_text, ppid_text, command = parts
        if pid_text.isdigit() and ppid_text.isdigit():
            rows.append((int(pid_text), int(ppid_text), command))
    return rows


def find_orphan_scheduler_processes() -> list[OrphanProcess]:
    """Return scheduler processes whose parent is dead or re-parented to init."""
    table = _read_process_table()
    live_pids = {pid for pid, _, _ in table}
    own_pid = os.getpid()
    orphans: list[OrphanProcess] = []
    for pid, ppid, command in table:
        if pid == own_pid:
            continue
        if not _SCHEDULER_PATTERN.search(command):
            continue
        # ppid == 1 means init adopted the process after its parent died
        if ppid == 1 or ppid not in live_pids:
            orphans.append(OrphanProcess(pid=pid, ppid=ppid, command=command))
    return orphans


def find_stale_ipc_files(scan_root: Path = Path("/tmp")) -> list[Path]:
    """Return Aurora IPC socket files not referenced by any live scheduler process."""
    table = _read_process_table()
    live_commands = " ".join(cmd for _, _, cmd in table)
    stale: list[Path] = []
    for ipc_path in scan_root.glob("aurora_scheduler_*/*.ipc"):
        if str(ipc_path) not in live_commands:
            stale.append(ipc_path)
    return sorted(stale)


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def _force_kill(pid: int) -> None:
    """SIGTERM the process group; escalate to SIGKILL if it survives."""
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return

    deadline = time.monotonic() + _TERM_WAIT_S
    while time.monotonic() < deadline:
        if not _process_alive(pid):
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


class SchedulerSupervisor:
    """Force-reclaim layer for orphaned flash_aurora scheduler processes.

    Operates independently of any scheduler configuration by reading the OS
    process table directly. Does not require running ZMQ sockets or model state.
    """

    def __init__(
        self,
        *,
        dry_run: bool = False,
        ipc_scan_root: Path = Path("/tmp"),
        log: logging.Logger | None = None,
    ) -> None:
        self._dry_run = dry_run
        self._ipc_scan_root = ipc_scan_root
        self._log = log or logger

    def scan(self) -> SupervisorReport:
        """One scan-and-reclaim cycle."""
        report = SupervisorReport()
        report.gpu_before = query_gpu_status()

        for orphan in find_orphan_scheduler_processes():
            report.orphans_found.append(orphan)
            self._log.info("orphan pid=%d ppid=%d cmd=%s", orphan.pid, orphan.ppid, orphan.command)
            if not self._dry_run:
                _force_kill(orphan.pid)
                report.orphans_killed.append(orphan.pid)

        for ipc_path in find_stale_ipc_files(self._ipc_scan_root):
            self._log.info("stale IPC: %s", ipc_path)
            if not self._dry_run:
                try:
                    ipc_path.unlink()
                    report.ipc_files_removed.append(ipc_path)
                except OSError as exc:
                    self._log.warning("could not remove %s: %s", ipc_path, exc)

        if not self._dry_run and report.orphans_found:
            # Give the GPU driver time to reclaim VRAM before sampling.
            time.sleep(2.0)

        report.gpu_after = query_gpu_status()
        self._log.info("scan complete: %s", report.summary())
        return report

    def run_daemon(self, interval_s: float = 60.0) -> None:
        """Scan repeatedly until SIGTERM or SIGINT."""
        self._log.info("supervisor started (interval=%.0fs dry_run=%s)", interval_s, self._dry_run)
        stop_requested = False

        def _on_signal(signum: int, _frame: object) -> None:
            nonlocal stop_requested
            self._log.info("signal %d received, stopping after current scan", signum)
            stop_requested = True

        signal.signal(signal.SIGTERM, _on_signal)
        signal.signal(signal.SIGINT, _on_signal)

        while not stop_requested:
            try:
                self.scan()
            except Exception:
                self._log.exception("scan raised an unexpected exception")
            if not stop_requested:
                time.sleep(interval_s)

        self._log.info("supervisor stopped")


def _build_parser() -> "argparse.ArgumentParser":
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m flash_aurora.scheduler.supervisor",
        description="Find and force-kill orphaned flash_aurora scheduler processes.",
    )
    parser.add_argument("--daemon", action="store_true", help="Run continuously.")
    parser.add_argument("--interval", type=float, default=60.0, metavar="SECONDS")
    parser.add_argument("--dry-run", action="store_true", help="Report only, do not kill.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        format="%(asctime)s [supervisor] %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        level=getattr(logging, args.log_level),
    )
    supervisor = SchedulerSupervisor(dry_run=args.dry_run)
    if args.daemon:
        supervisor.run_daemon(interval_s=args.interval)
    else:
        report = supervisor.scan()
        if report.orphans_found and not args.dry_run and not report.orphans_killed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
