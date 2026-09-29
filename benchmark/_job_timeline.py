"""Fold scheduler events into one record per job.

A job's life on the scheduler is ``submitted -> accepted -> preparing ->
running -> completed | failed``. The replayer stamps each event with its own
monotonic clock, since the wire protocol carries no timestamps, and feeds them
here.

Reading a record:
    wait_s     submitted to preparing: time spent queued at the coordinator
               and inside the worker, before any work on the job began
    prepare_s  preparing to running: IC build, plus model load when not resident
    rollout_s  running to finished: the autoregressive rollout and its export
    service_s  prepare_s + rollout_s: the time the job occupied its GPU
    latency_s  submitted to finished: what the requester experienced
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class JobStatus(Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    INCOMPLETE = "incomplete"


_TERMINAL_KINDS = {"completed": JobStatus.COMPLETED, "failed": JobStatus.FAILED}
_STAGE_KINDS = ("accepted", "preparing", "running")


@dataclass(frozen=True)
class TimedEvent:
    """A scheduler event plus the client-side time it was received."""

    t_s: float
    kind: str
    request_id: str
    worker_id: str | None = None
    worker_device: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class JobRecord:
    request_id: str
    preset: str
    submitted_s: float
    deadline_s: float | None  # absolute, on the same clock as the other stamps
    accepted_s: float | None
    preparing_s: float | None
    running_s: float | None
    finished_s: float | None
    status: JobStatus
    worker_id: str | None
    worker_device: str | None
    error: str | None

    @property
    def wait_s(self) -> float:
        return self._completed_stamp("preparing_s") - self.submitted_s

    @property
    def prepare_s(self) -> float:
        return self._completed_stamp("running_s") - self._completed_stamp("preparing_s")

    @property
    def rollout_s(self) -> float:
        return self._completed_stamp("finished_s") - self._completed_stamp("running_s")

    @property
    def service_s(self) -> float:
        return self._completed_stamp("finished_s") - self._completed_stamp("preparing_s")

    @property
    def latency_s(self) -> float:
        return self._completed_stamp("finished_s") - self.submitted_s

    @property
    def has_deadline(self) -> bool:
        return self.deadline_s is not None

    @property
    def met_deadline(self) -> bool:
        """True only for a completed job that finished by its deadline."""
        if self.deadline_s is None:
            raise ValueError(f"job {self.request_id!r} has no deadline")
        return self.status is JobStatus.COMPLETED and self._completed_stamp("finished_s") <= self.deadline_s

    def _completed_stamp(self, name: str) -> float:
        if self.status is not JobStatus.COMPLETED:
            raise ValueError(f"job {self.request_id!r} is {self.status.value}, not completed")
        stamp = getattr(self, name)
        if stamp is None:
            raise ValueError(f"completed job {self.request_id!r} never reported {name}")
        return stamp


@dataclass
class _JobState:
    preset: str
    submitted_s: float
    deadline_s: float | None
    stage_times_s: dict[str, float] = field(default_factory=dict)
    finished_s: float | None = None
    status: JobStatus = JobStatus.INCOMPLETE
    worker_id: str | None = None
    worker_device: str | None = None
    error: str | None = None

    @property
    def is_finished(self) -> bool:
        return self.status is not JobStatus.INCOMPLETE


class JobTimeline:
    """Accumulates submissions and events, then reports one JobRecord per job."""

    def __init__(self) -> None:
        self._jobs: dict[str, _JobState] = {}
        self._unknown_job_event_count = 0

    def record_submission(
        self,
        request_id: str,
        *,
        preset: str,
        t_s: float,
        deadline_after_s: float | None = None,
    ) -> None:
        if request_id in self._jobs:
            raise ValueError(f"job {request_id!r} was already submitted")
        deadline_s = None if deadline_after_s is None else t_s + deadline_after_s
        self._jobs[request_id] = _JobState(preset=preset, submitted_s=t_s, deadline_s=deadline_s)

    def record_event(self, event: TimedEvent) -> None:
        """Fold one event in. The first report of each stage wins; later ones are duplicates."""
        job = self._jobs.get(event.request_id)
        if job is None:
            self._unknown_job_event_count += 1
            return
        self._adopt_worker(job, event)
        if event.kind in _STAGE_KINDS:
            job.stage_times_s.setdefault(event.kind, event.t_s)
        elif event.kind in _TERMINAL_KINDS and not job.is_finished:
            self._finish(job, event)

    def records(self) -> tuple[JobRecord, ...]:
        """Every submitted job in submission order, finished or not."""
        return tuple(self._record_of(request_id, job) for request_id, job in self._jobs.items())

    @property
    def unfinished_count(self) -> int:
        return sum(1 for job in self._jobs.values() if not job.is_finished)

    @property
    def unknown_job_event_count(self) -> int:
        """Events naming a request this timeline never saw submitted, e.g. another client's."""
        return self._unknown_job_event_count

    @staticmethod
    def _adopt_worker(job: _JobState, event: TimedEvent) -> None:
        if job.worker_id is None and event.worker_id is not None:
            job.worker_id = event.worker_id
            job.worker_device = event.worker_device

    @staticmethod
    def _finish(job: _JobState, event: TimedEvent) -> None:
        job.finished_s = event.t_s
        job.status = _TERMINAL_KINDS[event.kind]
        job.error = event.error

    @staticmethod
    def _record_of(request_id: str, job: _JobState) -> JobRecord:
        return JobRecord(
            request_id=request_id,
            preset=job.preset,
            submitted_s=job.submitted_s,
            deadline_s=job.deadline_s,
            accepted_s=job.stage_times_s.get("accepted"),
            preparing_s=job.stage_times_s.get("preparing"),
            running_s=job.stage_times_s.get("running"),
            finished_s=job.finished_s,
            status=job.status,
            worker_id=job.worker_id,
            worker_device=job.worker_device,
            error=job.error,
        )
