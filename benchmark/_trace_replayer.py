"""Replay a job trace against a running coordinator and time every job.

The wire protocol carries no timestamps, so the replayer stamps each event with
its own monotonic clock at the moment the client receives it. Timestamps
therefore include one ZMQ hop per event, which is small against minute-scale
jobs but is not zero.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from _job_timeline import JobTimeline, TimedEvent
from _trace_generator import TraceJob
from flash_aurora.scheduler.client import ForecastClient
from flash_aurora.scheduler.protocol import ForecastEvent, ForecastOutputMode, ForecastRequest

_POLL_SLICE_MS = 50


@dataclass(frozen=True)
class CachedIngress:
    """One preset's already-cached analysis. The replayer must not download another."""

    valid_time: str
    cache_dir: str
    time_index: int = 1


class TraceReplayer:
    """Submit each job at its arrival time and record its lifecycle.

    ``output_mode`` defaults to ``metadata_only`` so a run measures scheduling and
    compute, not disk export. Pass ``export_paths`` with ``export_dir`` to include
    the NetCDF write in each job's rollout time.
    """

    def __init__(
        self,
        client: ForecastClient,
        *,
        output_mode: ForecastOutputMode = "metadata_only",
        export_dir: str | None = None,
        ingress_by_preset: Mapping[str, CachedIngress] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client = client
        self._output_mode = output_mode
        self._export_dir = export_dir
        self._ingress_by_preset = ingress_by_preset
        self._clock = clock

    def replay(self, trace: Sequence[TraceJob], *, timeout_s: float) -> JobTimeline:
        """Run the whole trace and return its timeline once every job has finished."""
        timeline = JobTimeline()
        pending = deque(sorted(trace, key=lambda job: job.arrival_s))
        start_s = self._clock()
        while pending or timeline.unfinished_count:
            elapsed_s = self._clock() - start_s
            if elapsed_s > timeout_s:
                raise TimeoutError(self._timeout_message(timeline, len(pending), timeout_s))
            self._submit_due_jobs(pending, timeline, start_s)
            self._record_next_event(timeline, start_s, wait_ms=self._wait_ms(pending, elapsed_s))
        return timeline

    def _submit_due_jobs(
        self,
        pending: deque[TraceJob],
        timeline: JobTimeline,
        start_s: float,
    ) -> None:
        while pending and pending[0].arrival_s <= self._clock() - start_s:
            job = pending.popleft()
            timeline.record_submission(
                job.job_id,
                preset=job.preset,
                t_s=self._clock() - start_s,
                deadline_after_s=job.deadline_s,
            )
            self._client.submit(self._request_for(job))

    def _record_next_event(self, timeline: JobTimeline, start_s: float, *, wait_ms: int) -> None:
        event = self._client.try_recv_event(timeout_ms=wait_ms)
        if event is None or event.request_id is None:
            return
        timeline.record_event(_timed(event, request_id=event.request_id, t_s=self._clock() - start_s))

    def _request_for(self, job: TraceJob) -> ForecastRequest:
        return forecast_request(
            job,
            output_mode=self._output_mode,
            export_dir=self._export_dir,
            ingress=self._ingress_for(job),
        )

    def _ingress_for(self, job: TraceJob) -> CachedIngress | None:
        if self._ingress_by_preset is None:
            return None
        try:
            return self._ingress_by_preset[job.preset]
        except KeyError as exc:
            raise KeyError(f"no cached ingress for preset {job.preset!r}") from exc

    @staticmethod
    def _wait_ms(pending: deque[TraceJob], elapsed_s: float) -> int:
        if not pending:
            return _POLL_SLICE_MS
        until_next_arrival_ms = int((pending[0].arrival_s - elapsed_s) * 1000) + 1
        return max(1, min(_POLL_SLICE_MS, until_next_arrival_ms))

    @staticmethod
    def _timeout_message(timeline: JobTimeline, unsubmitted: int, timeout_s: float) -> str:
        return (
            f"trace replay exceeded {timeout_s:.0f}s with {timeline.unfinished_count} jobs "
            f"unfinished and {unsubmitted} not yet submitted"
        )


def forecast_request(
    job: TraceJob,
    *,
    output_mode: ForecastOutputMode,
    export_dir: str | None,
    ingress: CachedIngress | None,
) -> ForecastRequest:
    """Build the wire request for one trace job.

    The trace ``valid_time`` is the operational label. When ``ingress`` is set, the
    tensor comes from that local cache instead, so the run does not fetch a new
    initial condition.
    """
    if ingress is None:
        return ForecastRequest(
            request_id=job.job_id,
            preset=job.preset,
            steps=job.steps,
            valid_time=job.valid_time,
            output_mode=output_mode,
            export_dir=export_dir,
        )
    return ForecastRequest(
        request_id=job.job_id,
        preset=job.preset,
        steps=job.steps,
        valid_time=ingress.valid_time,
        cache_dir=ingress.cache_dir,
        time_index=ingress.time_index,
        download=False,
        output_mode=output_mode,
        export_dir=export_dir,
    )


def _timed(event: ForecastEvent, *, request_id: str, t_s: float) -> TimedEvent:
    return TimedEvent(
        t_s=t_s,
        kind=event.kind,
        request_id=request_id,
        worker_id=event.worker_id,
        worker_device=event.worker_device,
        error=event.error,
    )
