"""Folding scheduler events into one record per job."""

from __future__ import annotations

import pytest

from _job_timeline import JobStatus, JobTimeline, TimedEvent


def _event(t_s: float, kind: str, request_id: str = "job", **fields) -> TimedEvent:
    return TimedEvent(t_s=t_s, kind=kind, request_id=request_id, **fields)


def _timeline_with(*events: TimedEvent, deadline_after_s: float | None = None) -> JobTimeline:
    timeline = JobTimeline()
    timeline.record_submission("job", preset="era5_pretrained", t_s=0.0, deadline_after_s=deadline_after_s)
    for event in events:
        timeline.record_event(event)
    return timeline


def _lifecycle(finished_s: float = 6.0, *, worker_id: str = "gpu-1") -> list[TimedEvent]:
    return [
        _event(0.1, "accepted", worker_id=worker_id, worker_device="cuda:1"),
        _event(2.0, "preparing", worker_id=worker_id, worker_device="cuda:1"),
        _event(5.0, "running", worker_id=worker_id, worker_device="cuda:1"),
        _event(finished_s, "completed", worker_id=worker_id, worker_device="cuda:1"),
    ]


def _only_record(timeline: JobTimeline):
    [record] = timeline.records()
    return record


def test_completed_job_splits_latency_into_wait_prepare_and_rollout() -> None:
    record = _only_record(_timeline_with(*_lifecycle()))

    assert record.status is JobStatus.COMPLETED
    assert (record.wait_s, record.prepare_s, record.rollout_s) == (2.0, 3.0, 1.0)


def test_service_time_is_the_gpu_occupancy_and_latency_is_what_the_requester_saw() -> None:
    record = _only_record(_timeline_with(*_lifecycle()))

    assert record.service_s == 4.0
    assert record.latency_s == 6.0


def test_job_is_attributed_to_the_worker_and_device_that_reported_it() -> None:
    record = _only_record(_timeline_with(*_lifecycle(worker_id="gpu-3")))

    assert (record.worker_id, record.worker_device) == ("gpu-3", "cuda:1")


def test_events_without_a_worker_leave_the_job_unattributed() -> None:
    record = _only_record(_timeline_with(_event(1.0, "preparing"), _event(2.0, "completed")))

    assert record.worker_id is None


def test_deadline_is_absolute_and_met_when_the_job_finishes_in_time() -> None:
    record = _only_record(_timeline_with(*_lifecycle(finished_s=6.0), deadline_after_s=10.0))

    assert record.deadline_s == 10.0
    assert record.met_deadline


def test_deadline_is_missed_when_the_job_finishes_late() -> None:
    record = _only_record(_timeline_with(*_lifecycle(finished_s=12.0), deadline_after_s=10.0))

    assert not record.met_deadline


def test_failed_job_records_its_error_and_misses_its_deadline() -> None:
    failed = _event(3.0, "failed", error="out of memory", worker_id="gpu-2")
    record = _only_record(_timeline_with(_event(1.0, "preparing"), failed, deadline_after_s=10.0))

    assert record.status is JobStatus.FAILED
    assert record.error == "out of memory"
    assert not record.met_deadline


def test_durations_of_a_failed_job_are_refused() -> None:
    record = _only_record(_timeline_with(_event(3.0, "failed", error="boom")))

    with pytest.raises(ValueError, match="not completed"):
        record.latency_s


def test_job_without_a_terminal_event_stays_incomplete() -> None:
    timeline = _timeline_with(_event(1.0, "accepted"), _event(2.0, "preparing"))

    assert _only_record(timeline).status is JobStatus.INCOMPLETE
    assert timeline.unfinished_count == 1


def test_finished_job_no_longer_counts_as_unfinished() -> None:
    assert _timeline_with(*_lifecycle()).unfinished_count == 0


def test_first_report_of_a_stage_wins() -> None:
    timeline = _timeline_with(_event(2.0, "preparing"), _event(9.0, "preparing"), _event(10.0, "completed"))

    assert _only_record(timeline).preparing_s == 2.0


def test_first_terminal_event_wins() -> None:
    timeline = _timeline_with(_event(4.0, "completed"), _event(5.0, "failed", error="late"))

    record = _only_record(timeline)
    assert record.status is JobStatus.COMPLETED
    assert record.finished_s == 4.0


def test_job_without_a_deadline_refuses_a_deadline_verdict() -> None:
    record = _only_record(_timeline_with(*_lifecycle()))

    with pytest.raises(ValueError, match="no deadline"):
        record.met_deadline


@pytest.mark.parametrize("kind", ["step", "health", "ready"])
def test_non_lifecycle_events_do_not_change_the_record(kind: str) -> None:
    before = _only_record(_timeline_with())

    after = _only_record(_timeline_with(_event(1.0, kind)))

    assert after == before


def test_events_for_unknown_requests_are_counted_not_folded() -> None:
    timeline = _timeline_with()

    timeline.record_event(_event(1.0, "completed", request_id="someone-else"))

    assert timeline.unknown_job_event_count == 1
    assert timeline.unfinished_count == 1


def test_resubmitting_a_job_is_rejected() -> None:
    timeline = _timeline_with()

    with pytest.raises(ValueError, match="already submitted"):
        timeline.record_submission("job", preset="era5_pretrained", t_s=1.0)


def test_records_come_back_in_submission_order() -> None:
    timeline = JobTimeline()
    for request_id in ("c", "a", "b"):
        timeline.record_submission(request_id, preset="cams", t_s=0.0)

    assert [record.request_id for record in timeline.records()] == ["c", "a", "b"]
