from __future__ import annotations

import time

import pytest

from flash_aurora.engine.core.prepare import overlap_ic_and_load, serial_ic_then_load


def test_overlap_ic_and_load_runs_faster_than_serial() -> None:
    def slow_ic() -> str:
        time.sleep(0.08)
        return "batch"

    def slow_load() -> tuple[str, object]:
        time.sleep(0.05)
        from flash_aurora.engine.core.prepare import LoadTiming

        return "model", LoadTiming(50.0, 10.0, 5.0)

    _, _, overlap_timing = overlap_ic_and_load(slow_ic, slow_load)
    _, _, serial_timing = serial_ic_then_load(slow_ic, slow_load)

    assert overlap_timing.prepare_wall_ms < serial_timing.prepare_wall_ms
    assert overlap_timing.build_ic_ms >= 70.0
    assert overlap_timing.build_model_ms >= 40.0


def test_overlap_ic_and_load_propagates_ic_errors() -> None:
    def failing_ic() -> str:
        raise ValueError("ic failed")

    def load() -> tuple[str, object]:
        from flash_aurora.engine.core.prepare import LoadTiming

        return "model", LoadTiming(1.0, 1.0, 1.0)

    with pytest.raises(ValueError, match="ic failed"):
        overlap_ic_and_load(failing_ic, load)
