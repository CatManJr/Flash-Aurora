"""Switching presets releases the loaded engine before a new one is built."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import zmq

from flash_aurora.scheduler.worker import ForecastWorker, ForecastWorkerConfig


@pytest.fixture
def context():
    zmq_context = zmq.Context()
    yield zmq_context
    zmq_context.term()

_PRESETS = ("era5_pretrained", "cams")


def test_preset_switch_releases_gpu_memory_and_builds_a_new_engine(
    tmp_path: Path, context: zmq.Context
) -> None:
    loaded = MagicMock()
    rebuilt = MagicMock()
    built: list[str] = []

    def build(preset: str) -> MagicMock:
        built.append(preset)
        return rebuilt

    worker = ForecastWorker(
        ForecastWorkerConfig(
            preset="era5_pretrained",
            presets=_PRESETS,
            asset_root=tmp_path,
            command_addr=f"ipc://{tmp_path / 'commands.ipc'}",
            event_addr=f"ipc://{tmp_path / 'events.ipc'}",
        ),
        engine=loaded,
        downloader=MagicMock(),
        context=context,
    )
    worker._build_engine = build  # type: ignore[method-assign]
    worker._set_model_ready(True)

    worker._prepare_engine_for("cams")

    loaded.release_gpu.assert_called_once_with(move_model_to_cpu=True)
    loaded.close.assert_called_once()
    assert built == ["cams"]
    rebuilt.load.assert_called_once()
    assert worker._loaded_preset == "cams"
    worker.close()
