"""Engine.prepare keeps a single-device model resident across jobs."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from flash_aurora.engine.core.engine import AuroraEngine


class _FakeModel:
    def cpu(self) -> None:
        pass


class _Harness:
    """An engine whose model loading, IC building and warmup are counted instead of run."""

    def __init__(self, engine: AuroraEngine, monkeypatch: pytest.MonkeyPatch) -> None:
        self.engine = engine
        self.loaded_models: list[_FakeModel] = []
        self.warmup = MagicMock()
        monkeypatch.setattr(engine, "acquire_gpu", lambda *, rollout_steps=None: None)
        monkeypatch.setattr(engine, "_load_model_to_device", self._load_model_to_device)
        monkeypatch.setattr(engine, "_builder", lambda: self._ic_builder())
        monkeypatch.setattr(engine._graph_pool, "warmup", self.warmup)

    def _load_model_to_device(self, *, rollout_steps: int | None = None):
        del rollout_steps
        model = _FakeModel()
        self.loaded_models.append(model)
        return model, MagicMock()

    def _ic_builder(self) -> MagicMock:
        builder = MagicMock()
        builder.from_source.side_effect = lambda _request: MagicMock()
        return builder

    def prepare(self) -> None:
        self.engine.prepare(MagicMock(), rollout_steps=1)


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Harness:
    engine = AuroraEngine.from_preset("era5_pretrained", asset_root=tmp_path)
    return _Harness(engine, monkeypatch)


def test_second_prepare_reuses_the_resident_model(harness: _Harness) -> None:
    harness.prepare()
    resident = harness.engine.model

    harness.prepare()

    assert len(harness.loaded_models) == 1
    assert harness.engine.model is resident


def test_second_prepare_does_not_warm_up_again(harness: _Harness) -> None:
    harness.prepare()
    harness.prepare()

    assert harness.warmup.call_count == 1


def test_prepare_reloads_and_rewarms_after_the_model_left_the_gpu(harness: _Harness) -> None:
    harness.prepare()
    harness.engine.release_gpu(move_model_to_cpu=True)

    harness.prepare()

    assert len(harness.loaded_models) == 2
    assert harness.engine.model is harness.loaded_models[1]
    assert harness.warmup.call_count == 2


def test_prepare_replans_a_distributed_model_every_time(harness: _Harness) -> None:
    harness.engine.config.distributed = MagicMock()

    harness.prepare()
    harness.prepare()

    assert len(harness.loaded_models) == 2


def test_prepare_recovers_after_a_failed_reload(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness.prepare()
    harness.engine.release_gpu(move_model_to_cpu=True)
    working_load = harness.engine._load_model_to_device
    monkeypatch.setattr(
        harness.engine,
        "_load_model_to_device",
        MagicMock(side_effect=RuntimeError("checkpoint unreadable")),
    )
    with pytest.raises(RuntimeError, match="checkpoint unreadable"):
        harness.prepare()

    monkeypatch.setattr(harness.engine, "_load_model_to_device", working_load)
    harness.prepare()

    assert harness.engine.model is harness.loaded_models[-1]


def test_load_replaces_the_model_and_drops_its_warmup_state(harness: _Harness) -> None:
    harness.prepare()

    harness.engine.load()

    assert harness.engine.model is harness.loaded_models[1]
    assert harness.engine._forward_warmed is False
