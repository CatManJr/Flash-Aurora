from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from flash_aurora.engine.runtime.cute_jit import model_uses_cute_window_attn
from flash_aurora.engine.runtime.graph_pool import GraphPool


def test_model_uses_cute_from_inference_config() -> None:
    model = MagicMock()
    model.inference_config = MagicMock(use_cute_window_attn=True)
    assert model_uses_cute_window_attn(model) is True


def test_model_uses_cute_false_for_baseline() -> None:
    model = MagicMock()
    model.inference_config = MagicMock(use_cute_window_attn=False)
    assert model_uses_cute_window_attn(model) is False


def test_graph_pool_skips_warmup_without_cute_or_cuda_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = GraphPool()
    model = MagicMock()
    model.inference_config = MagicMock(use_cute_window_attn=False, cuda_graph_scope="off")
    config = MagicMock(device="cpu", forward_warmup_iters=2, cuda_graph=False, distributed=None)
    warmup = MagicMock()
    monkeypatch.setattr("flash_aurora.engine.runtime.graph_pool.warmup_forwards", warmup)
    pool.warmup(model, MagicMock(), config)
    warmup.assert_not_called()


def test_graph_pool_warmup_for_cute_precision(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = GraphPool()
    model = MagicMock()
    model.inference_config = MagicMock(use_cute_window_attn=True, cuda_graph_scope="off")
    config = MagicMock(device="cpu", forward_warmup_iters=2, cuda_graph=False, distributed=None)
    monkeypatch.setattr(
        "flash_aurora.engine.runtime.graph_pool.prepare_cute_dsl_runtime",
        MagicMock(),
    )
    warmup = MagicMock(return_value=MagicMock())
    monkeypatch.setattr("flash_aurora.engine.runtime.graph_pool.warmup_forwards", warmup)
    monkeypatch.setattr(
        "flash_aurora.engine.runtime.graph_pool.prepare_rollout_batch",
        MagicMock(return_value=MagicMock()),
    )
    pool.warmup(model, MagicMock(), config)
    warmup.assert_called_once()
