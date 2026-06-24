from __future__ import annotations

import torch
from flash_aurora.aurora import Batch
from flash_aurora.aurora.model.aurora import Aurora

from flash_aurora.engine.core.config import EngineConfig
from flash_aurora.engine.runtime.rollout_prep import prepare_rollout_batch, warmup_forwards


class GraphPool:
    """Forward warmup and optional CUDA graph capture for fixed-shape inference."""

    def __init__(self) -> None:
        self._captured: dict[str, object] = {}

    def clear(self) -> None:
        self._captured.clear()

    def warmup(
        self,
        model: Aurora,
        batch: Batch,
        config: EngineConfig,
        *,
        forward_iters: int | None = None,
    ) -> None:
        iters = config.forward_warmup_iters if forward_iters is None else forward_iters
        if iters <= 0:
            return

        device = torch.device(config.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")

        prepared = prepare_rollout_batch(model, batch)
        prepared = warmup_forwards(model, prepared, iters=iters, device=device)

        if not config.cuda_graph or device.type != "cuda":
            return
        if model.inference_config is None or model.inference_config.cuda_graph_scope == "off":
            return

        model.capture_inference_cuda_graph(
            prepared,
            warmup_iters=max(1, min(iters, 2)),
        )
