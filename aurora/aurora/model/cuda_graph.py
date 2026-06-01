"""CUDA graph helpers for fixed-shape inference."""

from __future__ import annotations

import contextlib
from typing import Any

import torch


class CudaGraphSwin3DBlockRunner:
    """Replay one fixed-shape Swin3D block inference call with a CUDA graph.

    The graph is valid only for the input shapes, dtypes, device, ``patch_res``,
    ``rollout_step`` and ``warped`` value used during construction. Returned
    tensors alias the graph's static output buffer and are overwritten on each
    replay.
    """

    def __init__(
        self,
        block: Any,
        x: torch.Tensor,
        c: torch.Tensor,
        patch_res: tuple[int, int, int],
        *,
        rollout_step: int,
        warped: bool,
        autocast: bool = False,
        warmup_iters: int = 3,
    ) -> None:
        if not x.is_cuda or not c.is_cuda:
            raise ValueError("CudaGraphSwin3DBlockRunner requires CUDA tensors.")

        self.block = block
        self.static_x = torch.empty_like(x)
        self.static_c = torch.empty_like(c)
        self.patch_res = patch_res
        self.rollout_step = rollout_step
        self.warped = warped
        self.autocast = autocast
        self.graph = torch.cuda.CUDAGraph()

        self.static_x.copy_(x)
        self.static_c.copy_(c)
        torch.cuda.synchronize()

        if warmup_iters > 0:
            side_stream = torch.cuda.Stream()
            side_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side_stream):
                for _ in range(warmup_iters):
                    self._forward_static()
            torch.cuda.current_stream().wait_stream(side_stream)
            torch.cuda.synchronize()

        with torch.cuda.graph(self.graph):
            self.static_out = self._forward_static()

    def _forward_static(self) -> torch.Tensor:
        ctx: Any
        if self.autocast:
            ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            with torch.inference_mode():
                return self.block(
                    self.static_x,
                    self.static_c,
                    self.patch_res,
                    rollout_step=self.rollout_step,
                    warped=self.warped,
                )

    def __call__(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if x.shape != self.static_x.shape or c.shape != self.static_c.shape:
            raise ValueError("CUDA graph replay inputs must match captured shapes.")
        if x.dtype != self.static_x.dtype or c.dtype != self.static_c.dtype:
            raise ValueError("CUDA graph replay inputs must match captured dtypes.")
        if x.device != self.static_x.device or c.device != self.static_c.device:
            raise ValueError("CUDA graph replay inputs must match captured devices.")

        self.static_x.copy_(x)
        self.static_c.copy_(c)
        self.graph.replay()
        return self.static_out
