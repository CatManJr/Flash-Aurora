"""Copyright (c) Catman Jr. Licensed under the MIT license.

CUDA graph helpers for fixed-shape inference."""

from __future__ import annotations

import contextlib
from typing import Any

import torch

from flash_aurora.aurora.batch import Batch


def _batch_tensor_shapes(batch: Batch) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for group in ("surf_vars", "static_vars", "atmos_vars"):
        for name, tensor in getattr(batch, group).items():
            shapes[f"{group}.{name}"] = tuple(tensor.shape)
    return shapes


def _allocate_batch_like(batch: Batch) -> Batch:
    return Batch(
        surf_vars={k: torch.empty_like(v) for k, v in batch.surf_vars.items()},
        static_vars={k: torch.empty_like(v) for k, v in batch.static_vars.items()},
        atmos_vars={k: torch.empty_like(v) for k, v in batch.atmos_vars.items()},
        metadata=batch.metadata,
    )


def _copy_batch(src: Batch, dst: Batch) -> None:
    for group in ("surf_vars", "static_vars", "atmos_vars"):
        src_group = getattr(src, group)
        dst_group = getattr(dst, group)
        if src_group.keys() != dst_group.keys():
            raise ValueError("Batch keys must match for CUDA graph replay.")
        for name, src_tensor in src_group.items():
            dst_tensor = dst_group[name]
            if src_tensor.shape != dst_tensor.shape:
                raise ValueError(
                    f"Shape mismatch for {group}.{name}: {tuple(src_tensor.shape)} "
                    f"vs {tuple(dst_tensor.shape)}"
                )
            if src_tensor.dtype != dst_tensor.dtype:
                raise ValueError(f"Dtype mismatch for {group}.{name}.")
            if src_tensor.device != dst_tensor.device:
                raise ValueError(f"Device mismatch for {group}.{name}.")
            dst_tensor.copy_(src_tensor)


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


class CudaGraphAuroraBackboneRunner:
    """Replay a fixed-shape Swin3D backbone call with a CUDA graph."""

    def __init__(
        self,
        backbone: Any,
        x: torch.Tensor,
        patch_res: tuple[int, int, int],
        *,
        lead_time: Any,
        rollout_step: int,
        autocast: bool = False,
        backbone_compute_dtype: torch.dtype | None = None,
        backbone_matmul_bf16: bool = False,
        backbone_matmul_tf32: bool = False,
        warmup_iters: int = 3,
    ) -> None:
        if not x.is_cuda:
            raise ValueError("CudaGraphAuroraBackboneRunner requires CUDA tensors.")

        self.backbone = backbone
        self.static_x = torch.empty_like(x)
        self.patch_res = patch_res
        self.lead_time = lead_time
        self.rollout_step = rollout_step
        self.autocast = autocast
        self.backbone_compute_dtype = backbone_compute_dtype
        self.backbone_matmul_bf16 = backbone_matmul_bf16
        self.backbone_matmul_tf32 = backbone_matmul_tf32
        self.graph = torch.cuda.CUDAGraph()

        self.static_x.copy_(x)
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
        from flash_aurora.aurora.model.custom_op_paths import run_backbone_with_dtype_routing

        with torch.inference_mode():
            return run_backbone_with_dtype_routing(
                self.backbone,
                self.static_x,
                autocast=self.autocast,
                backbone_compute_dtype=self.backbone_compute_dtype,
                backbone_matmul_bf16=self.backbone_matmul_bf16,
                backbone_matmul_tf32=self.backbone_matmul_tf32,
                lead_time=self.lead_time,
                patch_res=self.patch_res,
                rollout_step=self.rollout_step,
            )

    def can_replay(self, x: torch.Tensor, rollout_step: int) -> bool:
        # Backbone capture fixes ``rollout_step`` in the graph. Replay is valid across rollout
        # steps when LoRA is step-invariant (``lora_mode='single'``) and AdaLN does not use step.
        del rollout_step
        return (
            x.shape == self.static_x.shape
            and x.dtype == self.static_x.dtype
            and x.device == self.static_x.device
        )

    def __call__(self, x: torch.Tensor, *, rollout_step: int) -> torch.Tensor:
        if not self.can_replay(x, rollout_step):
            raise ValueError("CUDA graph replay inputs must match captured shapes/dtypes/step.")
        self.static_x.copy_(x)
        self.graph.replay()
        return self.static_out


class CudaGraphAuroraGpuRunner:
    """Replay encoder -> backbone -> decoder on fixed-shape batches.

    Perceiver modules remain standard PyTorch (SDPA); this
    only removes launch overhead by capturing the GPU stack in one graph.
    """

    def __init__(
        self,
        model: Any,
        encoder_batch: Batch,
        patch_res: tuple[int, int, int],
        *,
        rollout_step: int,
        autocast_backbone: bool = False,
        autocast_encoder_decoder: bool = False,
        encoder_decoder_use_tensor_core: bool = False,
        warmup_iters: int = 2,
    ) -> None:
        if not next(iter(encoder_batch.surf_vars.values())).is_cuda:
            raise ValueError("CudaGraphAuroraGpuRunner requires CUDA batches.")

        self.model = model
        self.static_batch = _allocate_batch_like(encoder_batch)
        _copy_batch(encoder_batch, self.static_batch)
        self.patch_res = patch_res
        self.rollout_step = rollout_step
        self.autocast_backbone = autocast_backbone
        self.autocast_encoder_decoder = autocast_encoder_decoder
        self.encoder_decoder_use_tensor_core = encoder_decoder_use_tensor_core
        self.lead_time = model.timestep
        self.graph = torch.cuda.CUDAGraph()
        self._captured_shapes = _batch_tensor_shapes(encoder_batch)

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
            self.static_pred = self._forward_static()

    def _forward_static(self) -> Batch:
        from flash_aurora.aurora.model.custom_op_paths import run_with_encoder_decoder_routing

        with torch.inference_mode():
            x = run_with_encoder_decoder_routing(
                self.model.encoder,
                self.static_batch,
                autocast_bf16=self.autocast_encoder_decoder,
                use_tensor_core=self.encoder_decoder_use_tensor_core,
                lead_time=self.lead_time,
            )
            x = self.model._run_backbone(
                x,
                lead_time=self.lead_time,
                patch_res=self.patch_res,
                rollout_step=self.rollout_step,
            )
            return run_with_encoder_decoder_routing(
                self.model.decoder,
                x,
                self.static_batch,
                autocast_bf16=self.autocast_encoder_decoder,
                use_tensor_core=self.encoder_decoder_use_tensor_core,
                patch_res=self.patch_res,
                lead_time=self.lead_time,
            )

    def can_replay(self, encoder_batch: Batch, rollout_step: int) -> bool:
        if rollout_step != self.rollout_step:
            return False
        return _batch_tensor_shapes(encoder_batch) == self._captured_shapes

    def __call__(self, encoder_batch: Batch, *, rollout_step: int) -> Batch:
        if not self.can_replay(encoder_batch, rollout_step):
            raise ValueError("CUDA graph replay batch must match captured shapes and rollout step.")
        _copy_batch(encoder_batch, self.static_batch)
        self.graph.replay()
        return self.static_pred


def build_aurora_cuda_graph_runner(
    model: Any,
    *,
    scope: str,
    encoder_batch: Batch,
    backbone_input: torch.Tensor | None,
    patch_res: tuple[int, int, int],
    rollout_step: int,
    autocast_backbone: bool,
    autocast_encoder_decoder: bool = False,
    encoder_decoder_use_tensor_core: bool = False,
    warmup_iters: int = 2,
) -> CudaGraphAuroraBackboneRunner | CudaGraphAuroraGpuRunner:
    if scope == "backbone":
        if backbone_input is None:
            raise ValueError("backbone_input is required for scope='backbone'.")
        inf = getattr(model, "inference_config", None)
        backbone_compute_dtype = None
        backbone_matmul_bf16 = False
        backbone_matmul_tf32 = False
        if inf is not None and not autocast_backbone:
            if (
                inf.backbone_compute_dtype == "bfloat16"
                and not inf.backbone_matmul_bf16
                and not inf.backbone_matmul_tf32
            ):
                backbone_compute_dtype = model.cute_window_attn_dtype
            backbone_matmul_bf16 = inf.backbone_matmul_bf16
            backbone_matmul_tf32 = inf.backbone_matmul_tf32
        return CudaGraphAuroraBackboneRunner(
            model.backbone,
            backbone_input,
            patch_res,
            lead_time=model.timestep,
            rollout_step=rollout_step,
            autocast=autocast_backbone,
            backbone_compute_dtype=backbone_compute_dtype,
            backbone_matmul_bf16=backbone_matmul_bf16,
            backbone_matmul_tf32=backbone_matmul_tf32,
            warmup_iters=warmup_iters,
        )
    if scope == "full_gpu":
        return CudaGraphAuroraGpuRunner(
            model,
            encoder_batch,
            patch_res,
            rollout_step=rollout_step,
            autocast_backbone=autocast_backbone,
            autocast_encoder_decoder=autocast_encoder_decoder,
            encoder_decoder_use_tensor_core=encoder_decoder_use_tensor_core,
            warmup_iters=warmup_iters,
        )
    raise ValueError(f"Unsupported CUDA graph scope {scope!r}")
