"""Glue between Aurora model forward paths and custom CUDA/Triton/CuTe ops.

Custom kernels declare supported I/O dtypes; inference presets keep custom Swin paths
outside ``torch.autocast``. The ``bf16_mixed`` preset casts activations at the backbone
boundary and routes backbone ``F.linear`` matmuls through BF16 (Tensor Core), mimicking
autocast coverage without wrapping custom kernels in ``torch.autocast``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator, TypeVar

import torch

from aurora.model.inference_precision import BackboneComputeDtype

_TRITON_ELEM_DTYPES = frozenset({torch.float32, torch.bfloat16})

_T = TypeVar("_T")


@contextmanager
def encoder_decoder_autocast(*, enabled: bool) -> Iterator[None]:
    """Optional BF16 autocast for encoder/decoder PyTorch modules (not custom Swin ops)."""
    if not enabled:
        yield
        return
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.xpu.is_available():
        device_type = "xpu"
    else:
        device_type = "cpu"
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        yield


def run_with_encoder_decoder_autocast(
    fn: Callable[..., _T],
    *args: Any,
    enabled: bool,
    **kwargs: Any,
) -> _T:
    with encoder_decoder_autocast(enabled=enabled):
        return fn(*args, **kwargs)


@contextmanager
def backbone_bf16_matmul_context(*, enabled: bool) -> Iterator[None]:
    """Route backbone ``F.linear`` matmuls through BF16 on CUDA inference (Tensor Core).

    Patches ``torch.nn.functional.linear`` for the duration of a backbone forward so QKV,
    projection, MLP, and patch-merge linears match ``pytorch_autocast`` matmul coverage while
    Triton/CuTe custom ops stay outside ``torch.autocast``.
    """
    if not enabled:
        yield
        return

    _orig_linear = torch.nn.functional.linear

    def _bf16_linear(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            input.is_cuda
            and not torch.is_grad_enabled()
            and input.dtype in _TRITON_ELEM_DTYPES
        ):
            matmul_dtype = torch.bfloat16
            if input.dtype != matmul_dtype:
                input = input.to(matmul_dtype)
            weight = weight.to(device=input.device, dtype=matmul_dtype)
            if bias is not None:
                bias = bias.to(device=input.device, dtype=matmul_dtype)
            return _orig_linear(input, weight, bias)
        return _orig_linear(input, weight, bias)

    torch.nn.functional.linear = _bf16_linear  # type: ignore[method-assign]
    try:
        yield
    finally:
        torch.nn.functional.linear = _orig_linear


def backbone_dtype_from_name(name: BackboneComputeDtype) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def prepare_backbone_input(x: torch.Tensor, compute_dtype: torch.dtype) -> torch.Tensor:
    """Cast encoder output to the backbone compute dtype when needed."""
    if x.dtype == compute_dtype:
        return x
    return x.to(dtype=compute_dtype)


def finalize_backbone_output(x: torch.Tensor, *, decoder_dtype: torch.dtype) -> torch.Tensor:
    """Restore decoder input dtype after an explicit BF16 backbone pass."""
    if x.dtype == decoder_dtype:
        return x
    return x.to(dtype=decoder_dtype)


def run_backbone_with_dtype_routing(
    backbone: torch.nn.Module,
    x: torch.Tensor,
    *,
    autocast: bool,
    backbone_compute_dtype: torch.dtype | None = None,
    backbone_matmul_bf16: bool = False,
    lead_time: Any,
    patch_res: tuple[int, int, int],
    rollout_step: int,
) -> torch.Tensor:
    """Shared backbone entry for eager forward and CUDA graph capture."""
    decoder_dtype = x.dtype
    with backbone_bf16_matmul_context(enabled=backbone_matmul_bf16):
        if autocast:
            if torch.cuda.is_available():
                device_type = "cuda"
            elif torch.xpu.is_available():
                device_type = "xpu"
            else:
                device_type = "cpu"
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                x = backbone(
                    x,
                    lead_time=lead_time,
                    patch_res=patch_res,
                    rollout_step=rollout_step,
                )
            return finalize_backbone_output(x, decoder_dtype=decoder_dtype)

        if backbone_compute_dtype is not None and x.dtype != backbone_compute_dtype:
            x = prepare_backbone_input(x, backbone_compute_dtype)

        x = backbone(
            x,
            lead_time=lead_time,
            patch_res=patch_res,
            rollout_step=rollout_step,
        )

        if backbone_compute_dtype is not None:
            return finalize_backbone_output(x, decoder_dtype=decoder_dtype)
    return x


def triton_elemwise_dtype_ok(dtype: torch.dtype) -> bool:
    return dtype in _TRITON_ELEM_DTYPES


def align_activation_dtype(reference: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    """Cast ``tensor`` to ``reference.dtype`` when custom ops require matching dtypes."""
    if tensor.dtype == reference.dtype:
        return tensor
    return tensor.to(dtype=reference.dtype)


def align_binary_activations(
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align two activation tensors for fused binary custom ops."""
    if left.dtype == right.dtype:
        return left, right
    if left.dtype in _TRITON_ELEM_DTYPES and right.dtype in _TRITON_ELEM_DTYPES:
        if left.dtype == torch.bfloat16 or right.dtype == torch.bfloat16:
            target = torch.bfloat16
        else:
            target = torch.float32
        return left.to(dtype=target), right.to(dtype=target)
    return left, right


def can_use_triton_layout(x: torch.Tensor, *, enabled: bool) -> bool:
    return (
        enabled
        and x.is_cuda
        and triton_elemwise_dtype_ok(x.dtype)
    )


def can_use_triton_adaln(
    x: torch.Tensor,
    *,
    enabled: bool,
    training: bool,
    drop_path_is_identity: bool,
) -> bool:
    return (
        enabled
        and not training
        and drop_path_is_identity
        and x.is_cuda
        and triton_elemwise_dtype_ok(x.dtype)
    )


def can_use_triton_gelu(
    x: torch.Tensor,
    *,
    enabled: bool,
    training: bool,
    drop_p: float,
) -> bool:
    return (
        enabled
        and not training
        and drop_p == 0.0
        and x.is_cuda
        and triton_elemwise_dtype_ok(x.dtype)
    )


def can_use_cute_window_attention(
    qkv: torch.Tensor,
    *,
    enabled: bool,
    training: bool,
    attn_dropout: float,
) -> bool:
    return (
        enabled
        and not training
        and attn_dropout == 0.0
        and qkv.is_cuda
        and qkv.dtype in (torch.float32, torch.bfloat16)
        and not torch.is_grad_enabled()
    )


def can_use_cute_qkvpacked(
    qkv: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    cute_enabled: bool,
    training: bool,
    attn_dropout: float,
) -> bool:
    """CuTe qkv-packed path: contiguous ``Linear`` output, no explicit Q/K/V split."""
    return (
        can_use_cute_window_attention(
            qkv,
            enabled=cute_enabled,
            training=training,
            attn_dropout=attn_dropout,
        )
        and qkv.is_contiguous()
        and qkv.ndim == 3
        and qkv.shape[-1] == 3 * num_heads * head_dim
    )
