"""Glue between Aurora model forward paths and custom CUDA/Triton/CuTe ops.

Custom Triton/CuTe Swin paths never run inside ``torch.autocast``. ``bf16_mixed`` routes all
backbone ``F.linear`` matmuls (QKV, proj, MLP, patch merge, …) through BF16 Tensor Core and
keeps BF16 activations between consecutive GEMMs where LayerNorm does not intervene.
``F.layer_norm`` breaks the chain with FP32 statistics/output (like autocast's fp32-safe ops).
``tf32_1x`` keeps FP32 activations with TF32 matmul only.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Any, Callable, Iterator, TypeVar

import torch

from aurora.model.inference_precision import BackboneComputeDtype

_TRITON_ELEM_DTYPES = frozenset({torch.float32, torch.bfloat16})

_T = TypeVar("_T")

_bf16_backbone_matmul_enabled: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_bf16_backbone_matmul_enabled", default=False
)


def backbone_bf16_matmul_active() -> bool:
    """True while ``backbone_bf16_matmul_context`` is active (all hooked ``F.linear`` use BF16)."""
    return _bf16_backbone_matmul_enabled.get()


def backbone_bf16_mlp_matmul_active() -> bool:
    """Alias of :func:`backbone_bf16_matmul_active` (historical MLP-only name)."""
    return backbone_bf16_matmul_active()


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
    """BF16 Tensor Core for all backbone ``F.linear``; FP32 LayerNorm stats (``bf16_mixed``).

    * ``F.linear`` (QKV, proj, MLP, patch merge, AdaLN modulation, …): narrow CUDA BF16
      autocast on FP32/BF16 I/O (weights stay FP32; outputs are typically BF16).
    * ``F.layer_norm`` on BF16 input: FP32 statistics/output (AdaLN / patch norms).
    * Triton AdaLN still requires FP32 activations at its boundary; it upcasts internally.
    """
    if not enabled:
        yield
        return

    _orig_layer_norm = torch.nn.functional.layer_norm
    _orig_linear = torch.nn.functional.linear

    def _bf16_layer_norm(
        input: torch.Tensor,
        normalized_shape: int | list[int] | torch.Size,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        if (
            input.is_cuda
            and not torch.is_grad_enabled()
            and input.dtype == torch.bfloat16
        ):
            with torch.autocast(device_type="cuda", enabled=False):
                return _orig_layer_norm(
                    cast_activation_dtype(input, torch.float32),
                    normalized_shape,
                    weight,
                    bias,
                    eps=eps,
                )
        return _orig_layer_norm(input, normalized_shape, weight, bias, eps=eps)

    def _bf16_linear(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            input.is_cuda
            and not torch.is_grad_enabled()
            and input.dtype in (torch.float32, torch.bfloat16)
            and weight.dtype in (torch.float32, torch.bfloat16)
        ):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return _orig_linear(input, weight, bias)
        return _orig_linear(input, weight, bias)

    token = _bf16_backbone_matmul_enabled.set(True)
    torch.nn.functional.layer_norm = _bf16_layer_norm  # type: ignore[method-assign]
    torch.nn.functional.linear = _bf16_linear  # type: ignore[method-assign]
    try:
        yield
    finally:
        torch.nn.functional.linear = _orig_linear
        torch.nn.functional.layer_norm = _orig_layer_norm
        _bf16_backbone_matmul_enabled.reset(token)


@contextmanager
def backbone_tf32_matmul_context(*, enabled: bool) -> Iterator[None]:
    """Route backbone ``F.linear`` / ``addmm`` matmuls through TF32 Tensor Core, FP32 I/O.

    Enables ``float32_matmul_precision='high'`` and CUDA TF32 flags for the backbone forward.
    Each ``F.linear`` call is additionally wrapped so matmul flags apply even if outer code
    changed global settings. Activations stay FP32 (LayerNorm-safe).
    """
    if not enabled:
        yield
        return

    prev_precision = torch.get_float32_matmul_precision()
    prev_cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32

    def _enable_tf32_flags() -> None:
        torch.set_float32_matmul_precision("high")
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    _enable_tf32_flags()

    _orig_linear = torch.nn.functional.linear

    def _tf32_linear(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            input.is_cuda
            and not torch.is_grad_enabled()
            and input.dtype == torch.float32
            and weight.dtype == torch.float32
        ):
            _enable_tf32_flags()
            return _orig_linear(input, weight, bias)
        return _orig_linear(input, weight, bias)

    torch.nn.functional.linear = _tf32_linear  # type: ignore[method-assign]
    try:
        yield
    finally:
        torch.nn.functional.linear = _orig_linear
        torch.set_float32_matmul_precision(prev_precision)
        torch.backends.cuda.matmul.allow_tf32 = prev_cuda_tf32
        torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32


@contextmanager
def backbone_matmul_context(*, tf32: bool = False, bf16: bool = False) -> Iterator[None]:
    """Exactly one of ``tf32`` or ``bf16`` may be enabled for backbone matmul routing."""
    if tf32 and bf16:
        raise ValueError("backbone_matmul_context: tf32 and bf16 are mutually exclusive.")
    if bf16:
        with backbone_bf16_matmul_context(enabled=True):
            yield
    elif tf32:
        with backbone_tf32_matmul_context(enabled=True):
            yield
    else:
        yield


def backbone_dtype_from_name(name: BackboneComputeDtype) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def cast_activation_dtype(
    tensor: torch.Tensor,
    dtype: torch.dtype,
    *,
    non_blocking: bool | None = None,
) -> torch.Tensor:
    """Cast activations; use async GPU copies for inference FP32/BF16 pairs."""
    if tensor.dtype == dtype:
        return tensor
    if non_blocking is None:
        non_blocking = (
            tensor.is_cuda
            and not torch.is_grad_enabled()
            and tensor.dtype in _TRITON_ELEM_DTYPES
            and dtype in _TRITON_ELEM_DTYPES
        )
    return tensor.to(dtype=dtype, non_blocking=non_blocking)


def prepare_backbone_input(x: torch.Tensor, compute_dtype: torch.dtype) -> torch.Tensor:
    """Cast encoder output to the backbone compute dtype when needed."""
    return cast_activation_dtype(x, compute_dtype)


def finalize_backbone_output(x: torch.Tensor, *, decoder_dtype: torch.dtype) -> torch.Tensor:
    """Restore decoder input dtype after an explicit BF16 backbone pass."""
    return cast_activation_dtype(x, decoder_dtype)


def run_backbone_with_dtype_routing(
    backbone: torch.nn.Module,
    x: torch.Tensor,
    *,
    autocast: bool,
    backbone_compute_dtype: torch.dtype | None = None,
    backbone_matmul_bf16: bool = False,
    backbone_matmul_tf32: bool = False,
    lead_time: Any,
    patch_res: tuple[int, int, int],
    rollout_step: int,
) -> torch.Tensor:
    """Shared backbone entry for eager forward and CUDA graph capture."""
    decoder_dtype = x.dtype
    with backbone_matmul_context(tf32=backbone_matmul_tf32, bf16=backbone_matmul_bf16):
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

        if (
            backbone_compute_dtype is not None
            and not backbone_matmul_bf16
            and x.dtype != backbone_compute_dtype
        ):
            x = prepare_backbone_input(x, backbone_compute_dtype)

        x = backbone(
            x,
            lead_time=lead_time,
            patch_res=patch_res,
            rollout_step=rollout_step,
        )

        if backbone_matmul_bf16 and not autocast:
            return finalize_backbone_output(x, decoder_dtype=decoder_dtype)
        if backbone_compute_dtype is not None:
            return finalize_backbone_output(x, decoder_dtype=decoder_dtype)
    return x


def triton_elemwise_dtype_ok(dtype: torch.dtype) -> bool:
    return dtype in _TRITON_ELEM_DTYPES


def align_activation_dtype(reference: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    """Cast ``tensor`` to ``reference.dtype`` when custom ops require matching dtypes."""
    return cast_activation_dtype(tensor, reference.dtype)


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
        return (
            cast_activation_dtype(left, target),
            cast_activation_dtype(right, target),
        )
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
    """Fused AdaLN on CUDA FP32 (norm boundary vs ``tf32_1x`` reference; BF16 GEMM upstream)."""
    return (
        enabled
        and not training
        and drop_path_is_identity
        and x.is_cuda
        and x.dtype == torch.float32
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
