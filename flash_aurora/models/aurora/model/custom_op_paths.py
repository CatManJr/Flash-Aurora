"""Copyright (c) Catman Jr. Licensed under the MIT license.

Glue between Aurora model forward paths and custom CUDA/Triton/CuTe ops.

Custom Triton/CuTe Swin paths never run inside ``torch.autocast``.

* ``bf16_mixed`` - hybrid: BF16 WindowAttention QKV/proj + MLP; other linears TF32.
* ``bf16`` - speed / quant-prep tier: all backbone ``F.linear`` in BF16, fused CuTe
  attention chain (no mid-attn FP32 cast), entry cast to BF16; AdaLN/residual stay FP32.
* ``tf32`` - FP32 activations, TF32 backbone + E/D linears.
"""

from __future__ import annotations

import contextvars
import os
from contextlib import contextmanager
from typing import Any, Callable, Iterator, TypeVar

import torch

from flash_aurora.aurora.model.inference_precision import BackboneComputeDtype

_TRITON_ELEM_DTYPES = frozenset({torch.float32, torch.bfloat16})

_T = TypeVar("_T")

_bf16_matmul_routing: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_bf16_matmul_routing", default=False
)
_bf16_mlp_matmul_scope: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_bf16_mlp_matmul_scope", default=False
)
_bf16_attention_matmul_scope: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_bf16_attention_matmul_scope", default=False
)
_bf16_hybrid_matmul: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_bf16_hybrid_matmul", default=False
)


def backbone_bf16_matmul_active() -> bool:
    """True while a BF16 backbone matmul preset is active."""
    return _bf16_matmul_routing.get()


def backbone_bf16_hybrid_matmul_active() -> bool:
    """True for ``bf16_mixed`` (BF16 attention/MLP; TF32 otherwise)."""
    return _bf16_matmul_routing.get() and _bf16_hybrid_matmul.get()


def backbone_bf16_full_matmul_active() -> bool:
    """True for ``bf16`` (all backbone linears BF16, no TF32 hybrid hooks)."""
    return _bf16_matmul_routing.get() and not _bf16_hybrid_matmul.get()


def backbone_bf16_speed_stream_active() -> bool:
    """Optional BF16 activation stream between AdaLN islands (``bf16`` only).

    Opt-in via ``AURORA_BF16_SPEED_STREAM=1``; default off until tol-validated for production.
    """
    if not backbone_bf16_full_matmul_active():
        return False
    return os.environ.get("AURORA_BF16_SPEED_STREAM", "0") == "1"


def use_bf16_fused_attention_chain(
    *,
    use_cute_window_attn: bool,
    cute_window_attn_dtype: torch.dtype,
    is_cuda: bool,
) -> bool:
    """QKV -> CuTe attn -> proj without an intermediate FP32 cast."""
    if not (
        use_cute_window_attn
        and cute_window_attn_dtype == torch.bfloat16
        and is_cuda
        and not torch.is_grad_enabled()
    ):
        return False
    if backbone_bf16_full_matmul_active():
        return True
    return (
        backbone_bf16_hybrid_matmul_active()
        and bf16_mixed_attention_linear_enabled()
    )


def downcast_bf16_speed_activation(tensor: torch.Tensor) -> torch.Tensor:
    """Restore BF16 activation stream after an FP32 AdaLN/LN island."""
    if backbone_bf16_speed_stream_active() and tensor.dtype == torch.float32:
        return cast_activation_dtype(tensor, torch.bfloat16)
    return tensor


def bf16_mixed_attention_linear_enabled() -> bool:
    """Route WindowAttention QKV/proj linears through BF16 in bf16_mixed.

    The opt-out exists only for A/B regression checks against the old hybrid path.
    """
    return os.environ.get("AURORA_BF16_MIXED_ATTENTION_LINEAR", "1") != "0"


@contextmanager
def backbone_bf16_mlp_matmul_scope() -> Iterator[None]:
    """Mark ``F.linear`` inside MLP ``fc1``/``fc2`` for BF16 (``bf16_mixed`` hybrid only)."""
    token = _bf16_mlp_matmul_scope.set(True)
    try:
        yield
    finally:
        _bf16_mlp_matmul_scope.reset(token)


@contextmanager
def backbone_bf16_attention_matmul_scope() -> Iterator[None]:
    """Mark WindowAttention QKV/proj ``F.linear`` for BF16 in hybrid mode."""
    token = _bf16_attention_matmul_scope.set(True)
    try:
        yield
    finally:
        _bf16_attention_matmul_scope.reset(token)


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
    return run_with_encoder_decoder_routing(
        fn, *args, autocast_bf16=enabled, use_tensor_core=False, **kwargs
    )


def run_with_encoder_decoder_routing(
    fn: Callable[..., _T],
    *args: Any,
    autocast_bf16: bool = False,
    use_tensor_core: bool = False,
    **kwargs: Any,
) -> _T:
    """Encoder/decoder forward with optional BF16 autocast and/or TF32 ``F.linear`` matmul."""
    with encoder_decoder_autocast(enabled=autocast_bf16):
        with backbone_tf32_matmul_context(enabled=use_tensor_core):
            return fn(*args, **kwargs)


def _bf16_layer_norm_hook(
    orig_ln: Any,
    input: torch.Tensor,
    normalized_shape: int | list[int] | torch.Size,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    if (
        input.is_cuda
        and not torch.is_grad_enabled()
        and input.dtype == torch.bfloat16
    ):
        with torch.autocast(device_type="cuda", enabled=False):
            return orig_ln(
                cast_activation_dtype(input, torch.float32),
                normalized_shape,
                weight,
                bias,
                eps=eps,
            )
    return orig_ln(input, normalized_shape, weight, bias, eps=eps)


@contextmanager
def backbone_bf16_mixed_matmul_context(*, enabled: bool) -> Iterator[None]:
    """``bf16_mixed``: BF16 WindowAttention + MLP, TF32 for other linears.

    * WindowAttention QKV/proj: BF16 autocast.
    * MLP ``fc1``/``fc2``: BF16 autocast.
    * Patch merge, AdaLN modulation, and other backbone linears: TF32 Tensor Core, FP32 I/O.
    * ``F.layer_norm`` on BF16 input: FP32 statistics/output.
    """
    if not enabled:
        yield
        return

    _orig_layer_norm = torch.nn.functional.layer_norm
    _orig_linear = torch.nn.functional.linear

    def _enable_tf32_flags() -> None:
        torch.set_float32_matmul_precision("high")
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    _enable_tf32_flags()

    def _bf16_layer_norm(
        input: torch.Tensor,
        normalized_shape: int | list[int] | torch.Size,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        return _bf16_layer_norm_hook(
            _orig_layer_norm, input, normalized_shape, weight, bias, eps
        )

    def _hybrid_linear(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (_bf16_mlp_matmul_scope.get() or _bf16_attention_matmul_scope.get()) and (
            input.is_cuda
            and not torch.is_grad_enabled()
            and input.dtype in (torch.float32, torch.bfloat16)
            and weight.dtype in (torch.float32, torch.bfloat16)
        ):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return _orig_linear(input, weight, bias)
        if (
            input.is_cuda
            and not torch.is_grad_enabled()
            and input.dtype == torch.float32
            and weight.dtype == torch.float32
        ):
            _enable_tf32_flags()
            return _orig_linear(input, weight, bias)
        return _orig_linear(input, weight, bias)

    prev_precision = torch.get_float32_matmul_precision()
    prev_cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32

    token = _bf16_matmul_routing.set(True)
    hybrid_token = _bf16_hybrid_matmul.set(True)
    torch.nn.functional.layer_norm = _bf16_layer_norm  # type: ignore[method-assign]
    torch.nn.functional.linear = _hybrid_linear  # type: ignore[method-assign]
    try:
        yield
    finally:
        torch.nn.functional.linear = _orig_linear
        torch.nn.functional.layer_norm = _orig_layer_norm
        _bf16_hybrid_matmul.reset(hybrid_token)
        _bf16_matmul_routing.reset(token)
        torch.set_float32_matmul_precision(prev_precision)
        torch.backends.cuda.matmul.allow_tf32 = prev_cuda_tf32
        torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32


@contextmanager
def backbone_bf16_matmul_context(*, enabled: bool) -> Iterator[None]:
    """``bf16``: BF16 ``F.linear`` on the full backbone.

    * All backbone ``F.linear``: BF16 via ``torch.autocast``.
    * ``F.layer_norm`` on BF16 input: FP32 output (AdaLN stability).
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
        return _bf16_layer_norm_hook(
            _orig_layer_norm, input, normalized_shape, weight, bias, eps
        )

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

    token = _bf16_matmul_routing.set(True)
    torch.nn.functional.layer_norm = _bf16_layer_norm  # type: ignore[method-assign]
    torch.nn.functional.linear = _bf16_linear  # type: ignore[method-assign]
    try:
        yield
    finally:
        torch.nn.functional.linear = _orig_linear
        torch.nn.functional.layer_norm = _orig_layer_norm
        _bf16_matmul_routing.reset(token)


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
    """Backbone matmul routing: hybrid BF16, full BF16 backbone, or TF32."""
    if bf16 and tf32:
        with backbone_bf16_mixed_matmul_context(enabled=True):
            yield
    elif bf16:
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

        if backbone_matmul_bf16 and not backbone_matmul_tf32 and not autocast:
            x = prepare_backbone_input(x, torch.bfloat16)
        elif (
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
    """Fused AdaLN on CUDA FP32 activations (residual stream stays FP32)."""
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
