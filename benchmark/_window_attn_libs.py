"""Window-attention adapters on Aurora QKV layout ``(Bwin, H, N, Dh)``.

The microbench compares the fused ladder, every PyTorch SDPA backend, and
FlashAttention-4. SDPA ``FLASH_ATTENTION`` is the in-tree FlashAttention-2
path, so a separate ``flash-attn`` package is not added when that backend
exists. FA-4 is CuTe DSL and is timed only in BF16. A library that cannot
consume this layout, or that raises on ``sm_120a``, is recorded as unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from _aurora_attn_shapes import DEFAULT_WINDOW_SIZE
from flash_aurora.models.ops.cute.window_attn_fwd import (
    WinAttnPrecision,
    _expand_bias_for_sdpa,
    window_attn_fwd_cute,
)

AURORA_WINDOW_DHW = DEFAULT_WINDOW_SIZE
MICROBENCH_DTYPES: tuple[torch.dtype, ...] = (torch.float32, torch.bfloat16)
FUSED_LIBRARY_NAMES: frozenset[str] = frozenset(
    {"fast_fp32", "cute_tf32", "cute_tf32x3", "cute_bf16"}
)
SDPA_BACKEND_NAMES: tuple[tuple[str, str], ...] = (
    ("flash", "FLASH_ATTENTION"),
    ("mem_eff", "EFFICIENT_ATTENTION"),
    ("math", "MATH"),
    ("cudnn", "CUDNN_ATTENTION"),
)
# FlashAttention kernels are fp16/bf16. They do not expose an FP32 or TF32 MMA.
FLASH_KERNEL_DTYPES: tuple[torch.dtype, ...] = (torch.bfloat16,)


@dataclass(frozen=True)
class LibrarySpec:
    name: str
    supports_bias: bool
    run: Callable[..., torch.Tensor]
    dtypes: tuple[torch.dtype, ...] = (torch.bfloat16,)
    # If set, the microbench times this factory's zero-arg kernel after layout
    # conversion. ``run`` still includes BHSD<->BSHD copies for the probe.
    make_timed: Callable[..., Callable[[], Any]] | None = None
    # Dispatching libraries (SDPA without a forced backend) name the backend
    # PyTorch selected, so reports never have to say only "auto".
    resolve_backend: Callable[..., str] | None = None
    # SDPA needs the compact (nW, N, N) bias expanded to a dense per-window mask,
    # while the fused kernels read the compact bias directly. True marks specs
    # whose timed kernel excludes that expansion and that also report the
    # expansion-included latency for masked shapes.
    reports_mask_build_variant: bool = False


def sdpa_backend_enum(attr_name: str) -> SDPBackend | None:
    return getattr(SDPBackend, attr_name, None)


def available_sdpa_backends() -> tuple[tuple[str, SDPBackend], ...]:
    found: list[tuple[str, SDPBackend]] = []
    for short, attr in SDPA_BACKEND_NAMES:
        backend = sdpa_backend_enum(attr)
        if backend is not None:
            found.append((short, backend))
    return tuple(found)


def sdpa_covers_flash2() -> bool:
    """True when PyTorch SDPA already ships the FlashAttention-2 backend."""
    return any(short == "flash" for short, _ in available_sdpa_backends())


def to_bshd(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2).contiguous()


def from_bshd(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2).contiguous()


def unpack_attn_out(result: torch.Tensor | tuple[object, ...]) -> torch.Tensor:
    # FA-4 FlashAttnFunc.apply always returns (out, lse). lse is None when
    # return_lse is False, so the adapter must not transpose the tuple.
    if isinstance(result, tuple):
        result = result[0]
    if not isinstance(result, torch.Tensor):
        raise TypeError(f"expected attention tensor, got {type(result).__name__}")
    return from_bshd(result)


def expand_bias(bias: torch.Tensor | None, q: torch.Tensor) -> torch.Tensor | None:
    if bias is None:
        return None
    return _expand_bias_for_sdpa(bias, q.shape[0], q.shape[1], q.shape[2]).to(dtype=q.dtype)


def resolve_sdpa_auto_backend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    bias: torch.Tensor | None,
) -> str:
    """Short name (``flash``, ``mem_eff``, ...) of the backend SDPA auto-dispatch picks.

    ``scale`` is accepted for signature parity with the run adapters; the
    dispatcher's choice does not depend on it.
    """
    chosen = SDPBackend(
        torch._fused_sdp_choice(q, k, v, attn_mask=expand_bias(bias, q), dropout_p=0.0, is_causal=False)
    )
    for short, attr in SDPA_BACKEND_NAMES:
        if sdpa_backend_enum(attr) == chosen:
            return short
    return chosen.name.lower()


def _run_cute(precision: WinAttnPrecision) -> Callable[..., torch.Tensor]:
    def _fwd(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        return window_attn_fwd_cute(
            q,
            k,
            v,
            bias=bias,
            precision=precision,
            scale_qk=scale,
        )

    return _fwd


def _run_sdpa_auto(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    mask = expand_bias(bias, q)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)


def _make_sdpa_timed(backend: SDPBackend | None) -> Callable[..., Callable[[], Any]]:
    """Build the dense mask once so the timed kernel is attention only.

    ``backend=None`` keeps PyTorch's auto-dispatch.
    """

    def _factory(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float,
        bias: torch.Tensor | None,
    ) -> Callable[[], Any]:
        mask = expand_bias(bias, q)

        def _kernel() -> torch.Tensor:
            if backend is None:
                return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
            with sdpa_kernel(backends=[backend]):
                return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)

        return _kernel

    return _factory


def _run_sdpa(
    backend: SDPBackend,
) -> Callable[..., torch.Tensor]:
    def _fwd(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        mask = expand_bias(bias, q)
        with sdpa_kernel(backends=[backend]):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)

    return _fwd


def _run_fa4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    if bias is not None:
        raise TypeError("FA-4 does not take a Swin additive bias")
    from flash_attn.cute import flash_attn_func

    result = flash_attn_func(to_bshd(q), to_bshd(k), to_bshd(v), softmax_scale=scale)
    return unpack_attn_out(result)


def _make_fa4_timed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    bias: torch.Tensor | None,
) -> Callable[[], Any]:
    # FA-4 consumes BSHD. Convert once outside the CUDA-event window so the
    # kernel row is comparable to SDPA/CuTe, which already sit in BHSD.
    if bias is not None:
        raise TypeError("FA-4 does not take a Swin additive bias")
    q_bshd = to_bshd(q)
    k_bshd = to_bshd(k)
    v_bshd = to_bshd(v)
    from flash_attn.cute import flash_attn_func

    def _kernel() -> Any:
        return flash_attn_func(q_bshd, k_bshd, v_bshd, softmax_scale=scale)

    return _kernel


def _run_fa2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    if bias is not None:
        raise TypeError("FA-2 does not take a Swin additive bias")
    from flash_attn import flash_attn_func

    out = flash_attn_func(to_bshd(q), to_bshd(k), to_bshd(v), softmax_scale=scale)
    return from_bshd(out)


def library_specs() -> tuple[LibrarySpec, ...]:
    fp32 = (torch.float32,)
    bf16 = (torch.bfloat16,)
    both = MICROBENCH_DTYPES
    specs: list[LibrarySpec] = [
        LibrarySpec(
            "fast_fp32", True, _run_sdpa_auto, fp32, _make_sdpa_timed(None),
            resolve_backend=resolve_sdpa_auto_backend, reports_mask_build_variant=True,
        ),
        LibrarySpec("cute_tf32", True, _run_cute(WinAttnPrecision.TF32), fp32),
        LibrarySpec("cute_tf32x3", True, _run_cute(WinAttnPrecision.TF32X3), fp32),
        LibrarySpec("cute_bf16", True, _run_cute(WinAttnPrecision.BF16_MIXED), bf16),
        LibrarySpec(
            "sdpa_auto", True, _run_sdpa_auto, both, _make_sdpa_timed(None),
            resolve_backend=resolve_sdpa_auto_backend, reports_mask_build_variant=True,
        ),
    ]
    for short, backend in available_sdpa_backends():
        specs.append(
            LibrarySpec(
                f"sdpa_{short}", True, _run_sdpa(backend), both, _make_sdpa_timed(backend),
                reports_mask_build_variant=True,
            )
        )
    if not sdpa_covers_flash2():
        specs.append(LibrarySpec("fa2", False, _run_fa2, FLASH_KERNEL_DTYPES))
    specs.append(LibrarySpec("fa4", False, _run_fa4, FLASH_KERNEL_DTYPES, _make_fa4_timed))
    return tuple(specs)


def specs_for_dtype(dtype: torch.dtype) -> tuple[LibrarySpec, ...]:
    return tuple(spec for spec in library_specs() if dtype in spec.dtypes)


def probe_library(
    spec: LibrarySpec,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    bias: torch.Tensor | None,
) -> tuple[torch.Tensor | None, str | None]:
    if bias is not None and not spec.supports_bias:
        return None, "TypeError"
    try:
        out = spec.run(q, k, v, scale=scale, bias=bias)
        return out, None
    except Exception as exc:  # noqa: BLE001
        return None, type(exc).__name__
