"""Window-attention library adapters: layout and availability, no GPU."""

from __future__ import annotations

import torch

from _aurora_attn_shapes import DEFAULT_WINDOW_SIZE
from _window_attn_libs import (
    AURORA_WINDOW_DHW,
    FLASH_KERNEL_DTYPES,
    FUSED_LIBRARY_NAMES,
    LibrarySpec,
    MICROBENCH_DTYPES,
    available_sdpa_backends,
    library_specs,
    probe_library,
    sdpa_backend_enum,
    sdpa_covers_flash2,
    specs_for_dtype,
    unpack_attn_out,
)


def test_aurora_window_matches_encoder_dhw() -> None:
    assert AURORA_WINDOW_DHW == DEFAULT_WINDOW_SIZE
    assert AURORA_WINDOW_DHW[0] * AURORA_WINDOW_DHW[1] * AURORA_WINDOW_DHW[2] == 144


def test_library_order_is_fused_sdpa_fa4() -> None:
    names = [spec.name for spec in library_specs()]
    assert names[:4] == ["fast_fp32", "cute_tf32", "cute_tf32x3", "cute_bf16"]
    assert "sdpa_auto" in names
    for short, _backend in available_sdpa_backends():
        assert f"sdpa_{short}" in names
    assert "fa4" in names
    if sdpa_covers_flash2():
        assert "fa2" not in names
    for dropped in ("flash_attn", "xformers", "sageattn", "flex_attention", "natten_na3d"):
        assert dropped not in names


def test_baselines_have_fp32_and_bf16() -> None:
    assert MICROBENCH_DTYPES == (torch.float32, torch.bfloat16)
    flash_names = {"fa2", "fa4"}
    for spec in library_specs():
        if spec.name in FUSED_LIBRARY_NAMES or spec.name in flash_names:
            continue
        assert torch.float32 in spec.dtypes
        assert torch.bfloat16 in spec.dtypes
    fused = {spec.name: spec for spec in library_specs() if spec.name in FUSED_LIBRARY_NAMES}
    assert fused["fast_fp32"].dtypes == (torch.float32,)
    assert fused["cute_tf32"].dtypes == (torch.float32,)
    assert fused["cute_tf32x3"].dtypes == (torch.float32,)
    assert fused["cute_bf16"].dtypes == (torch.bfloat16,)
    bf16_names = {spec.name for spec in specs_for_dtype(torch.bfloat16)}
    fp32_names = {spec.name for spec in specs_for_dtype(torch.float32)}
    assert "cute_bf16" in bf16_names and "cute_bf16" not in fp32_names
    assert "fast_fp32" in fp32_names and "fast_fp32" not in bf16_names
    assert "fa4" in bf16_names and "fa4" not in fp32_names
    assert FLASH_KERNEL_DTYPES == (torch.bfloat16,)
    if "fa2" in bf16_names:
        assert "fa2" not in fp32_names


def test_unpack_attn_out_accepts_fa4_tuple() -> None:
    out = torch.zeros(2, 144, 8, 64)
    got = unpack_attn_out((out, None))
    assert got.shape == (2, 8, 144, 64)
    got_plain = unpack_attn_out(out)
    assert got_plain.shape == (2, 8, 144, 64)


def test_fa4_microbench_excludes_layout_conversion() -> None:
    specs = {spec.name: spec for spec in library_specs()}
    assert specs["fa4"].make_timed is not None
    for name, spec in specs.items():
        if name == "fa4":
            continue
        assert spec.make_timed is None


def test_probe_records_missing_optional_library() -> None:
    def _missing(q, k, v, *, scale, bias):
        raise ImportError("flash_attn")

    spec = LibrarySpec("fa4", False, _missing)
    q = torch.zeros(1, 1, 144, 8)
    _out, err = probe_library(spec, q, q, q, scale=0.1, bias=None)
    assert _out is None
    assert err == "ImportError"


def test_probe_skips_bias_when_unsupported() -> None:
    spec = LibrarySpec("fa4", False, lambda **kwargs: None)
    q = torch.zeros(1, 1, 144, 8)
    bias = torch.zeros(1, 144, 144)
    _out, err = probe_library(spec, q, q, q, scale=0.1, bias=bias)
    assert err == "TypeError"


def test_sdpa_backend_enum_handles_missing_cudnn() -> None:
    backends = dict(available_sdpa_backends())
    if sdpa_backend_enum("CUDNN_ATTENTION") is None:
        assert "cudnn" not in backends
    else:
        assert "cudnn" in backends
