"""Group A baseline-matrix rows: no GPU."""

from __future__ import annotations

import pytest

from _ablation_loo import COMPILE_EXTRA_WARMUP
from _baseline_matrix import (
    DEFAULT_WARMUP,
    HEADLINE_IDS,
    SDPA_MATH_REF_ID,
    STATIC_ROWS,
    annotate_speedups,
    apply_row_flags,
    extra_warmup,
    get_row,
    row_ids,
    strongest_passing_baseline,
)
from _baseline_matrix_worker import build_parser


class _Backbone:
    def __init__(self) -> None:
        self.use_cute_window_attn = True

    def modules(self):
        yield self


class _Model:
    def __init__(self) -> None:
        self.backbone = _Backbone()
        self.use_cute_window_attn = True

    def modules(self):
        yield self
        yield self.backbone


def test_row_ids_are_unique() -> None:
    ids = row_ids()
    assert len(ids) == len(set(ids))
    for required in (
        "mixed",
        "eager_fp32",
        "eager_tf32",
        "compile_fp32",
        "autocast",
        "compile_autocast",
        "fast_fp32",
        "tf32_fused",
        "tf32x3_fused",
        "sdpa_auto_fp32",
        "sdpa_auto_bf16",
    ):
        assert required in ids


def test_eager_tf32_uses_named_pytorch_tf32_preset() -> None:
    row = get_row("eager_tf32")
    assert row.precision == "pytorch_tf32"
    assert row.compile_after_load is False
    assert row.disable_cute is False


def test_eager_tf32_is_not_fused_tf32_combo() -> None:
    from flash_aurora.models.inference_precision import resolve_inference_config

    eager = resolve_inference_config(get_row("eager_tf32").precision)
    fused = resolve_inference_config("tf32@fp32")
    assert eager is not None
    assert fused is not None
    assert eager.use_cute_window_attn is False
    assert fused.use_cute_window_attn is True


def test_compile_rows_add_warmup() -> None:
    compile_fp32 = get_row("compile_fp32")
    compile_autocast = get_row("compile_autocast")
    assert extra_warmup(compile_fp32, DEFAULT_WARMUP) == DEFAULT_WARMUP + COMPILE_EXTRA_WARMUP
    assert extra_warmup(compile_autocast, DEFAULT_WARMUP) == DEFAULT_WARMUP + COMPILE_EXTRA_WARMUP
    assert extra_warmup(get_row("eager_fp32"), DEFAULT_WARMUP) == DEFAULT_WARMUP


def test_fused_ladder_rows_use_named_combos() -> None:
    from flash_aurora.models.inference_precision import resolve_inference_config

    assert get_row("fast_fp32").precision == "fast_fp32"
    assert get_row("tf32_fused").precision == "tf32@fp32"
    assert get_row("tf32x3_fused").precision == "tf32x3@fp32"
    assert get_row("mixed").precision == "bf16_mixed@fp32"
    assert get_row("fast_fp32").disable_cute is False
    assert get_row("tf32_fused").disable_cute is False
    for row_id in ("fast_fp32", "tf32_fused", "tf32x3_fused", "mixed"):
        cfg = resolve_inference_config(get_row(row_id).precision)
        assert cfg is not None


def test_sdpa_rows_cover_fp32_and_bf16() -> None:
    for row_id in row_ids():
        if not row_id.startswith("sdpa_"):
            continue
        row = get_row(row_id)
        assert row.disable_cute is True
        if row_id.endswith("_fp32"):
            assert row.precision == "fp32"
        elif row_id.endswith("_bf16"):
            assert row.precision == "bf16_mixed@fp32"
        else:
            raise AssertionError(f"SDPA row {row_id} must end in _fp32 or _bf16")
    assert get_row("sdpa_auto_fp32").sdpa_backend is None
    assert get_row("sdpa_auto_bf16").sdpa_backend is None
    from _window_attn_libs import available_sdpa_backends

    ids = row_ids()
    for short, _backend in available_sdpa_backends():
        assert f"sdpa_{short}_fp32" in ids
        assert f"sdpa_{short}_bf16" in ids


def test_apply_row_flags_disables_cute() -> None:
    model = _Model()
    apply_row_flags(model, get_row("sdpa_auto_bf16"))
    assert model.backbone.use_cute_window_attn is False


def test_apply_compile_wraps_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}

    def _fake_compile(module, *, dynamic: bool):
        called["module"] = module
        called["dynamic"] = dynamic
        return "compiled"

    monkeypatch.setattr("_baseline_matrix.torch.compile", _fake_compile)
    model = _Model()
    original = model.backbone
    apply_row_flags(model, get_row("compile_fp32"))
    assert called["dynamic"] is False
    assert called["module"] is original
    assert model.backbone == "compiled"


def test_strongest_passing_baseline_picks_fastest_ok_row() -> None:
    rows = {
        "eager_fp32": {"ok": True, "mean": 2000.0, "quality": {"n_fail": 0}},
        "eager_tf32": {"ok": True, "mean": 900.0, "quality": {"n_fail": 0}},
        "compile_fp32": {"ok": True, "mean": 800.0, "quality": {"n_fail": 1}},
        "autocast": {"ok": False, "mean": 100.0, "quality": {"n_fail": 0}},
        "compile_autocast": {"ok": True, "mean": 850.0, "quality": {"n_fail": 0}},
    }
    assert strongest_passing_baseline(rows) == "compile_autocast"
    assert set(HEADLINE_IDS) == {
        "eager_fp32",
        "eager_tf32",
        "compile_fp32",
        "autocast",
        "compile_autocast",
    }


def test_speedup_is_against_sdpa_math_fp32() -> None:
    assert SDPA_MATH_REF_ID == "sdpa_math_fp32"
    rows = {
        "sdpa_math_fp32": {"ok": True, "mean": 2400.0},
        "mixed": {"ok": True, "mean": 600.0},
        "sdpa_math_bf16": {"ok": True, "mean": 1200.0},
        "sdpa_flash_fp32": {"ok": False},
    }
    annotate_speedups(rows)
    assert rows["sdpa_math_fp32"]["vs_sdpa_math"] == pytest.approx(1.0)
    assert rows["mixed"]["vs_sdpa_math"] == pytest.approx(4.0)
    assert rows["sdpa_math_bf16"]["vs_sdpa_math"] == pytest.approx(2.0)
    assert rows["sdpa_flash_fp32"]["vs_sdpa_math"] is None
    assert rows["mixed"]["speedup_ref"] == "sdpa_math_fp32"


def test_static_rows_do_not_include_sdpa() -> None:
    assert all(not row.row_id.startswith("sdpa_") for row in STATIC_ROWS)


def test_get_row_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown baseline row"):
        get_row("not_a_row")


def test_worker_help_exits_zero() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0
