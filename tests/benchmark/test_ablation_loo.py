"""Leave-one-out ablation helpers: flag mutation and worker CLI, no GPU."""

from __future__ import annotations

import pytest

from _ablation_loo import (
    COMPILE_EXTRA_WARMUP,
    DEFAULT_WARMUP,
    LOO_ROWS,
    apply_loo_flags,
    extra_warmup,
    get_row,
    intervals_overlap,
    row_ids,
)
from _ablation_loo_worker import build_parser


class _Norm:
    def __init__(self) -> None:
        self.use_triton = True


class _Block:
    def __init__(self) -> None:
        self.use_triton_layout = True
        self.use_cute_window_attn = True
        self.norm1 = _Norm()
        self.norm2 = _Norm()

    def modules(self):
        yield self


class _Backbone:
    def __init__(self) -> None:
        self.use_cute_window_attn = True
        self.block = _Block()

    def modules(self):
        yield self
        yield self.block


class _Model:
    def __init__(self) -> None:
        self.backbone = _Backbone()
        self.use_triton_layout = True
        self.use_cute_window_attn = True

    def modules(self):
        yield self
        yield from self.backbone.modules()


def test_row_ids_are_unique() -> None:
    ids = row_ids()
    assert ids == tuple(row.row_id for row in LOO_ROWS)
    assert len(ids) == len(set(ids))
    assert "full" in ids
    assert "no_layout" in ids
    assert "no_adaln" in ids
    assert "no_cute" in ids
    assert "no_bf16_routing" in ids


def test_get_row_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown ablation row"):
        get_row("not_a_row")


def test_no_layout_clears_triton_layout_only() -> None:
    model = _Model()
    apply_loo_flags(model, get_row("no_layout"))
    assert model.use_triton_layout is False
    assert model.backbone.block.use_triton_layout is False
    assert model.backbone.block.norm1.use_triton is True
    assert model.backbone.block.use_cute_window_attn is True


def test_no_adaln_clears_norm_triton_only() -> None:
    model = _Model()
    apply_loo_flags(model, get_row("no_adaln"))
    assert model.backbone.block.norm1.use_triton is False
    assert model.backbone.block.norm2.use_triton is False
    assert model.backbone.block.use_triton_layout is True
    assert model.backbone.block.use_cute_window_attn is True


def test_no_cute_clears_window_attn_only() -> None:
    model = _Model()
    apply_loo_flags(model, get_row("no_cute"))
    assert model.backbone.use_cute_window_attn is False
    assert model.backbone.block.use_cute_window_attn is False
    assert model.backbone.block.use_triton_layout is True
    assert model.backbone.block.norm1.use_triton is True


def test_full_row_does_not_mutate_flags() -> None:
    model = _Model()
    apply_loo_flags(model, get_row("full"))
    assert model.use_triton_layout is True
    assert model.backbone.block.norm1.use_triton is True
    assert model.backbone.use_cute_window_attn is True


def test_compile_after_load_wraps_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}

    def _fake_compile(module, *, dynamic: bool):
        called["module"] = module
        called["dynamic"] = dynamic
        return "compiled-backbone"

    monkeypatch.setattr("_ablation_loo.torch.compile", _fake_compile)
    model = _Model()
    original = model.backbone
    apply_loo_flags(model, get_row("compile_after_load"))
    assert called["dynamic"] is False
    assert called["module"] is original
    assert model.backbone == "compiled-backbone"


def test_compile_row_uses_extra_warmup() -> None:
    assert extra_warmup(get_row("compile_after_load"), DEFAULT_WARMUP) == (
        DEFAULT_WARMUP + COMPILE_EXTRA_WARMUP
    )
    assert extra_warmup(get_row("full"), DEFAULT_WARMUP) == DEFAULT_WARMUP


def test_no_bf16_routing_uses_tf32_fused_precision() -> None:
    row = get_row("no_bf16_routing")
    assert row.precision == "tf32@fp32"
    assert row.disable_layout is False
    assert row.disable_cute is False


def test_intervals_overlap() -> None:
    assert intervals_overlap(10.0, 1.0, 10.5, 1.0) is True
    assert intervals_overlap(10.0, 0.1, 20.0, 0.1) is False


def test_worker_help_exits_zero() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_worker_rejects_unknown_row() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--row", "not_a_row", "--preset", "era5_pretrained"])
    assert exc.value.code != 0
