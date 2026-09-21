"""Default isolate-tiers ladder includes 3xTF32 combos."""

from __future__ import annotations

from _latency_bench import (
    DEFAULT_LATENCY_REPEAT,
    DEFAULT_LATENCY_TIERS,
    DEFAULT_LATENCY_WARMUP,
    PYTORCH_FP32_REF_TIER,
    resolve_tier_specs,
)
from _pretrained_era5 import summarize_repeat_ms


def test_default_latency_tiers_include_tf32x3() -> None:
    assert DEFAULT_LATENCY_TIERS[0] == PYTORCH_FP32_REF_TIER
    assert "tf32@fp32" in DEFAULT_LATENCY_TIERS
    assert "tf32x3@fp32" in DEFAULT_LATENCY_TIERS
    assert "tf32x3@tf32" in DEFAULT_LATENCY_TIERS
    tf32_at = DEFAULT_LATENCY_TIERS.index("tf32@tf32")
    assert DEFAULT_LATENCY_TIERS[tf32_at + 1] == "tf32x3@fp32"
    assert DEFAULT_LATENCY_TIERS[tf32_at + 2] == "tf32x3@tf32"


def test_tf32x3_latency_specs_keep_x3_precision_string() -> None:
    specs = dict(resolve_tier_specs(["tf32x3@fp32", "tf32@fp32"]))
    assert specs["tf32x3@fp32"] == "tf32x3@fp32"
    assert specs["tf32@fp32"] == "tf32@fp32"


def test_default_latency_warmup_repeat() -> None:
    assert DEFAULT_LATENCY_WARMUP == 5
    assert DEFAULT_LATENCY_REPEAT == 20


def test_summarize_repeat_ms_sample_std() -> None:
    mean, std = summarize_repeat_ms([10.0, 12.0, 11.0])
    assert abs(mean - 11.0) < 1e-9
    assert abs(std - 1.0) < 1e-9
    mean_one, std_one = summarize_repeat_ms([7.5])
    assert mean_one == 7.5
    assert std_one == 0.0
