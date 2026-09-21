"""One-step bring-up profile: mixed precision, numbers only."""

from __future__ import annotations

from bench_one_step_profile import PROFILE_PRECISION, format_profile_table


def test_profile_uses_fast_mixed_precision() -> None:
    assert PROFILE_PRECISION == "bf16_mixed@fp32"


def test_profile_table_has_bringup_columns() -> None:
    lines = format_profile_table(
        [
            {
                "preset": "era5_pretrained",
                "grid": "721x1440",
                "data_read_s": 1.5,
                "construct_s": 0.4,
                "checkpoint_s": 8.0,
                "model_to_device_s": 2.0,
                "ic_to_device_s": 0.3,
                "first_step_s": 12.0,
                "warmed_step_ms": 676.3,
                "peak_gib": 26.6,
            }
        ]
    )
    header = lines[0]
    assert "data read" in header
    assert "first step" in header
    assert "warmed step" in header
    assert "era5_pretrained" in lines[2]
    assert "676.3" in lines[2]
