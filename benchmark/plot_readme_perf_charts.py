#!/usr/bin/env python3
"""Regenerate README performance bar charts from published benchmark numbers.

Outputs SVG and PNG under docs/image/. Source numbers match docs/benchmarks.md
(isolate-tiers on RTX PRO 6000 Blackwell unless noted).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "image"

C_CUTE_BF16 = "#0D7377"
C_SDPA_BF16 = "#B0BEC5"
C_CUTE_TF32 = "#C45C26"
C_SDPA_FP32 = "#90A4AE"

# Same family as window-attention bars (teal / terracotta / slate).
MODEL_COLORS = {
    "era5_pretrained": C_CUTE_BF16,
    "aurora_v1p5": "#14919B",
    "hres_t0_finetuned": C_CUTE_TF32,
    "hres_0.1": C_SDPA_FP32,
    "cams": "#78909C",
    "tc_tracking": "#5C7A86",
    "aurora_v1p5_ensemble": "#8C4A2F",
}

# Microsoft Aurora names from docs/models.md (not Flash-Aurora preset tokens).
MODEL_DISPLAY = {
    "era5_pretrained": "Aurora 0.25° Pretrained",
    "aurora_v1p5": "Aurora 1.5",
    "aurora_v1p5_ensemble": "Aurora 1.5 Ensemble",
    "hres_t0_finetuned": "Aurora 0.25° Fine-Tuned",
    "hres_0.1": "Aurora 0.1° Fine-Tuned",
    "cams": "Aurora 0.4° Air Pollution",
    "tc_tracking": "Aurora 0.25° Fine-Tuned (TC)",
}

# X-axis: five custom tiers + two PyTorch baselines.
TIER_LABELS = [
    "bf16_mixed\n@fp32",
    "bf16_mixed\n@tf32",
    "tf32\n@fp32",
    "tf32\n@tf32",
    "fp32\n@fp32",
    "PyTorch\nautocast",
    "PyTorch\nFP32 ref",
]

# Forward latency (ms), isolate-tiers. Finetuned presets use lora_merged.
# Order matches TIER_LABELS. Omits small_pretrained (scale).
# aurora_v1p5 from benchmark/latency_aurora_v1p5_latest.md
# aurora_v1p5_ensemble from benchmark/latency_aurora_v1p5_ensemble_latest.md
MODEL_LATENCY_MS = {
    "era5_pretrained": [676.4, 676.8, 1077.5, 919.2, 1945.0, 1004.4, 2128.2],
    "aurora_v1p5": [700.8, 702.4, 1109.3, 946.0, 2005.6, 1034.5, 2185.8],
    "aurora_v1p5_ensemble": [1004.9, 1006.6, 1411.9, 1248.7, 2450.8, 1137.1, 2535.1],
    "hres_t0_finetuned": [638.7, 638.4, 1006.3, 846.5, 1890.4, 967.7, 2061.9],
    "hres_0.1": [672.0, 672.4, 1019.9, 861.3, 1838.0, 986.2, 1994.6],
    "cams": [571.0, 571.9, 916.5, 718.3, 1562.3, 888.6, 1691.6],
    "tc_tracking": [638.5, 638.3, 1006.1, 847.0, 1890.9, 967.4, 2059.9],
}


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 18,
            "axes.labelsize": 12,
            "axes.titlepad": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.2,
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    walk = Path(r"D:\Courses\capstone\Flash-Aurora-walkthrough\figures")
    walk.mkdir(parents=True, exist_ok=True)
    for dest in (OUT_DIR, walk):
        fig.savefig(dest / f"{stem}.svg")
        fig.savefig(dest / f"{stem}.png", dpi=160)
        print(f"wrote {dest / stem}.svg")
    plt.close(fig)


def _annotate_speedups(
    ax: plt.Axes,
    x: np.ndarray,
    cute: list[float],
    sdpa: list[float],
    color: str,
    fontsize: float = 12,
) -> None:
    """Place speedup labels in a fixed band above all bars (never over short bars)."""
    y_lo, y_hi = ax.get_ylim()
    speedup_y = y_lo + 0.92 * (y_hi - y_lo)
    for i, (a, b) in enumerate(zip(cute, sdpa)):
        ax.text(
            x[i],
            speedup_y,
            f"{b / a:.2f}x",
            ha="center",
            va="top",
            fontsize=fontsize,
            fontweight="bold",
            color=color,
            clip_on=False,
        )


def _bar_values(
    ax: plt.Axes,
    bars,
    vals: list[float],
    y_span: float,
    fontsize: float = 10,
) -> None:
    pad = 0.015 * y_span
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + pad,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="#333333",
        )


def plot_window_attention() -> None:
    """CuTe BF16 and TF32-acc operators vs PyTorch SDPA (Blackwell sm_120a).

    Layout tensors are (B, H, N, D_h): B = folded window batch,
    H = heads, N = tokens per window, D_h = head dim. Masked bars use
    Swin shifted-window bias -100 on the largest ERA5 encoder shape.
    """
    # Concrete Q/K/V shapes (B, H, N, D_h) from 0.25-degree ERA5 encoder
    categories = [
        "(1800, 8, 144, 64)\nunmasked",
        "(1800, 8, 144, 64)\nmasked",
        "(450, 16, 144, 64)\nunmasked",
        "(128, 32, 144, 64)\nunmasked",
    ]
    bf16_cute = [0.727, 0.829, 0.374, 0.220]
    bf16_sdpa = [0.780, 1.014, 0.407, 0.239]
    tf32_cute = [1.613, 1.906, 0.819, 0.477]
    fp32_sdpa = [2.582, 3.221, 1.308, 0.760]

    x = np.arange(len(categories))
    w = 0.18
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 9.0))

    ax = axes[0]
    b1 = ax.bar(x - w, bf16_cute, w * 1.7, label="CuTe DSL BF16", color=C_CUTE_BF16, zorder=3)
    b2 = ax.bar(x + w, bf16_sdpa, w * 1.7, label="PyTorch SDPA BF16", color=C_SDPA_BF16, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=13)
    ax.set_ylabel("Latency (ms)", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_title("BF16 window attention", fontsize=20, pad=12)
    y_hi = max(bf16_cute + bf16_sdpa) * 1.62
    ax.set_ylim(0, y_hi)
    _bar_values(ax, b1, bf16_cute, y_hi, fontsize=11)
    _bar_values(ax, b2, bf16_sdpa, y_hi, fontsize=11)
    _annotate_speedups(ax, x, bf16_cute, bf16_sdpa, C_CUTE_BF16, fontsize=13)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=2,
        fontsize=13,
    )

    ax = axes[1]
    b1 = ax.bar(x - w, tf32_cute, w * 1.7, label="CuTe DSL TF32", color=C_CUTE_TF32, zorder=3)
    b2 = ax.bar(x + w, fp32_sdpa, w * 1.7, label="PyTorch SDPA FP32", color=C_SDPA_FP32, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=13)
    ax.set_ylabel("Latency (ms)", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_title("TF32-acc FP32 window attention", fontsize=20, pad=12)
    y_hi = max(tf32_cute + fp32_sdpa) * 1.58
    ax.set_ylim(0, y_hi)
    _bar_values(ax, b1, tf32_cute, y_hi, fontsize=11)
    _bar_values(ax, b2, fp32_sdpa, y_hi, fontsize=11)
    _annotate_speedups(ax, x, tf32_cute, fp32_sdpa, C_CUTE_TF32, fontsize=13)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=2,
        fontsize=13,
    )

    fig.tight_layout(h_pad=2.0)
    _save(fig, "window_attn_cute_vs_sdpa_blackwell")


def plot_e2e_latency_by_tier() -> None:
    """Grouped bars: x = precision tiers, color = model, y = forward latency (ms).

    A dashed horizontal line marks each model's baseline (the "PyTorch FP32
    ref" tier, last entry of TIER_LABELS) so every other tier's bar can be
    read as faster/slower than that model without cross-referencing values.
    """
    models = list(MODEL_LATENCY_MS.keys())
    n_tiers = len(TIER_LABELS)
    n_models = len(models)
    x = np.arange(n_tiers)
    width = 0.78 / n_models
    baseline_tier_index = n_tiers - 1

    fig, ax = plt.subplots(figsize=(13.2, 5.8))
    for i, name in enumerate(models):
        offset = (i - (n_models - 1) / 2.0) * width
        vals = MODEL_LATENCY_MS[name]
        ax.bar(
            x + offset,
            vals,
            width * 0.92,
            label=MODEL_DISPLAY[name],
            color=MODEL_COLORS[name],
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        ax.hlines(
            vals[baseline_tier_index],
            xmin=x[0] - 0.42,
            xmax=x[baseline_tier_index] + offset - width * 0.46,
            color=MODEL_COLORS[name],
            linestyle="--",
            linewidth=1.3,
            alpha=0.75,
            zorder=2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(TIER_LABELS, fontsize=13)
    ax.set_ylabel("One-step forward latency (ms)", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    y_max = max(max(v) for v in MODEL_LATENCY_MS.values())
    ax.set_ylim(0, y_max * 1.12)
    ax.set_xlim(x[0] - 0.42, x[-1] + 0.42)
    ax.set_title(
        "End-to-end latency by precision tier (single rollout step / model.forward)",
        fontsize=18,
        pad=16,
    )
    ax.text(
        x[0] - 0.42,
        y_max * 1.10,
        "dashed = each model's PyTorch FP32 ref baseline",
        ha="left",
        va="top",
        fontsize=10.5,
        color="#555555",
        style="italic",
    )
    ax.legend(
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        fontsize=9,
    )
    fig.tight_layout()
    _save(fig, "e2e_latency_by_tier_all_presets")


def main() -> None:
    _style()
    plot_window_attention()
    plot_e2e_latency_by_tier()


if __name__ == "__main__":
    main()
