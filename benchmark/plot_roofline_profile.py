#!/usr/bin/env python3
"""Roofline and Swin-block profiling figure for the technical brief.

Peaks: NVIDIA RTX PRO 6000 Blackwell Server Edition datasheet
(memory 1597 GB/s, FP32 120 TFLOPS, TF32 tensor 234 TFLOPS).
BF16 tensor is plotted at 500 TFLOPS dense (vendor 1 PFLOPS includes 2:4 sparsity).

Attention points: docs/benchmarks.md window-attn microbench (sm_120a).
AdaLN / GEMM / stage bars: CUDA-event timings from
``profiling/profiling_swin3d_block.py --preset aurora --patch-h 180 --patch-w 360``
on one encoder stage-1 Swin block. After crop and patch embedding, Aurora 0.25
Pretrained (patch 4 on 720 x 1440) and Aurora 0.1 Fine-Tuned (patch 10 on
1800 x 3600) share patch_res=(4, 180, 360), dim=512, heads=8, N=144.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "image"

# Device peaks (RTX PRO 6000 Blackwell Server Edition).
BW_GB_S = 1597.0
FP32_TFLOPS = 120.0
TF32_TFLOPS = 234.0
BF16_DENSE_TFLOPS = 500.0

# Window attention, Aurora 0.25 Pretrained enc1 (1800, 8, 144, 64).
# FLOPs = 4 B H N N Dh.
ATTN_GFLOP = 4 * 1800 * 8 * 144 * 144 * 64 / 1e9
ATTN_CUTE_BF16_MS = 0.727
ATTN_SDPA_BF16_MS = 0.780
ATTN_CUTE_TF32_MS = 1.613
ATTN_SDPA_FP32_MS = 2.582

# Stage-1 Swin block, fused bf16_mixed (Triton layout/AdaLN + CuTe).
FUSED_025_MS = {
    "GEMM (QKV/MLP/proj)": 8.5902 + 8.4673 + 8.4203 + 3.0649,
    "GELU": 2.8837,
    "AdaLN + residual": 1.0974 + 1.0721,
    "Window attention": 1.5107,
    "Window layout": 0.7771 + 0.7288,
}
EAGER_025_MS = {
    "GEMM (QKV/MLP/proj)": 8.5958 + 8.4728 + 8.4223 + 3.0689,
    "GELU": 2.8833,
    "AdaLN + residual": 3.2479 + 3.2188,
    "Window attention": 2.5928,
    "Window layout": 1.4872 + 0.7251,
}

# Aurora 0.1 Fine-Tuned: same stage-1 geometry; independently timed.
FUSED_01_MS = {
    "GEMM (QKV/MLP/proj)": 8.5905 + 8.4672 + 8.4211 + 3.0660,
    "GELU": 2.8833,
    "AdaLN + residual": 1.0973 + 1.0724,
    "Window attention": 1.5113,
    "Window layout": 0.7796 + 0.7284,
}
EAGER_01_MS = {
    "GEMM (QKV/MLP/proj)": 8.5955 + 8.4767 + 8.4203 + 3.0673,
    "GELU": 2.8828,
    "AdaLN + residual": 3.2470 + 3.2179,
    "Window attention": 2.5927,
    "Window layout": 1.4747 + 0.7261,
}

# AdaLN intensity: FiLM + residual, FP32 tensors, one fused kernel.
# FLOPs ~ 12 L D; bytes ~ 3 L D * 4 (x, residual, y).
L_TOKENS = 4 * 180 * 360
DIM = 512
ADALN_FLOP = 12.0 * L_TOKENS * DIM
ADALN_BYTES_FUSED = 3.0 * L_TOKENS * DIM * 4.0
ADALN_FUSED_MS = FUSED_025_MS["AdaLN + residual"] / 2.0
ADALN_EAGER_MS = EAGER_025_MS["AdaLN + residual"] / 2.0

# QKV linear: (M, K) x (K, N) with M=L, K=512, N=1536, FP32 I/O.
M, K, N_OUT = L_TOKENS, 512, 1536
QKV_FLOP = 2.0 * M * K * N_OUT
QKV_BYTES_FP32 = 4.0 * (M * K + K * N_OUT + M * N_OUT)
QKV_MS = 8.5902

C_CUTE = "#0D7377"
C_SDPA = "#90A4AE"
C_TF32 = "#C45C26"
C_ADALN = "#5C7A86"
C_GEMM = "#1B4F72"
C_EAGER = "#B0BEC5"

STAGE_LABELS = list(FUSED_025_MS.keys())
BAR_XLIM = 32.5


def _tflops(gflop: float, ms: float) -> float:
    return gflop / (ms * 1e-3) / 1e3


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.4,
            "axes.titlesize": 11,
            "axes.labelsize": 9.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.10,
        }
    )


def _block_bars(
    ax: plt.Axes,
    fused: dict[str, float],
    eager: dict[str, float],
    title: str,
    *,
    show_ylabel: bool,
    show_legend: bool,
) -> None:
    fused_vals = [fused[k] for k in STAGE_LABELS]
    eager_vals = [eager[k] for k in STAGE_LABELS]
    y = np.arange(len(STAGE_LABELS))
    h = 0.36
    ax.barh(y + h / 2, eager_vals, h, color=C_EAGER, label="Eager FP32 (TF32 GEMM)")
    ax.barh(y - h / 2, fused_vals, h, color=C_CUTE, label="Fused + CuTe")
    for yi, e, f in zip(y, eager_vals, fused_vals):
        ax.text(e + 0.18, yi + h / 2, f"{e:.1f}", va="center", fontsize=6.8, color="#555555")
        ax.text(f + 0.18, yi - h / 2, f"{f:.1f}", va="center", fontsize=6.8, color=C_CUTE)
    ax.set_yticks(y)
    ax.set_yticklabels(STAGE_LABELS if show_ylabel else [])
    ax.tick_params(axis="y", length=0 if not show_ylabel else 3)
    ax.invert_yaxis()
    ax.set_xlim(0, BAR_XLIM)
    ax.set_xlabel("CUDA-event time (ms)")
    ax.set_title(title)
    if show_legend:
        ax.legend(loc="lower right", fontsize=6.8, frameon=False)
    fused_total = sum(fused_vals)
    gemm_pct = 100.0 * fused["GEMM (QKV/MLP/proj)"] / fused_total
    attn_pct = 100.0 * fused["Window attention"] / fused_total
    ax.text(
        0.98,
        0.02,
        f"Fused: GEMM {gemm_pct:.0f}%  ·  attn {attn_pct:.0f}%",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        color="#333333",
    )


def plot() -> None:
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.95), width_ratios=[1.18, 1.0, 0.92])

    ax = axes[0]
    intensity = np.logspace(-0.4, 3.15, 400)
    bw_tflops = (BW_GB_S * 1e9 * intensity) / 1e12
    ax.plot(intensity, bw_tflops, color="#333333", lw=1.4, label="1.597 TB/s HBM")
    for y, ls, lab in (
        (FP32_TFLOPS, "--", "FP32 120 TFLOPS"),
        (TF32_TFLOPS, "-.", "TF32 234 TFLOPS"),
        (BF16_DENSE_TFLOPS, ":", "BF16 500 TFLOPS (dense)"),
    ):
        ax.axhline(y, color="#555555", ls=ls, lw=1.0, label=lab)

    points = [
        (
            ATTN_GFLOP * 1e9 / (8 * 1800 * 8 * 144 * 64),
            _tflops(ATTN_GFLOP, ATTN_CUTE_BF16_MS),
            "CuTe BF16 attn",
            C_CUTE,
            "o",
        ),
        (
            144 / 2,
            _tflops(ATTN_GFLOP, ATTN_SDPA_BF16_MS),
            "SDPA BF16",
            C_SDPA,
            "o",
        ),
        (
            144 / 4,
            _tflops(ATTN_GFLOP, ATTN_CUTE_TF32_MS),
            "CuTe TF32 attn",
            C_TF32,
            "s",
        ),
        (
            144 / 4,
            _tflops(ATTN_GFLOP, ATTN_SDPA_FP32_MS),
            "SDPA FP32",
            "#8D6E63",
            "s",
        ),
        (
            ADALN_FLOP / ADALN_BYTES_FUSED,
            _tflops(ADALN_FLOP / 1e9, ADALN_FUSED_MS),
            "Fused AdaLN",
            C_ADALN,
            "D",
        ),
        (
            ADALN_FLOP / (ADALN_BYTES_FUSED * 2.0),
            _tflops(ADALN_FLOP / 1e9, ADALN_EAGER_MS),
            "Eager AdaLN",
            C_EAGER,
            "D",
        ),
        (
            QKV_FLOP / QKV_BYTES_FP32,
            _tflops(QKV_FLOP / 1e9, QKV_MS),
            "QKV GEMM",
            C_GEMM,
            "^",
        ),
    ]
    for x, y, lab, color, marker in points:
        ax.scatter(
            [x],
            [y],
            s=38,
            c=color,
            marker=marker,
            zorder=5,
            label=lab,
            edgecolors="white",
            linewidths=0.4,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.25, 900)
    ax.set_ylim(0.15, 900)
    ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("Achieved (TFLOP/s)")
    ax.set_title("Roofline, RTX PRO 6000 Blackwell")
    ax.legend(loc="lower right", fontsize=6.4, frameon=False, ncol=1)

    _block_bars(
        axes[1],
        FUSED_025_MS,
        EAGER_025_MS,
        "Aurora 0.25° Pretrained, stage 1",
        show_ylabel=True,
        show_legend=True,
    )
    _block_bars(
        axes[2],
        FUSED_01_MS,
        EAGER_01_MS,
        "Aurora 0.1° Fine-Tuned, stage 1",
        show_ylabel=False,
        show_legend=False,
    )

    fig.tight_layout(w_pad=1.15)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / "roofline_swin_block.png"
    svg = OUT_DIR / "roofline_swin_block.svg"
    fig.savefig(svg)
    fig.savefig(png, dpi=170)
    Image.open(png).convert("RGB").save(png)
    print(f"wrote {png}")
    print(f"wrote {svg}")
    plt.close(fig)


if __name__ == "__main__":
    plot()
