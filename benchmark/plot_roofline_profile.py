#!/usr/bin/env python3
"""Roofline and Swin-block profiling figure for the technical brief.

Peaks: NVIDIA RTX PRO 6000 Blackwell Server Edition datasheet
(memory 1597 GB/s, FP32 120 TFLOPS, TF32 tensor 234 TFLOPS).
BF16 tensor is plotted at 500 TFLOPS dense (vendor 1 PFLOPS includes 2:4 sparsity).

Attention points: docs/benchmarks.md window-attn microbench (sm_120a), encoder stage 1.
Stage bars: CUDA-event timings from profiling/profiling_swin3d_block.py on one
encoder block per stage. Decoder stages reverse the same three grids.
Aurora 0.25 Pretrained and Aurora 0.1 Fine-Tuned share the grids after patch
embedding (patch 4 on 720 x 1440 vs patch 10 on 1800 x 3600).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "image"
STAGE_JSON = ROOT / "docs" / "brief" / "swin_encoder_stages.json"

# Device peaks (RTX PRO 6000 Blackwell Server Edition).
BW_GB_S = 1597.0
FP32_TFLOPS = 120.0
TF32_TFLOPS = 234.0
BF16_DENSE_TFLOPS = 500.0

# Window attention, Aurora 0.25 Pretrained enc1 (1800, 8, 144, 64).
ATTN_GFLOP = 4 * 1800 * 8 * 144 * 144 * 64 / 1e9
ATTN_CUTE_BF16_MS = 0.727
ATTN_SDPA_BF16_MS = 0.780
ATTN_CUTE_TF32_MS = 1.613
ATTN_SDPA_FP32_MS = 2.582

STAGE_LABELS = [
    "GEMM (QKV/MLP/proj)",
    "GELU",
    "AdaLN + residual",
    "Window attention",
    "Window layout",
]
GEMM_KEYS = ("qkv_linear", "mlp_fc1", "mlp_fc2", "proj_linear")
LAYOUT_KEYS = (
    "layout_partition",
    "layout_unmerge",
    "layout_flatten",
    "qkv_rearrange_split",
    "attn_output_layout",
)

# AdaLN / QKV intensity from encoder stage 1 (same L D^2 at deeper stages).
L_TOKENS = 4 * 180 * 360
DIM = 512
ADALN_FLOP = 12.0 * L_TOKENS * DIM
ADALN_BYTES_FUSED = 3.0 * L_TOKENS * DIM * 4.0
M, K, N_OUT = L_TOKENS, 512, 1536
QKV_FLOP = 2.0 * M * K * N_OUT
QKV_BYTES_FP32 = 4.0 * (M * K + K * N_OUT + M * N_OUT)

C_CUTE = "#0D7377"
C_SDPA = "#90A4AE"
C_TF32 = "#C45C26"
C_ADALN = "#5C7A86"
C_GEMM = "#1B4F72"
C_EAGER = "#B0BEC5"

BAR_XLIM = 33.0

PANEL_TITLES = {
    "enc1": "Enc 1 / Dec 3   (4, 180, 360)",
    "enc2": "Enc 2 / Dec 2   (4, 90, 180)",
    "enc3": "Enc 3 / Dec 1   (4, 45, 90)",
}


def _tflops(gflop: float, ms: float) -> float:
    return gflop / (ms * 1e-3) / 1e3


def _bucket(raw: dict[str, float]) -> dict[str, float]:
    return {
        "GEMM (QKV/MLP/proj)": sum(raw.get(k, 0.0) for k in GEMM_KEYS),
        "GELU": raw.get("mlp_gelu", 0.0),
        "AdaLN + residual": raw.get("residual_adaln1", 0.0) + raw.get("residual_adaln2", 0.0),
        "Window attention": raw.get("attention_core", 0.0),
        "Window layout": sum(raw.get(k, 0.0) for k in LAYOUT_KEYS),
    }


def _load_stages() -> dict[str, dict[str, dict[str, float]]]:
    payload = json.loads(STAGE_JSON.read_text(encoding="utf-8"))
    return {
        sid: {"fused": _bucket(row["fused"]), "eager": _bucket(row["eager"])}
        for sid, row in payload.items()
    }


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 10.2,
            "axes.labelsize": 9.0,
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
) -> None:
    fused_vals = [fused[k] for k in STAGE_LABELS]
    eager_vals = [eager[k] for k in STAGE_LABELS]
    y = np.arange(len(STAGE_LABELS))
    h = 0.36
    ax.barh(y + h / 2, eager_vals, h, color=C_EAGER, label="Eager FP32")
    ax.barh(y - h / 2, fused_vals, h, color=C_CUTE, label="Fused + CuTe DSL")
    for yi, e, f in zip(y, eager_vals, fused_vals):
        ax.text(e + 0.18, yi + h / 2, f"{e:.1f}", va="center", fontsize=6.4, color="#555555")
        ax.text(f + 0.18, yi - h / 2, f"{f:.1f}", va="center", fontsize=6.4, color=C_CUTE)
    ax.set_yticks(y)
    ax.set_yticklabels(STAGE_LABELS if show_ylabel else [])
    ax.tick_params(axis="y", length=0 if not show_ylabel else 3)
    ax.invert_yaxis()
    ax.set_xlim(0, BAR_XLIM)
    ax.set_xlabel("CUDA-event time (ms)")
    ax.set_title(title, fontsize=9.6)
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
        fontsize=6.8,
        color="#333333",
    )


def _roofline(ax: plt.Axes, fused_s1: dict[str, float], eager_s1: dict[str, float]) -> None:
    intensity = np.logspace(-0.4, 3.15, 400)
    bw_tflops = (BW_GB_S * 1e9 * intensity) / 1e12
    ax.plot(intensity, bw_tflops, color="#333333", lw=1.4, label="1.597 TB/s HBM")
    for y, ls, lab in (
        (FP32_TFLOPS, "--", "FP32 120 TFLOPS"),
        (TF32_TFLOPS, "-.", "TF32 234 TFLOPS"),
        (BF16_DENSE_TFLOPS, ":", "BF16 500 TFLOPS (dense)"),
    ):
        ax.axhline(y, color="#555555", ls=ls, lw=1.0, label=lab)

    adaln_fused_ms = fused_s1["AdaLN + residual"] / 2.0
    adaln_eager_ms = eager_s1["AdaLN + residual"] / 2.0
    qkv_ms = 8.5928

    points = [
        (
            ATTN_GFLOP * 1e9 / (8 * 1800 * 8 * 144 * 64),
            _tflops(ATTN_GFLOP, ATTN_CUTE_BF16_MS),
            "CuTe BF16 attn",
            C_CUTE,
            "o",
        ),
        (144 / 2, _tflops(ATTN_GFLOP, ATTN_SDPA_BF16_MS), "SDPA BF16", C_SDPA, "o"),
        (144 / 4, _tflops(ATTN_GFLOP, ATTN_CUTE_TF32_MS), "CuTe TF32 attn", C_TF32, "s"),
        (144 / 4, _tflops(ATTN_GFLOP, ATTN_SDPA_FP32_MS), "SDPA FP32", "#8D6E63", "s"),
        (
            ADALN_FLOP / ADALN_BYTES_FUSED,
            _tflops(ADALN_FLOP / 1e9, adaln_fused_ms),
            "Fused AdaLN",
            C_ADALN,
            "D",
        ),
        (
            ADALN_FLOP / (ADALN_BYTES_FUSED * 2.0),
            _tflops(ADALN_FLOP / 1e9, adaln_eager_ms),
            "Eager AdaLN",
            C_EAGER,
            "D",
        ),
        (
            QKV_FLOP / QKV_BYTES_FP32,
            _tflops(QKV_FLOP / 1e9, qkv_ms),
            "QKV GEMM FP32",
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
    qkv_x = QKV_FLOP / QKV_BYTES_FP32
    qkv_y = _tflops(QKV_FLOP / 1e9, qkv_ms)
    ax.annotate(
        "FP32",
        xy=(qkv_x, qkv_y),
        xytext=(6, 8),
        textcoords="offset points",
        fontsize=6.6,
        color=C_GEMM,
        ha="left",
        va="bottom",
        fontweight="bold",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.25, 900)
    ax.set_ylim(0.15, 900)
    ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("Achieved (TFLOP/s)")
    ax.set_title("Roofline, RTX PRO 6000 Blackwell")
    ax.legend(loc="lower right", fontsize=5.8, frameon=False, ncol=1)


def plot() -> None:
    _style()
    stages = _load_stages()

    fig, axes = plt.subplots(1, 4, figsize=(16.2, 3.72), width_ratios=[1.18, 1.0, 0.88, 0.88])
    _roofline(axes[0], stages["enc1"]["fused"], stages["enc1"]["eager"])
    _block_bars(
        axes[1],
        stages["enc1"]["fused"],
        stages["enc1"]["eager"],
        PANEL_TITLES["enc1"],
        show_ylabel=True,
    )
    _block_bars(
        axes[2],
        stages["enc2"]["fused"],
        stages["enc2"]["eager"],
        PANEL_TITLES["enc2"],
        show_ylabel=False,
    )
    _block_bars(
        axes[3],
        stages["enc3"]["fused"],
        stages["enc3"]["eager"],
        PANEL_TITLES["enc3"],
        show_ylabel=False,
    )
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.995, 1.04),
        ncol=2,
        frameon=False,
        fontsize=8.0,
        borderaxespad=0.0,
    )
    fig.tight_layout(w_pad=1.05, rect=(0.0, 0.0, 1.0, 0.93))
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
