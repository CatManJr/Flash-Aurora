#!/usr/bin/env python3
"""Stacked mean-relative-error charts from the published precision suite.

Parses benchmark/precision_all_seed42.md and precision_aurora_v1p5_latest.md.
Omits full ``bf16@*`` tiers. Colors follow the window-attention palette
(teal / terracotta / slate) used in plot_readme_perf_charts.py.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "image"
WALK_DIR = Path(r"D:\Courses\capstone\Flash-Aurora-walkthrough\figures")

PRECISION_SHARED = ROOT / "benchmark" / "precision_all_seed42.md"
PRECISION_V1P5 = ROOT / "benchmark" / "precision_aurora_v1p5_latest.md"
PRECISION_ENSEMBLE = ROOT / "benchmark" / "precision_aurora_v1p5_ensemble_latest.md"

# Window-attention palette anchors
C_TEAL = "#0D7377"
C_TEAL_LIGHT = "#14919B"
C_TERRACOTTA = "#C45C26"
C_SLATE = "#90A4AE"
C_SLATE_LIGHT = "#B0BEC5"
C_SLATE_DARK = "#5C7A86"

# Recommended / reference tiers only (no full bf16@*).
TIERS = [
    "bf16_mixed@fp32",
    "tf32@fp32",
    "fp32@fp32",
    "pytorch_backbone_autocast_bf16_encoder_decoder_fp32",
]

TIER_LABELS = [
    "bf16_mixed\n@fp32",
    "tf32\n@fp32",
    "fp32\n@fp32",
    "PyTorch\nautocast",
]

MODELS = [
    "era5_pretrained",
    "small_pretrained",
    "aurora_v1p5",
    "aurora_v1p5_ensemble",
    "hres_t0_finetuned",
    "tc_tracking",
    "hres_0.1",
    "cams",
]

MODEL_COLORS = {
    "era5_pretrained": C_TEAL,
    "small_pretrained": "#0F766E",
    "aurora_v1p5": C_TEAL_LIGHT,
    "aurora_v1p5_ensemble": "#8C4A2F",
    "hres_t0_finetuned": C_TERRACOTTA,
    "hres_0.1": C_SLATE,
    "cams": "#78909C",
    "tc_tracking": C_SLATE_DARK,
}

# Microsoft Aurora names from docs/models.md (not Flash-Aurora preset tokens).
MODEL_DISPLAY = {
    "era5_pretrained": "Aurora 0.25° Pretrained",
    "small_pretrained": "Aurora 0.25° Small",
    "aurora_v1p5": "Aurora 1.5",
    "aurora_v1p5_ensemble": "Aurora 1.5 Ensemble",
    "hres_t0_finetuned": "Aurora 0.25° Fine-Tuned",
    "hres_0.1": "Aurora 0.1° Fine-Tuned",
    "cams": "Aurora 0.4° Air Pollution",
    "tc_tracking": "Aurora 0.25° Fine-Tuned (TC)",
}

# High-contrast categorical colors (still in the teal / terracotta / slate family,
# plus a few cooler/warmer accents so adjacent stack segments stay readable).
VAR_COLORS = [
    "#0D7377",
    "#C45C26",
    "#3D5A80",
    "#B08900",
    "#14919B",
    "#8C4A2F",
    "#5C7A86",
    "#D97706",
    "#0F766E",
    "#7C6F64",
    "#2563EB",
    "#B45309",
    "#475569",
    "#CA8A04",
    "#0E7490",
    "#9A3412",
    "#64748B",
    "#A16207",
    "#115E59",
    "#78716C",
    "#1D4ED8",
    "#C2410C",
    "#334155",
    "#EAB308",
    "#155E75",
    "#92400E",
    "#6B7280",
    "#854D0E",
    "#164E63",
    "#7F1D1D",
    "#57534E",
]


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
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
    WALK_DIR.mkdir(parents=True, exist_ok=True)
    for dest in (OUT_DIR, WALK_DIR):
        fig.savefig(dest / f"{stem}.svg")
        fig.savefig(dest / f"{stem}.png", dpi=160)
        print(f"wrote {dest / stem}.svg")
    plt.close(fig)


def _parse_section(text: str, heading: str) -> tuple[list[str], dict[str, dict[str, float]]]:
    pattern = rf"## {re.escape(heading)}\n\n(\|.+?)(?=\n## |\Z)"
    match = re.search(pattern, text, re.S)
    if not match:
        raise ValueError(f"section not found: {heading}")
    lines = [ln for ln in match.group(1).splitlines() if ln.startswith("|")]
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    var_names = header[2:]
    rows: dict[str, dict[str, float]] = {}
    for line in lines[2:]:
        cells = [c.strip().replace("**", "") for c in line.strip("|").split("|")]
        tier = cells[0]
        vals = {name: float(raw) for name, raw in zip(var_names, cells[2:])}
        rows[tier] = vals
    return var_names, rows


def load_all() -> dict[str, tuple[list[str], dict[str, dict[str, float]]]]:
    shared = PRECISION_SHARED.read_text(encoding="utf-8")
    v1 = PRECISION_V1P5.read_text(encoding="utf-8")
    ens = PRECISION_ENSEMBLE.read_text(encoding="utf-8")
    out: dict[str, tuple[list[str], dict[str, dict[str, float]]]] = {}
    for name in MODELS:
        if name == "aurora_v1p5":
            src = v1
        elif name == "aurora_v1p5_ensemble":
            src = ens
        else:
            src = shared
        out[name] = _parse_section(src, name)
    return out


def _var_colors(n: int) -> list[str]:
    if n <= len(VAR_COLORS):
        return VAR_COLORS[:n]
    colors = list(VAR_COLORS)
    while len(colors) < n:
        colors.append(VAR_COLORS[len(colors) % len(VAR_COLORS)])
    return colors


def plot_stacked_by_model() -> None:
    data = load_all()
    fig, axes = plt.subplots(2, 4, figsize=(16.4, 8.6), sharey=False)
    axes_flat = axes.ravel()

    for ax, model in zip(axes_flat, MODELS):
        var_names, rows = data[model]
        colors = _var_colors(len(var_names))
        x = np.arange(len(TIERS))
        bottoms = np.zeros(len(TIERS))
        for vi, var in enumerate(var_names):
            heights = np.array([rows[tier][var] for tier in TIERS], dtype=float)
            ax.bar(
                x,
                heights,
                bottom=bottoms,
                width=0.72,
                color=colors[vi],
                edgecolor="white",
                linewidth=0.3,
                label=var,
            )
            bottoms += heights

        ax.set_xticks(x)
        ax.set_xticklabels(TIER_LABELS, fontsize=8)
        ax.set_title(MODEL_DISPLAY[model], fontsize=11, pad=8, color=MODEL_COLORS[model])
        ax.set_ylabel("mean relative error")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, -2))

        n_vars = len(var_names)
        ncol = 4 if n_vars > 12 else 3
        fontsize = 5.5 if n_vars > 12 else 6.5
        ax.legend(
            frameon=False,
            fontsize=fontsize,
            ncol=ncol,
            loc="upper left",
            bbox_to_anchor=(0.0, -0.30),
            handlelength=0.9,
            columnspacing=0.8,
            handletextpad=0.3,
        )

    fig.suptitle(
        "Stacked mean relative error vs PyTorch FP32 baseline (seed 42)",
        fontsize=16,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    # Extra vertical gap so row-1 legends do not cover row-2 titles/bars.
    fig.subplots_adjust(hspace=0.98, wspace=0.32)
    _save(fig, "precision_mean_rel_stacked_by_model")


def plot_stacked_bf16_mixed() -> None:
    data = load_all()
    tier = "bf16_mixed@fp32"

    all_vars: list[str] = []
    for model in MODELS:
        for v in data[model][0]:
            if v not in all_vars:
                all_vars.append(v)

    colors = _var_colors(len(all_vars))
    var_color = {v: colors[i] for i, v in enumerate(all_vars)}

    fig, ax = plt.subplots(figsize=(14.0, 6.4))
    x = np.arange(len(MODELS))
    bottoms = np.zeros(len(MODELS))

    for var in all_vars:
        heights = np.array(
            [data[model][1][tier].get(var, 0.0) for model in MODELS],
            dtype=float,
        )
        if float(heights.sum()) <= 0:
            continue
        ax.bar(
            x,
            heights,
            bottom=bottoms,
            width=0.7,
            color=var_color[var],
            edgecolor="white",
            linewidth=0.25,
            label=var,
        )
        bottoms += heights

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY[m] for m in MODELS], rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("Stacked mean relative error")
    ax.set_title(
        "bf16_mixed@fp32: stacked per-variable mean_rel vs FP32 baseline (seed 42)",
        fontsize=15,
        pad=12,
    )
    ax.legend(
        frameon=False,
        fontsize=7,
        ncol=6,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
    )
    fig.tight_layout()
    _save(fig, "precision_mean_rel_stacked_bf16_mixed")


def main() -> None:
    _style()
    plot_stacked_by_model()
    plot_stacked_bf16_mixed()


if __name__ == "__main__":
    main()
