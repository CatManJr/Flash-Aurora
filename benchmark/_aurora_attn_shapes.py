"""Derive window-attention (Bwin, H, N, Dh) for every Aurora checkpoint variant.

Used by ``bench_window_attn.py`` to verify CuTe kernel coverage across all
production model shapes (not just 0.25° ERA5).
"""

from __future__ import annotations

import inspect
import os
import sys
from dataclasses import dataclass

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_AURORA_ROOT = os.path.join(_REPO_ROOT, "aurora")
for _p in (_REPO_ROOT, _AURORA_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from aurora.model.aurora import (  # noqa: E402
    Aurora,
    Aurora12hPretrained,
    AuroraAirPollution,
    AuroraHighRes,
    AuroraPretrained,
    AuroraSmallPretrained,
    AuroraWave,
)
from engine.core.presets import VARIANTS  # noqa: E402

# (preset name, model class)
CHECKPOINT_VARIANTS: tuple[tuple[str, type], ...] = (
    ("aurora-0.25-pretrained", AuroraPretrained),
    ("aurora-0.25-small-pretrained", AuroraSmallPretrained),
    ("aurora-0.25-finetuned", Aurora),
    ("aurora-0.25-12h-pretrained", Aurora12hPretrained),
    ("aurora-0.1-finetuned", AuroraHighRes),
    ("aurora-0.4-air-pollution", AuroraAirPollution),
    ("aurora-0.25-wave", AuroraWave),
)

DEFAULT_WINDOW_SIZE = (2, 6, 12)


@dataclass(frozen=True)
class AttnShape:
    """One unique window-attention geometry shared by one or more checkpoint sites."""

    bwin: int
    heads: int
    n_tokens: int
    head_dim: int
    label: str
    variants: tuple[str, ...]


def _init_defaults(cls: type) -> dict:
    sig = inspect.signature(cls.__init__)
    return {
        k: v.default
        for k, v in sig.parameters.items()
        if k != "self" and v.default is not inspect.Parameter.empty
    }


def _crop_hw(h: int, w: int, patch_size: int) -> tuple[int, int]:
    return h - (h % patch_size), w - (w % patch_size)


def _padded_res(res: tuple[int, int, int], ws: tuple[int, int, int]) -> tuple[int, int, int]:
    c, h, w = res
    pad = ((-c) % ws[0], (-h) % ws[1], (-w) % ws[2])
    return c + pad[0], h + pad[1], w + pad[2]


def bwin_from_res(
    res: tuple[int, int, int],
    ws: tuple[int, int, int],
    *,
    batch: int = 1,
) -> int:
    c, h, w = _padded_res(res, ws)
    return batch * (c // ws[0]) * (h // ws[1]) * (w // ws[2])


def encoder_stage_res(
    patch_res: tuple[int, int, int],
    num_stages: int = 3,
) -> list[tuple[int, int, int]]:
    stages = [patch_res]
    for _ in range(1, num_stages):
        c, h, w = stages[-1]
        stages.append((c, (h + h % 2) // 2, (w + w % 2) // 2))
    return stages


def iter_variant_attn_sites(
    variant_name: str,
    cls: type,
    *,
    batch: int = 1,
    include_decoder: bool = True,
) -> list[tuple[str, tuple[int, int, int, int], tuple[int, int, int]]]:
    """Yield (site_label, (Bwin,H,N,Dh), patch_res_at_stage) for one checkpoint."""
    kw = _init_defaults(cls)
    patch = kw.get("patch_size", 4)
    ws = kw.get("window_size", DEFAULT_WINDOW_SIZE)
    latent = kw.get("latent_levels", 4)
    embed = kw.get("embed_dim", 512)
    enc_heads = kw.get("encoder_num_heads", (8, 16, 32))
    dec_heads = kw.get("decoder_num_heads", (32, 16, 8))

    spec = VARIANTS[variant_name]
    h, w = _crop_hw(*spec.resolution, patch)
    patch_res = (latent, h // patch, w // patch)
    n_tokens = ws[0] * ws[1] * ws[2]
    enc_res = encoder_stage_res(patch_res, len(enc_heads))
    dec_res = list(reversed(enc_res))

    sites: list[tuple[str, tuple[int, int, int, int], tuple[int, int, int]]] = []
    for side, res_list, heads_list in (
        ("enc", enc_res, enc_heads),
        ("dec", dec_res, dec_heads),
    ):
        if side == "dec" and not include_decoder:
            continue
        for i, (res_i, heads) in enumerate(zip(res_list, heads_list)):
            stage_i = i if side == "enc" else (len(heads_list) - 1 - i)
            dim = embed * (2**stage_i)
            dh = dim // heads
            bwin = bwin_from_res(res_i, ws, batch=batch)
            sites.append(
                (
                    f"{variant_name}/{side}{i + 1}",
                    (bwin, heads, n_tokens, dh),
                    res_i,
                )
            )
    return sites


def all_unique_attn_shapes(
    *,
    batch: int = 1,
    include_decoder: bool = True,
) -> tuple[AttnShape, ...]:
    """Deduplicated shapes across all checkpoint variants."""
    by_key: dict[tuple[int, int, int, int], list[str]] = {}
    for variant_name, cls in CHECKPOINT_VARIANTS:
        for site, shape, _res in iter_variant_attn_sites(
            variant_name, cls, batch=batch, include_decoder=include_decoder
        ):
            by_key.setdefault(shape, []).append(site)

    out: list[AttnShape] = []
    for (bwin, heads, n_tokens, head_dim), sites in sorted(
        by_key.items(), key=lambda kv: (-kv[0][0], -kv[0][1], kv[0][2], kv[0][3])
    ):
        short = _short_label(bwin, heads, n_tokens, head_dim)
        out.append(
            AttnShape(
                bwin=bwin,
                heads=heads,
                n_tokens=n_tokens,
                head_dim=head_dim,
                label=short,
                variants=tuple(sorted(sites)),
            )
        )
    return tuple(out)


def _short_label(bwin: int, heads: int, n_tokens: int, head_dim: int) -> str:
    return f"Bwin={bwin} H={heads} N={n_tokens} Dh={head_dim}"


def shapes_for_benchmark(
    *,
    batch: int = 1,
    include_decoder: bool = True,
) -> list[tuple[int, int, int, int, str]]:
    """(Bwin, H, N, Dh, label) tuples for ``bench_window_attn.py``."""
    return [
        (s.bwin, s.heads, s.n_tokens, s.head_dim, s.label)
        for s in all_unique_attn_shapes(batch=batch, include_decoder=include_decoder)
    ]


# 0.25° ERA5 family - enc stages only (legacy name kept for perf section titles).
SHAPES_ERA5_025 = [
    (1800, 8, 144, 64, "era5 enc1 1800×8"),
    (450, 16, 144, 64, "era5 enc2 450×16"),
    (128, 32, 144, 64, "era5 enc3 128×32"),
]

SHAPES_ALL_CHECKPOINTS = shapes_for_benchmark()
