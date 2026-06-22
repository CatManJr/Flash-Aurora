#!/usr/bin/env python3
"""Trace backbone hidden-state drift vs fp32 on AuroraSmallPretrained.

Encoder/decoder (Perceiver) stay FP32 for all inference presets
(``autocast_encoder_decoder=False``). This script verifies encoder outputs match
across tiers, then records how token-space error grows through each
Swin encoder/decoder stage in the backbone.

Examples::

    uv run python benchmark/bench_backbone_error_accum.py
    uv run python benchmark/bench_backbone_error_accum.py --tier tf32
    uv run python benchmark/bench_backbone_error_accum.py --tier bf16_mixed --per-block
"""

from __future__ import annotations

import argparse
import gc
import os
import pickle
import sys
import warnings
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
import _bootstrap  # noqa: F401, E402
from _asset_root import default_asset_root


import torch

_REPO = Path(__file__).resolve().parents[1]

from _asset_root import default_asset_root

_DEFAULT_DATA_DIR = str(default_asset_root())
_CHECKPOINT_NAME = "aurora-0.25-small-pretrained.ckpt"
_INPUT_NAME = "aurora-0.25-small-pretrained-test-input.pickle"
_STATIC_NAME = "aurora-0.25-static.pickle"


@dataclass(frozen=True)
class TensorDiff:
    max_abs: float
    mean_abs: float
    rel_l2: float
    cos_sim: float


@dataclass(frozen=True)
class TracePoint:
    name: str
    tensor: torch.Tensor


def _purge_gpu(*objs: Any) -> None:
    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _load_batch(data_dir: Path) -> Any:
    from flash_aurora.aurora import Batch, Metadata
    from flash_aurora.aurora.batch import interpolate_numpy

    import numpy as np

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with open(data_dir / _INPUT_NAME, "rb") as f:
            test_input = pickle.load(f)
        with open(data_dir / _STATIC_NAME, "rb") as f:
            static_vars = pickle.load(f)

    static_vars = {
        k: interpolate_numpy(
            v,
            np.linspace(90, -90, v.shape[0]),
            np.linspace(0, 360, v.shape[1], endpoint=False),
            test_input["metadata"]["lat"],
            test_input["metadata"]["lon"],
        )
        for k, v in static_vars.items()
    }

    return Batch(
        surf_vars={k: torch.from_numpy(v) for k, v in test_input["surf_vars"].items()},
        static_vars={k: torch.from_numpy(v) for k, v in static_vars.items()},
        atmos_vars={k: torch.from_numpy(v) for k, v in test_input["atmos_vars"].items()},
        metadata=Metadata(
            lat=torch.from_numpy(test_input["metadata"]["lat"]),
            lon=torch.from_numpy(test_input["metadata"]["lon"]),
            atmos_levels=tuple(test_input["metadata"]["atmos_levels"]),
            time=tuple(test_input["metadata"]["time"]),
        ),
    )


def _build_model(precision: str, checkpoint: Path, device: torch.device) -> Any:
    from flash_aurora.aurora import AuroraSmallPretrained

    model = AuroraSmallPretrained(use_lora=False, inference_precision=precision)
    model.load_checkpoint_local(str(checkpoint), strict=True)
    model.eval()
    return model.to(device)


def _tensor_diff(reference: torch.Tensor, candidate: torch.Tensor) -> TensorDiff:
    ref = reference.detach().float().cpu().flatten().double()
    cand = candidate.detach().float().cpu().flatten().double()
    diff = cand - ref
    max_abs = float(diff.abs().max().item())
    mean_abs = float(diff.abs().mean().item())
    ref_norm = ref.norm()
    rel_l2 = float(diff.norm().item() / ref_norm.item()) if ref_norm.item() > 0 else 0.0
    cand_norm = cand.norm()
    if ref_norm.item() == 0.0 and cand_norm.item() == 0.0:
        cos_sim = 1.0
    elif ref_norm.item() == 0.0 or cand_norm.item() == 0.0:
        cos_sim = 0.0
    else:
        cos_sim = float(torch.dot(ref, cand).item() / (ref_norm.item() * cand_norm.item()))
    return TensorDiff(max_abs=max_abs, mean_abs=mean_abs, rel_l2=rel_l2, cos_sim=cos_sim)



def _run_encoder(model: Any, batch: Any) -> tuple[tuple[int, int, int], torch.Tensor]:
    from flash_aurora.aurora.model.custom_op_paths import run_with_encoder_decoder_autocast

    _, transformed, patch_res = model._prepare_encoder_batch(batch)
    with torch.inference_mode():
        x = run_with_encoder_decoder_autocast(
            model.encoder,
            transformed,
            enabled=model.autocast_encoder_decoder,
            lead_time=model.timestep,
        )
    return patch_res, x


def _trace_backbone(
    model: Any,
    x: torch.Tensor,
    *,
    patch_res: tuple[int, int, int],
    rollout_step: int,
    per_block: bool,
) -> dict[str, torch.Tensor]:
    backbone = model.backbone
    points: dict[str, torch.Tensor] = {"backbone_in": x.detach().clone()}
    hooks: list[Any] = []

    def _save(name: str):
        def hook(_mod: torch.nn.Module, _inp: tuple[Any, ...], out: Any) -> None:
            tensor = out[0] if isinstance(out, tuple) else out
            points[name] = tensor.detach().clone()

        return hook

    if per_block:
        for stage_i, layer in enumerate(backbone.encoder_layers):
            for block_i, block in enumerate(layer.blocks):
                hooks.append(block.register_forward_hook(_save(f"enc{stage_i}.blk{block_i}")))
        for stage_i, layer in enumerate(backbone.decoder_layers):
            for block_i, block in enumerate(layer.blocks):
                hooks.append(block.register_forward_hook(_save(f"dec{stage_i}.blk{block_i}")))

    orig_forward = backbone.forward

    def traced_forward(
        tokens: torch.Tensor,
        lead_time: timedelta,
        rollout_step: int,
        patch_res: tuple[int, int, int],
    ) -> torch.Tensor:
        from flash_aurora.aurora.model.fourier import lead_time_expansion

        all_enc_res, padded_outs = backbone.get_encoder_specs(patch_res)
        lead_hours = lead_time / timedelta(hours=1)
        lead_times = lead_hours * torch.ones(tokens.shape[0], dtype=torch.float32, device=tokens.device)
        c = backbone.time_mlp(lead_time_expansion(lead_times, backbone.embed_dim))

        skips: list[torch.Tensor | None] = []
        for stage_i, layer in enumerate(backbone.encoder_layers):
            tokens, x_unscaled = layer(tokens, c, all_enc_res[stage_i], rollout_step=rollout_step)
            skips.append(x_unscaled)
            points[f"enc{stage_i}_out"] = tokens.detach().clone()

        for stage_i, layer in enumerate(backbone.decoder_layers):
            index = backbone.num_decoder_layers - stage_i - 1
            tokens, _ = layer(
                tokens,
                c,
                all_enc_res[index],
                padded_outs[index - 1],
                rollout_step=rollout_step,
            )
            if 0 < stage_i < backbone.num_decoder_layers - 1:
                tokens = tokens + skips[index - 1]
                points[f"dec{stage_i}_skip_add"] = tokens.detach().clone()
            elif stage_i == backbone.num_decoder_layers - 1:
                tokens = torch.cat([tokens, skips[0]], dim=-1)
                points[f"dec{stage_i}_concat"] = tokens.detach().clone()
            else:
                points[f"dec{stage_i}_out"] = tokens.detach().clone()

        points["backbone_out"] = tokens.detach().clone()
        return tokens

    backbone.forward = traced_forward  # type: ignore[method-assign]
    try:
        with torch.inference_mode():
            out = model._run_backbone(
                x,
                lead_time=model.timestep,
                patch_res=patch_res,
                rollout_step=rollout_step,
            )
        points["backbone_out_to_decoder"] = out.detach().clone()
    finally:
        backbone.forward = orig_forward  # type: ignore[method-assign]
        for hook in hooks:
            hook.remove()

    return points


def _decoder_msl_diff(model: Any, backbone_x: torch.Tensor, batch: Any, patch_res: tuple[int, int, int]) -> TensorDiff:
    from flash_aurora.aurora.model.custom_op_paths import run_with_encoder_decoder_autocast

    with torch.inference_mode():
        pred = run_with_encoder_decoder_autocast(
            model.decoder,
            backbone_x,
            batch,
            enabled=model.autocast_encoder_decoder,
            lead_time=model.timestep,
            patch_res=patch_res,
        )
    return pred.surf_vars["msl"].detach()


def _print_diff_table(rows: list[tuple[str, TensorDiff]]) -> None:
    print(f"\n{'checkpoint':<22} {'max_abs':>10} {'mean_abs':>10} {'rel_l2':>10} {'cos_sim':>12}")
    print("-" * 70)
    for name, d in rows:
        print(f"{name:<22} {d.max_abs:10.4g} {d.mean_abs:10.4g} {d.rel_l2:10.4g} {d.cos_sim:12.10f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(_DEFAULT_DATA_DIR))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--tier", default="bf16_mixed", help="Candidate tier vs fp32 (default: bf16_mixed)")
    parser.add_argument("--per-block", action="store_true", help="Record every Swin block inside each stage")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    device = torch.device("cuda")
    data_dir = args.data_dir.expanduser().resolve()
    checkpoint = (args.checkpoint or data_dir / _CHECKPOINT_NAME).expanduser().resolve()
    batch = _load_batch(data_dir).to(device)
    rollout_step = batch.metadata.rollout_step

    print(f"[config] grid={batch.metadata.lat.numel()}x{batch.metadata.lon.numel()} tier={args.tier}")
    print("[config] autocast_encoder_decoder=False for all presets → Perceiver stays FP32")

    ref_model = _build_model("fp32", checkpoint, device)
    cand_model = _build_model(args.tier, checkpoint, device)

    prepared, _, _ = ref_model._prepare_encoder_batch(batch)
    patch_res, x_ref = _run_encoder(ref_model, batch)
    _, x_cand = _run_encoder(cand_model, batch)
    enc_diff = _tensor_diff(x_ref, x_cand)
    print(
        f"\n[encoder] fp32 vs {args.tier}: max_abs={enc_diff.max_abs:.4g} "
        f"rel_l2={enc_diff.rel_l2:.4g} cos_sim={enc_diff.cos_sim:.10f}"
    )

    ref_pts = _trace_backbone(
        ref_model, x_ref, patch_res=patch_res, rollout_step=rollout_step, per_block=args.per_block
    )
    cand_pts = _trace_backbone(
        cand_model, x_cand, patch_res=patch_res, rollout_step=rollout_step, per_block=args.per_block
    )

    rows: list[tuple[str, TensorDiff]] = []
    for name in ref_pts:
        if name not in cand_pts:
            continue
        rows.append((name, _tensor_diff(ref_pts[name], cand_pts[name])))
    _print_diff_table(rows)

    ref_bb = ref_pts["backbone_out_to_decoder"]
    cand_bb = cand_pts["backbone_out_to_decoder"]
    msl_ref = _decoder_msl_diff(ref_model, ref_bb, prepared, patch_res)
    msl_cand = _decoder_msl_diff(cand_model, cand_bb, prepared, patch_res)
    msl_e2e = _tensor_diff(msl_ref, msl_cand)

    # Decoder amplification: feed both decoders the *same* fp32 backbone output.
    msl_cand_same_bb = _decoder_msl_diff(cand_model, ref_bb, prepared, patch_res)
    msl_backbone_only = _tensor_diff(msl_ref, msl_cand_same_bb)

    print(
        f"\n[decoder] msl with native backbone inputs: max_abs={msl_e2e.max_abs:.4g} "
        f"rel_l2={msl_e2e.rel_l2:.4g} cos_sim={msl_e2e.cos_sim:.10f}"
    )
    print(
        f"[decoder] msl with fp32 backbone fed to both (isolates decoder): max_abs={msl_backbone_only.max_abs:.4g} "
        f"rel_l2={msl_backbone_only.rel_l2:.4g} cos_sim={msl_backbone_only.cos_sim:.10f}"
    )

    _purge_gpu(ref_model, cand_model, batch)


if __name__ == "__main__":
    main()
