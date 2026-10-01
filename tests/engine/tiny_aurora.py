"""Small Auroras with random weights whose grid exercises every sharding edge case.

Latent grid ``(C, H, W) = (4, 6, 22)`` with window ``(2, 2, 4)`` and two longitude
shards gives three backbone levels with merge-aligned shards ``[0, 12, 22]``,
``[0, 6, 11]``, and ``[0, 3, 6]``:

* level 1, width 22: padded to 24 on both sides, six windows, unequal shards;
* level 2, width 11: odd, so the east-most rank pads the merge and crops the split;
  three windows over two shards, so one window straddles the shard boundary; an odd
  height (3) that needs latitude padding;
* level 3, width 6: padded to 8, two windows, and a height equal to the window,
  which turns the latitude shift off.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import torch

from flash_aurora.models.aurora import Batch, Metadata

PATCH_RES = (4, 6, 22)
WINDOW_SIZE = (2, 2, 4)
LEAD_TIME = timedelta(hours=6)
EMBED_DIM = 16
_PATCH_SIZE = 2
_LEVELS = (100, 250, 500, 850)
_BACKBONE_SHAPE = {
    "embed_dim": EMBED_DIM,
    "encoder_depths": (2, 2, 2),
    "encoder_num_heads": (2, 4, 8),
    "decoder_depths": (2, 2, 2),
    "decoder_num_heads": (8, 4, 2),
    "window_size": WINDOW_SIZE,
}
_MODEL_SHAPE = {
    **_BACKBONE_SHAPE,
    "latent_levels": PATCH_RES[0],
    "patch_size": _PATCH_SIZE,
    "num_heads": 2,
    "use_lora": False,
}
# Random weights everywhere: the checkpoint initialisation zeroes the AdaLN modulation,
# which would make every block an identity and hide sharding errors.
_WEIGHT_STD = 0.05


def _randomize(module: torch.nn.Module, seed: int) -> torch.nn.Module:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * _WEIGHT_STD)
    return module.eval()


def tiny_backbone(*, use_lora: bool, seed: int = 0) -> torch.nn.Module:
    """0.25/0.1 degree family backbone."""
    from flash_aurora.models.aurora.model.swin3d import Swin3DTransformerBackbone

    return _randomize(Swin3DTransformerBackbone(**_BACKBONE_SHAPE, use_lora=use_lora), seed)


def tiny_stochastic_backbone(seed: int = 0) -> torch.nn.Module:
    """Aurora 1.5 ensemble backbone: per-token noise and per-level context down-sampling."""
    from flash_aurora.models.aurora_v1p5.model.swin3d import Swin3DTransformerBackbone

    backbone = Swin3DTransformerBackbone(
        **_BACKBONE_SHAPE, stochastic=True, use_updated_lead_time_embedding=True
    )
    return _randomize(backbone, seed)


def backbone_tokens(seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    levels, height, width = PATCH_RES
    return torch.randn(1, levels * height * width, EMBED_DIM, generator=generator)


def longitude_shard_tokens(tokens: torch.Tensor, columns: slice) -> torch.Tensor:
    """Tokens ``(B, C*H*W, D)`` restricted to longitude ``columns``."""
    levels, height, width = PATCH_RES
    batch, _, channels = tokens.shape
    grid = tokens.view(batch, levels, height, width, channels)
    return grid[:, :, :, columns].reshape(batch, -1, channels)


def tiny_aurora(seed: int = 0) -> torch.nn.Module:
    """0.25/0.1 degree family model."""
    from flash_aurora.models.aurora.model.aurora import Aurora

    return _randomize(Aurora(**_MODEL_SHAPE), seed)


def tiny_aurora_ensemble(seed: int = 0) -> torch.nn.Module:
    """Aurora 1.5 family model with the stochastic backbone and variable lead times."""
    from flash_aurora.models.aurora_v1p5.model.aurora import Aurora

    model = Aurora(
        **_MODEL_SHAPE,
        stochastic=True,
        variable_lead_time=True,
        use_updated_lead_time_embedding=True,
    )
    return _randomize(model, seed)


def tiny_batch(seed: int = 2, *, latent_width: int = PATCH_RES[2]) -> Batch:
    """Batch on the latent grid ``(C, H, latent_width)``; widen it to host more spatial shards."""
    generator = torch.Generator().manual_seed(seed)
    height, width = PATCH_RES[1] * _PATCH_SIZE, latent_width * _PATCH_SIZE

    def field(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator)

    return Batch(
        surf_vars={name: field(1, 2, height, width) for name in ("2t", "10u", "10v", "msl")},
        static_vars={name: field(height, width) for name in ("lsm", "z", "slt")},
        atmos_vars={
            name: field(1, 2, len(_LEVELS), height, width) for name in ("z", "u", "v", "t", "q")
        },
        metadata=Metadata(
            lat=torch.linspace(90, -90, height),
            lon=torch.arange(width) * (360.0 / width),
            time=(datetime(2023, 1, 1, 6),),
            atmos_levels=_LEVELS,
        ),
    )
