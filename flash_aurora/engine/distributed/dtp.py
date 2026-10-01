"""BEAST-style 4D parallel inference for Aurora: entry point and preconditions.

:func:`apply_domain_tensor_parallel` rewires a loaded model so that, on every rank of
a :class:`ProcessMesh`, ``model.forward(batch)`` runs the Perceiver encoder on this
rank's longitude shard (D5), the Swin backbone on a ``1/c x 1/s`` shard (D1, D2, D3),
and the decoder on the longitude shard, then all-gathers the prediction. The
standard rollout loop drives the result unchanged.

Scope:

* Every Aurora model in the engine runs: the 0.25 and 0.1 degree family (pretrained,
  fine-tuned with LoRA, air pollution, wave) and Aurora 1.5, including the stochastic
  ensemble backbone. Longitude shards are merge-aligned per level, so grids whose
  latent widths are odd at some level (0.4 degree air pollution) split too.
* The model only uses the channel and spatial axes. Uncertainty ranks ``u`` each run
  a whole DTP instance on their own ensemble members; the member seeds and the
  inference-time moment reduce (D4, :mod:`ensemble_moments`) live with the rollout
  driver. Data ranks ``d`` are independent forecasts.
* Every precision tier runs. Inside each shard the tier's kernels run unchanged:
  CuTe window attention on the local windows and heads, Triton GELU, the BF16/TF32
  ``F.linear`` routing, and backbone autocast. Two fused kernels assume an unsharded
  tensor and are replaced: the Triton roll/pad/partition layout, by the PyTorch
  layout plus the window-column exchange, and the Triton AdaLN, by the channel-group
  LayerNorm (D1). A 4D run at tier X is therefore tier X without those two fusions,
  and its single-device twin under the same tier label differs by that fusion as
  well as by reassociation. Collective outputs are FP32 at every tier.
* A ``torch.compile``-wrapped backbone is rejected; the DTP wrappers need the eager
  module tree.
"""

from __future__ import annotations

from typing import Any

from flash_aurora.engine.distributed.dtp_backbone import DomainTensorParallelBackbone
from flash_aurora.engine.distributed.perceiver_spatial import (
    LongitudeShardedDecoder,
    LongitudeShardedEncoder,
)
from flash_aurora.engine.distributed.process_mesh import ProcessMesh

_DTP_MESH_ATTR = "_flash_aurora_dtp_mesh"


def apply_domain_tensor_parallel(model: Any, mesh: ProcessMesh) -> Any:
    """Shard ``model`` over the channel and spatial axes of ``mesh`` in place.

    Call after the checkpoint is loaded and before the model is moved to its device.
    """
    if is_domain_tensor_parallel(model):
        raise RuntimeError("model is already domain-tensor-parallel")
    if getattr(model, "compile_backbone", False):
        raise ValueError("4D inference needs an eager backbone; build the model without compile_backbone")
    if hasattr(model, "clear_inference_cuda_graph"):
        model.clear_inference_cuda_graph()

    num_levels = model.backbone.num_encoder_layers
    model.encoder = LongitudeShardedEncoder(model.encoder, mesh, num_levels)
    model.backbone = DomainTensorParallelBackbone(model.backbone, mesh)
    model.decoder = LongitudeShardedDecoder(model.decoder, mesh, num_levels)
    setattr(model, _DTP_MESH_ATTR, mesh)
    return model


def is_domain_tensor_parallel(model: Any) -> bool:
    return getattr(model, _DTP_MESH_ATTR, None) is not None
