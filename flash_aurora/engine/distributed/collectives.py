"""FP32 collectives for tensor-parallel and domain-tensor-parallel inference.

Every partial sum that crosses ranks is promoted to FP32 before the reduction, so
the only numerical change a parallel scheme introduces is the order of the sum.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def all_reduce_sum_fp32(partial: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Sum ``partial`` over ``group`` in FP32 and return the full sum on every rank."""
    total = partial.to(torch.float32).contiguous()
    dist.all_reduce(total, op=dist.ReduceOp.SUM, group=group)
    return total


def reduce_scatter_last_dim_fp32(partial: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Sum ``partial`` over ``group`` in FP32; rank ``r`` keeps chunk ``r`` of the last dim."""
    size = dist.get_world_size(group)
    width = partial.shape[-1]
    if width % size != 0:
        raise ValueError(f"last dim {width} is not divisible by group size {size}")
    chunk = width // size
    # reduce_scatter splits dim 0, so move the rank chunk index to the front.
    chunk_major = (
        partial.to(torch.float32)
        .reshape(-1, size, chunk)
        .transpose(0, 1)
        .contiguous()
    )
    rows = chunk_major.shape[1]
    chunk_major = chunk_major.reshape(size * rows, chunk)
    out = torch.empty((rows, chunk), dtype=torch.float32, device=partial.device)
    dist.reduce_scatter_tensor(out, chunk_major, op=dist.ReduceOp.SUM, group=group)
    return out.reshape(*partial.shape[:-1], chunk)


def all_gather_stacked(local: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Gather equally shaped ``local`` tensors into a new leading rank dimension.

    Flat views are used because backends validate the output against the input shape
    differently (concatenate versus stack), while a 1-D buffer is accepted by all.
    """
    size = dist.get_world_size(group)
    flat_input = local.contiguous().reshape(-1)
    flat_output = torch.empty(size * flat_input.numel(), dtype=local.dtype, device=local.device)
    dist.all_gather_into_tensor(flat_output, flat_input, group=group)
    return flat_output.reshape(size, *local.shape)


def all_gather_last_dim(local: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Concatenate ``local`` from every rank of ``group`` along the last dim, in rank order."""
    size = dist.get_world_size(group)
    if size == 1:
        return local
    chunk = local.shape[-1]
    gathered = all_gather_stacked(local, group)
    return gathered.movedim(0, -2).reshape(*local.shape[:-1], size * chunk)


def all_gather_uneven_last_dim(
    local: torch.Tensor, widths: tuple[int, ...], group: dist.ProcessGroup
) -> torch.Tensor:
    """Concatenate along the last dim when rank ``r`` holds ``widths[r]`` entries.

    Each rank pads to the widest shard so one fixed-size all-gather suffices.
    """
    if len(widths) == 1:
        return local
    widest = max(widths)
    padded = torch.nn.functional.pad(local, (0, widest - local.shape[-1]))
    gathered = all_gather_stacked(padded, group)
    return torch.cat([gathered[rank, ..., :width] for rank, width in enumerate(widths)], dim=-1)
