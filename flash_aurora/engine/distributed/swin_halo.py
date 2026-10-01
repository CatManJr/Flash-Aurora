"""Shifted-window exchange for a longitude-sharded 3D Swin backbone (deviation D2).

Spatial ranks store contiguous longitude ranges at every backbone level (see
:mod:`longitude_partition`). A halo after each shift is not enough for Aurora: with
window width 12 and two spatial ranks, the 0.25 and 0.1 degree grids give 30
windows at level 1, 15 at level 2 (7.5 per shard), and 8 at level 3 after padding
90 columns to 96. No single longitude split aligns all three levels, so windows
straddle shards even without a shift.

The exchange therefore re-shards around every attention call. Before attention,
each rank gathers the columns of a contiguous run of complete windows, in the
rolled and padded coordinates of the single-device block. After attention it sends
every column back to its owner. Windows attend over exactly the tokens they hold on
one device, so the exchange is a copy and preserves the map. Latitude and levels
stay local; their roll and padding are applied unchanged.

Column bookkeeping: the single-device block rolls by ``-shift`` and then pads
``pad_left`` columns on the west. Padded column ``p`` therefore holds global column
``(p - pad_left + shift) mod W``, or zero padding outside ``[pad_left, pad_left + W)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.distributed as dist

from flash_aurora.engine.distributed.longitude_partition import LongitudeShards

_LONGITUDE_DIM = 3  # tokens are laid out as (B, C, H, W, D)


@dataclass(frozen=True)
class WindowColumnPlan:
    """Global longitude column, or padding, at each attention position of each rank."""

    shards: LongitudeShards
    window_ranges: tuple[tuple[int, int], ...]
    attention_columns: tuple[tuple[int | None, ...], ...]

    def window_range(self, rank: int) -> tuple[int, int]:
        return self.window_ranges[rank]

    def positions_from(self, receiver: int, sender: int) -> list[int]:
        """Attention positions of ``receiver`` whose column ``sender`` stores, in order."""
        return [
            position
            for position, column in enumerate(self.attention_columns[receiver])
            if column is not None and self.shards.owner(column) == sender
        ]

    def local_columns_for(self, receiver: int, sender: int) -> list[int]:
        """``sender``-local storage indices of those columns, in the same order."""
        first = self.shards.first_column(sender)
        return [
            self.attention_columns[receiver][position] - first
            for position in self.positions_from(receiver, sender)
        ]


@lru_cache(maxsize=128)
def plan_window_columns(
    shards: LongitudeShards,
    window_width: int,
    shift: int,
    pad_left: int,
    pad_right: int,
) -> WindowColumnPlan:
    """Assign contiguous runs of complete windows to spatial ranks, as evenly as possible."""
    width = shards.width
    num_shards = shards.num_shards
    padded_width = pad_left + width + pad_right
    if padded_width % window_width != 0:
        raise ValueError(f"padded width {padded_width} is not a multiple of {window_width}")
    num_windows = padded_width // window_width
    if num_windows < num_shards:
        raise ValueError(f"{num_windows} windows cannot be spread over {num_shards} spatial ranks")

    window_ranges = tuple(
        (rank * num_windows // num_shards, (rank + 1) * num_windows // num_shards)
        for rank in range(num_shards)
    )

    def column_at(padded: int) -> int | None:
        rolled = padded - pad_left
        if not 0 <= rolled < width:
            return None
        return (rolled + shift) % width

    attention_columns = tuple(
        tuple(column_at(p) for p in range(first * window_width, end * window_width))
        for first, end in window_ranges
    )
    return WindowColumnPlan(
        shards=shards,
        window_ranges=window_ranges,
        attention_columns=attention_columns,
    )


def _index(columns: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(columns, dtype=torch.long, device=device)


def _exchange(
    sends: dict[int, torch.Tensor],
    recv_shapes: dict[int, tuple[int, ...]],
    *,
    like: torch.Tensor,
    group: dist.ProcessGroup,
) -> dict[int, torch.Tensor]:
    """Point-to-point exchange with peers (group ranks); returns received tensors."""
    received = {
        peer: torch.empty(shape, dtype=like.dtype, device=like.device)
        for peer, shape in recv_shapes.items()
    }
    ops = [
        dist.P2POp(dist.isend, tensor, peer=dist.get_global_rank(group, peer), group=group)
        for peer, tensor in sends.items()
    ] + [
        dist.P2POp(dist.irecv, tensor, peer=dist.get_global_rank(group, peer), group=group)
        for peer, tensor in received.items()
    ]
    if ops:
        for request in dist.batch_isend_irecv(ops):
            request.wait()
    return received


def _with_width(x: torch.Tensor, width: int) -> tuple[int, ...]:
    shape = list(x.shape)
    shape[_LONGITUDE_DIM] = width
    return tuple(shape)


def gather_window_columns(
    stored: torch.Tensor,
    plan: WindowColumnPlan,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Storage layout ``(B, C, H, W_s, D)`` to this rank's window layout ``(B, C, H, W_a, D)``.

    Padding positions are zero, as in the single-device ``pad_3d``.
    """
    rank = dist.get_rank(group)
    size = dist.get_world_size(group)
    attention_width = len(plan.attention_columns[rank])
    gathered = stored.new_zeros(_with_width(stored, attention_width))

    sends = {
        peer: stored.index_select(
            _LONGITUDE_DIM, _index(plan.local_columns_for(peer, rank), stored.device)
        )
        for peer in range(size)
        if peer != rank and plan.positions_from(peer, rank)
    }
    recv_shapes = {
        peer: _with_width(stored, len(plan.positions_from(rank, peer)))
        for peer in range(size)
        if peer != rank and plan.positions_from(rank, peer)
    }
    received = _exchange(sends, recv_shapes, like=stored, group=group)
    received[rank] = stored.index_select(
        _LONGITUDE_DIM, _index(plan.local_columns_for(rank, rank), stored.device)
    )
    for sender, columns in received.items():
        positions = plan.positions_from(rank, sender)
        if positions:
            gathered.index_copy_(_LONGITUDE_DIM, _index(positions, stored.device), columns)
    return gathered


def scatter_window_columns(
    attended: torch.Tensor,
    plan: WindowColumnPlan,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Inverse of :func:`gather_window_columns`: window layout back to storage layout."""
    rank = dist.get_rank(group)
    size = dist.get_world_size(group)
    stored = attended.new_empty(_with_width(attended, plan.shards.shard_width(rank)))

    sends = {
        owner: attended.index_select(
            _LONGITUDE_DIM, _index(plan.positions_from(rank, owner), attended.device)
        )
        for owner in range(size)
        if owner != rank and plan.positions_from(rank, owner)
    }
    recv_shapes = {
        peer: _with_width(attended, len(plan.positions_from(peer, rank)))
        for peer in range(size)
        if peer != rank and plan.positions_from(peer, rank)
    }
    received = _exchange(sends, recv_shapes, like=attended, group=group)
    received[rank] = attended.index_select(
        _LONGITUDE_DIM, _index(plan.positions_from(rank, rank), attended.device)
    )
    for sender, columns in received.items():
        local_columns = plan.local_columns_for(sender, rank)
        if local_columns:
            stored.index_copy_(_LONGITUDE_DIM, _index(local_columns, attended.device), columns)
    return stored
