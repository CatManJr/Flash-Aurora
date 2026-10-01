"""Ensemble mean and spread across uncertainty-parallel ranks (deviation D4).

BEAST (arXiv:2609.12815, III-B) reduces two tensors per rank in training: the local
sum ``S_u`` and squared sum ``Q_u`` of its ``m_u`` members. The paper does not display
the reconstruction; the unique one is ``mu = S / m`` and ``sigma^2 = Q / m - mu^2``. At
inference the uncertainty groups are independent (V-B), so a reduce at inference is
our addition.

``Q / m - mu^2`` cancels catastrophically on large-magnitude, low-spread fields such
as geopotential, where ``|mu|`` is about ``5 x 10^4`` and the spread a few tens. Each
rank therefore keeps ``(m_u, mu_u, M2_u)`` in FP64, with ``M2`` the sum of squared
deviations from the local mean, and the ranks merge them with the parallel update of
Chan, Golub, and LeVeque:

    delta = mu_b - mu_a,  m = m_a + m_b,
    mu = mu_a + delta * m_b / m,  M2 = M2_a + M2_b + delta^2 * m_a * m_b / m.

The communication stays at two fields per variable, as with ``(S, Q)``. Every rank
merges the gathered moments in rank order, so all ranks hold bitwise-equal results.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from flash_aurora.engine.distributed.collectives import all_gather_stacked


@dataclass(frozen=True)
class EnsembleMoments:
    """Member count, mean, and sum of squared deviations of one field, in FP64."""

    count: int
    mean: torch.Tensor
    squared_deviations: torch.Tensor

    def spread(self, ddof: int) -> torch.Tensor:
        """Standard deviation with ``count - ddof`` in the denominator.

        ``ddof = 1`` is the unbiased spread that the fair CRPS uses.
        """
        if self.count <= ddof:
            raise ValueError(f"{self.count} members cannot give a spread with ddof={ddof}")
        return torch.sqrt(self.squared_deviations / (self.count - ddof))


def local_moments(members: list[torch.Tensor]) -> EnsembleMoments:
    """Welford accumulation over this rank's members of one field."""
    if not members:
        raise ValueError("at least one member is required")
    mean = torch.zeros_like(members[0], dtype=torch.float64)
    squared_deviations = torch.zeros_like(mean)
    for count, member in enumerate(members, start=1):
        value = member.to(torch.float64)
        delta = value - mean
        mean = mean + delta / count
        squared_deviations = squared_deviations + delta * (value - mean)
    return EnsembleMoments(len(members), mean, squared_deviations)


def merge_moments(a: EnsembleMoments, b: EnsembleMoments) -> EnsembleMoments:
    """Chan's parallel update of two disjoint member sets."""
    count = a.count + b.count
    delta = b.mean - a.mean
    mean = a.mean + delta * (b.count / count)
    squared_deviations = (
        a.squared_deviations + b.squared_deviations + delta * delta * (a.count * b.count / count)
    )
    return EnsembleMoments(count, mean, squared_deviations)


def all_reduce_moments(local: EnsembleMoments, group: dist.ProcessGroup) -> EnsembleMoments:
    """Merge the moments of every rank of ``group``; every rank gets the same result."""
    size = dist.get_world_size(group)
    if size == 1:
        return local
    counts: list[int] = [0] * size
    dist.all_gather_object(counts, local.count, group=group)
    stacked = torch.stack([local.mean, local.squared_deviations]).contiguous()
    gathered = all_gather_stacked(stacked, group)
    merged = EnsembleMoments(counts[0], gathered[0, 0], gathered[0, 1])
    for rank in range(1, size):
        merged = merge_moments(merged, EnsembleMoments(counts[rank], gathered[rank, 0], gathered[rank, 1]))
    return merged
