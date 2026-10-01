"""Roll several ensemble members of one model in lockstep, one RNG stream per member.

Member ``k`` is defined by its seed alone: the generators are seeded with
``base_seed + k`` immediately before its rollout starts, and its RNG state is saved
after every step and restored before the next. Interleaving members on one model
therefore draws exactly the noise that member ``k`` would draw if it ran alone, on
any device and under any parallel scheme whose noise draw matches the single-device
draw (D3). Lockstep stepping puts step ``t`` of every local member in memory at once,
which the per-step ensemble moments (D4) need.

Noise accumulation across sub-steps keeps one cache per backbone, so this driver
runs the main-step rollout only (no ``fine_lead_times``), where the cache is off.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch


def member_seed(base_seed: int, member: int) -> int:
    return base_seed + member


def _capture_rng() -> tuple[torch.Tensor, list[torch.Tensor]]:
    return torch.get_rng_state(), torch.cuda.get_rng_state_all()


def _restore_rng(state: tuple[torch.Tensor, list[torch.Tensor]]) -> None:
    cpu_state, cuda_states = state
    torch.set_rng_state(cpu_state)
    torch.cuda.set_rng_state_all(cuda_states)


@dataclass
class _MemberStream:
    member: int
    seed: int
    rng_state: tuple[torch.Tensor, list[torch.Tensor]]
    predictions: Iterator[Any]


class MemberRollouts:
    """Steps the given members of ``model`` from the same initial condition."""

    def __init__(self, model: Any, batch: Any, steps: int, members: list[int], base_seed: int) -> None:
        from flash_aurora.engine.core.rollout_session import RolloutSession

        self.members = members
        self._streams = []
        for member in members:
            seed = member_seed(base_seed, member)
            torch.manual_seed(seed)
            self._streams.append(
                _MemberStream(
                    member=member,
                    seed=seed,
                    rng_state=_capture_rng(),
                    predictions=iter(RolloutSession(model, cache_in_dram=False).run(batch, steps)),
                )
            )

    def seeds(self) -> dict[int, int]:
        return {stream.member: stream.seed for stream in self._streams}

    def step(self) -> dict[int, Any]:
        """Advance every member by one step; return ``{member: prediction}``."""
        predictions = {}
        for stream in self._streams:
            _restore_rng(stream.rng_state)
            predictions[stream.member] = next(stream.predictions)
            stream.rng_state = _capture_rng()
        return predictions

    def close(self) -> None:
        for stream in self._streams:
            stream.predictions.close()
