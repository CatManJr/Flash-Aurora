"""Structural surface shared by optimized Aurora and Aurora 1.5 backends."""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Protocol, runtime_checkable

from flash_aurora.models.aurora.batch import Batch


@runtime_checkable
class AuroraModel(Protocol):
    """Minimal contract used by Engine / CheckpointLoader / RolloutSession."""

    patch_size: int
    timestep: timedelta
    batch_transform_hook: Any

    def eval(self) -> Any: ...

    def forward(self, batch: Batch, *args: Any, **kwargs: Any) -> Batch: ...

    def load_checkpoint_local(self, path: str, strict: bool = True) -> None: ...


OPTIMIZED_MODEL_CLASSES: frozenset[str] = frozenset(
    {
        "Aurora",
        "AuroraPretrained",
        "AuroraSmallPretrained",
        "Aurora12hPretrained",
        "AuroraHighRes",
        "AuroraAirPollution",
        "AuroraWave",
    }
)

V1P5_MODEL_CLASSES: frozenset[str] = frozenset(
    {
        "AuroraV1p5",
        "AuroraV1p5Ensemble",
    }
)


def is_v1p5_model_class(class_name: str) -> bool:
    return class_name in V1P5_MODEL_CLASSES


def is_optimized_model_class(class_name: str) -> bool:
    return class_name in OPTIMIZED_MODEL_CLASSES


def model_uses_v1p5_rollout(model: object) -> bool:
    """Dispatch rollouts by capability rather than import-path strings."""
    if type(model).__name__ in V1P5_MODEL_CLASSES:
        return True
    return bool(getattr(model, "variable_lead_time", False))
