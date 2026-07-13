"""Compatibility shim for ``flash_aurora.aurora``.

Prefer ``flash_aurora.models.aurora``. This module re-exports the optimized
Aurora package surface so existing ``from flash_aurora.aurora import ...``
imports keep working for one release cycle.
"""

from __future__ import annotations

from flash_aurora.models.aurora import (
    Aurora,
    Aurora12hPretrained,
    AuroraAirPollution,
    AuroraHighRes,
    AuroraInferencePrecision,
    AuroraPretrained,
    AuroraSmall,
    AuroraSmallPretrained,
    AuroraWave,
    Batch,
    Metadata,
    Tracker,
    rollout,
)

__all__ = [
    "Aurora",
    "AuroraPretrained",
    "AuroraSmallPretrained",
    "AuroraSmall",
    "Aurora12hPretrained",
    "AuroraHighRes",
    "AuroraAirPollution",
    "AuroraWave",
    "AuroraInferencePrecision",
    "Batch",
    "Metadata",
    "rollout",
    "Tracker",
]
