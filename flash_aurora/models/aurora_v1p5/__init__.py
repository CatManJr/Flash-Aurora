"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Aurora 1.5 side-path package derived from https://github.com/microsoft/aurora
(tag v2.0.1). Shares ``Batch`` / ``Metadata`` with ``flash_aurora.models.aurora``.
See ``LICENSE.txt`` and ``NOTICE.md`` in this package for redistribution terms.
"""

from __future__ import annotations

from flash_aurora.models.aurora.batch import Batch, Metadata
from flash_aurora.models.aurora_v1p5.insolation import insolation
from flash_aurora.models.aurora_v1p5.model.aurora import AuroraV1p5, AuroraV1p5Ensemble
from flash_aurora.models.aurora_v1p5.rollout import rollout

__all__ = [
    "AuroraV1p5",
    "AuroraV1p5Ensemble",
    "Batch",
    "Metadata",
    "insolation",
    "rollout",
]
