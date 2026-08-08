"""Model backends and shared acceleration under ``flash_aurora.models``.

Layout:

- ``flash_aurora.models.aurora`` -- legacy optimized Aurora family (upstream
  baseline **v1.8.0**, last release before Aurora 2.0.0 / Aurora 1.5)
- ``flash_aurora.models.aurora_v1p5`` -- Aurora 1.5 side path (upstream v2.0.1)
- ``flash_aurora.models.ops`` -- Triton / CuTe kernels (shared accel)
- ``flash_aurora.models.inference_precision`` -- precision presets

Families do not import each other. Engine composes them via registry/presets.
"""

from __future__ import annotations

__all__: list[str] = []
