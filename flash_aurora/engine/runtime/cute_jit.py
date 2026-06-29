"""CuTe DSL JIT helpers for engine forward warmup."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flash_aurora.aurora.model.aurora import Aurora


def model_uses_cute_window_attn(model: Aurora) -> bool:
    """Return True when the model runs custom CuTe window attention."""
    cfg = getattr(model, "inference_config", None)
    if cfg is not None:
        return cfg.use_cute_window_attn
    backbone = getattr(model, "backbone", None)
    return bool(getattr(backbone, "use_cute_window_attn", False))


def prepare_cute_dsl_runtime() -> str | None:
    """Set ``CUTE_DSL_ARCH`` from the local GPU before the first CuTe JIT compile."""
    from flash_aurora.aurora.ops.cute._arch_env import ensure_cute_dsl_arch

    return ensure_cute_dsl_arch()
