from __future__ import annotations

from flash_aurora.models.aurora import (
    Aurora,
    Aurora12hPretrained,
    AuroraAirPollution,
    AuroraHighRes,
    AuroraPretrained,
    AuroraSmallPretrained,
    AuroraWave,
)
from flash_aurora.models.aurora_v1p5 import AuroraV1p5, AuroraV1p5Ensemble

from flash_aurora.engine.core.model_protocol import (
    AuroraModel,
    V1P5_MODEL_CLASSES,
    is_v1p5_model_class,
)

MODEL_REGISTRY: dict[str, type] = {
    "Aurora": Aurora,
    "AuroraPretrained": AuroraPretrained,
    "AuroraSmallPretrained": AuroraSmallPretrained,
    "Aurora12hPretrained": Aurora12hPretrained,
    "AuroraHighRes": AuroraHighRes,
    "AuroraAirPollution": AuroraAirPollution,
    "AuroraWave": AuroraWave,
    "AuroraV1p5": AuroraV1p5,
    "AuroraV1p5Ensemble": AuroraV1p5Ensemble,
}

# Kwargs that belong to the optimized flash family only (not yet on Aurora 1.5).
_OPTIMIZED_ONLY_KWARGS: frozenset[str] = frozenset(
    {
        "use_lora_merged_inference",
    }
)



class ModelFactory:
    @staticmethod
    def create(
        class_name: str,
        *,
        use_lora: bool,
        lora_mode: str,
        **kwargs: object,
    ) -> AuroraModel:
        model_cls = MODEL_REGISTRY.get(class_name)
        if model_cls is None:
            raise KeyError(f"Unknown model class: {class_name}")

        if is_v1p5_model_class(class_name):
            filtered = {key: value for key, value in kwargs.items() if key not in _OPTIMIZED_ONLY_KWARGS}
            # AuroraV1p5 defaults use_lora=False; do not pass flash-only lora_mode.
            return model_cls(use_lora=use_lora, **filtered)

        if class_name in {"AuroraPretrained", "AuroraSmallPretrained", "Aurora12hPretrained"}:
            return model_cls(use_lora=use_lora, **kwargs)
        return model_cls(use_lora=use_lora, lora_mode=lora_mode, **kwargs)


__all__ = [
    "MODEL_REGISTRY",
    "ModelFactory",
    "V1P5_MODEL_CLASSES",
]
