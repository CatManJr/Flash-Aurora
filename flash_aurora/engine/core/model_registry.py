from __future__ import annotations

from flash_aurora.aurora import (
    Aurora,
    Aurora12hPretrained,
    AuroraAirPollution,
    AuroraHighRes,
    AuroraPretrained,
    AuroraSmallPretrained,
    AuroraWave,
)
from flash_aurora.aurora.model.aurora import Aurora as AuroraBase

MODEL_REGISTRY: dict[str, type[AuroraBase]] = {
    "Aurora": Aurora,
    "AuroraPretrained": AuroraPretrained,
    "AuroraSmallPretrained": AuroraSmallPretrained,
    "Aurora12hPretrained": Aurora12hPretrained,
    "AuroraHighRes": AuroraHighRes,
    "AuroraAirPollution": AuroraAirPollution,
    "AuroraWave": AuroraWave,
}


class ModelFactory:
    @staticmethod
    def create(class_name: str, *, use_lora: bool, lora_mode: str) -> AuroraBase:
        model_cls = MODEL_REGISTRY.get(class_name)
        if model_cls is None:
            raise KeyError(f"Unknown model class: {class_name}")
        if class_name in {"AuroraPretrained", "AuroraSmallPretrained", "Aurora12hPretrained"}:
            return model_cls(use_lora=use_lora)
        return model_cls(use_lora=use_lora, lora_mode=lora_mode)
