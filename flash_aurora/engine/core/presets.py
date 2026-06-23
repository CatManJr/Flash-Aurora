from __future__ import annotations

from dataclasses import replace

from flash_aurora.engine.core.config import (
    EngineConfig,
    ModelVariantSpec,
    SourceProfile,
    STANDARD_ATMOS,
    STANDARD_SURF,
    CAMS_ATMOS_POLLUTION,
    CAMS_STATIC,
    CAMS_SURF_POLLUTION,
    WAVE_STATIC,
    WAVE_SURF_WAM,
)

VARIANTS: dict[str, ModelVariantSpec] = {
    "aurora-0.25-pretrained": ModelVariantSpec(
        name="aurora-0.25-pretrained",
        model_class="AuroraPretrained",
        checkpoint_filename="aurora-0.25-pretrained.ckpt",
        use_lora=False,
        strict_checkpoint=True,
    ),
    "aurora-0.25-small-pretrained": ModelVariantSpec(
        name="aurora-0.25-small-pretrained",
        model_class="AuroraSmallPretrained",
        checkpoint_filename="aurora-0.25-small-pretrained.ckpt",
        use_lora=False,
        strict_checkpoint=False,
        surf_vars=STANDARD_SURF,
        atmos_vars=("u", "v", "t", "q"),
        levels=(50, 250, 500, 600, 700, 850, 925),
        resolution=(400, 800),
    ),
    "aurora-0.25-finetuned": ModelVariantSpec(
        name="aurora-0.25-finetuned",
        model_class="Aurora",
        checkpoint_filename="aurora-0.25-finetuned.ckpt",
        use_lora=True,
        strict_checkpoint=False,
    ),
    "aurora-0.25-12h-pretrained": ModelVariantSpec(
        name="aurora-0.25-12h-pretrained",
        model_class="Aurora12hPretrained",
        checkpoint_filename="aurora-0.25-12h-pretrained.ckpt",
        use_lora=False,
        timestep_hours=12,
        strict_checkpoint=True,
    ),
    "aurora-0.1-finetuned": ModelVariantSpec(
        name="aurora-0.1-finetuned",
        model_class="AuroraHighRes",
        checkpoint_filename="aurora-0.1-finetuned.ckpt",
        resolution=(1801, 3600),
        static_pickle="aurora-0.1-static.pickle",
        strict_checkpoint=False,
    ),
    "aurora-0.4-air-pollution": ModelVariantSpec(
        name="aurora-0.4-air-pollution",
        model_class="AuroraAirPollution",
        checkpoint_filename="aurora-0.4-air-pollution.ckpt",
        resolution=(451, 900),
        static_pickle="aurora-0.4-air-pollution-static.pickle",
        surf_vars=STANDARD_SURF + CAMS_SURF_POLLUTION,
        static_vars=CAMS_STATIC,
        atmos_vars=STANDARD_ATMOS + CAMS_ATMOS_POLLUTION,
        strict_checkpoint=False,
    ),
    "aurora-0.25-wave": ModelVariantSpec(
        name="aurora-0.25-wave",
        model_class="AuroraWave",
        checkpoint_filename="aurora-0.25-wave.ckpt",
        surf_vars=STANDARD_SURF + WAVE_SURF_WAM,
        static_vars=WAVE_STATIC,
        static_pickle="aurora-0.25-wave-static.pickle",
        strict_checkpoint=False,
    ),
}

SOURCES: dict[str, SourceProfile] = {
    "cds_era5": SourceProfile(
        name="cds_era5",
        schema="cds_era5",
        time_policy="first_two",
        flip_lat=False,
    ),
    "wb2_hres": SourceProfile(
        name="wb2_hres",
        schema="wb2_hres",
        time_policy="pair",
        flip_lat=True,
    ),
    "grib_ifs_0.1": SourceProfile(
        name="grib_ifs_0.1",
        schema="grib_ifs_analysis",
        time_policy="pair",
        regrid_res=0.1,
        raw_layout="grib",
    ),
    "cams": SourceProfile(
        name="cams",
        schema="cams",
        time_policy="pair",
        raw_layout="netcdf",
    ),
    "wb2_wam": SourceProfile(
        name="wb2_wam",
        schema="wb2_wam",
        time_policy="first_two",
        flip_lat=True,
        flip_lat_wave=False,
        raw_layout="mixed",
    ),
}


class PresetRegistry:
    def __init__(self) -> None:
        self._configs: dict[str, EngineConfig] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register(
            "era5_pretrained",
            EngineConfig(variant=VARIANTS["aurora-0.25-pretrained"], source=SOURCES["cds_era5"]),
        )
        self.register(
            "hres_t0_finetuned",
            EngineConfig(variant=VARIANTS["aurora-0.25-finetuned"], source=SOURCES["wb2_hres"]),
        )
        self.register(
            "small_pretrained",
            EngineConfig(
                variant=VARIANTS["aurora-0.25-small-pretrained"],
                source=SOURCES["cds_era5"],
            ),
        )
        self.register(
            "hres_0.1",
            EngineConfig(variant=VARIANTS["aurora-0.1-finetuned"], source=SOURCES["grib_ifs_0.1"]),
        )
        self.register(
            "cams",
            EngineConfig(variant=VARIANTS["aurora-0.4-air-pollution"], source=SOURCES["cams"]),
        )
        self.register(
            "wave",
            EngineConfig(variant=VARIANTS["aurora-0.25-wave"], source=SOURCES["wb2_wam"]),
        )
        self.register(
            "tc_tracking",
            EngineConfig(variant=VARIANTS["aurora-0.25-finetuned"], source=SOURCES["wb2_hres"]),
        )

    def register(self, name: str, config: EngineConfig) -> None:
        self._configs[name] = config

    def get(self, name: str) -> EngineConfig:
        if name not in self._configs:
            raise KeyError(f"Unknown preset: {name}")
        return replace(self._configs[name])

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._configs))

    def items(self) -> tuple[tuple[str, EngineConfig], ...]:
        return tuple(sorted(self._configs.items()))


DEFAULT_PRESETS = PresetRegistry()
