from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

STANDARD_LEVELS: tuple[int, ...] = (
    50,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
    700,
    850,
    925,
    1000,
)

STANDARD_SURF: tuple[str, ...] = ("2t", "10u", "10v", "msl")
STANDARD_STATIC: tuple[str, ...] = ("lsm", "slt", "z")
STANDARD_ATMOS: tuple[str, ...] = ("t", "u", "v", "q", "z")


@dataclass(frozen=True)
class ModelVariantSpec:
    name: str
    model_class: str
    checkpoint_filename: str
    hf_repo: str = "microsoft/aurora"
    use_lora: bool = True
    lora_mode: str = "single"
    timestep_hours: int = 6
    surf_vars: tuple[str, ...] = STANDARD_SURF
    static_vars: tuple[str, ...] = STANDARD_STATIC
    atmos_vars: tuple[str, ...] = STANDARD_ATMOS
    levels: tuple[int | float, ...] = STANDARD_LEVELS
    resolution: tuple[int, int] = (721, 1440)
    static_pickle: str = "aurora-0.25-static.pickle"
    strict_checkpoint: bool = True


@dataclass(frozen=True)
class SourceProfile:
    name: str
    schema: str
    time_policy: str = "pair"
    flip_lat: bool = False
    flip_lat_wave: bool = False
    static_source: str = "hf_pickle"
    regrid_res: float | None = None
    raw_layout: str = "netcdf"


@dataclass
class EngineConfig:
    variant: ModelVariantSpec
    source: SourceProfile
    asset_root: Path | None = None
    allow_hub_download: bool = True
    export_dir: Path = field(default_factory=lambda: Path("output"))
    inference_precision: str | None = None
    cuda_graph: bool = False
    device: str = "cuda:0"
