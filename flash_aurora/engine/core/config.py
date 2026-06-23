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

CAMS_SURF_POLLUTION: tuple[str, ...] = (
    "pm1",
    "pm2p5",
    "pm10",
    "tcco",
    "tc_no",
    "tcno2",
    "gtco3",
    "tcso2",
)
CAMS_ATMOS_POLLUTION: tuple[str, ...] = ("co", "no", "no2", "go3", "so2")
CAMS_STATIC: tuple[str, ...] = STANDARD_STATIC + (
    "static_ammonia",
    "static_ammonia_log",
    "static_co",
    "static_co_log",
    "static_nox",
    "static_nox_log",
    "static_so2",
    "static_so2_log",
)

WAVE_SURF_WAM: tuple[str, ...] = (
    "swh",
    "mwd",
    "mwp",
    "pp1d",
    "shww",
    "mdww",
    "mpww",
    "shts",
    "mdts",
    "mpts",
    "swh1",
    "mwd1",
    "mwp1",
    "swh2",
    "mwd2",
    "mwp2",
    "wind",
    "dwi",
)
WAVE_STATIC: tuple[str, ...] = STANDARD_STATIC + ("wmb", "lat_mask")


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
    checkpoint_path: Path | None = None
    user_cwd: Path | None = None
    allow_hub_download: bool = True
    hf_endpoint: str | None = None
    hf_revision: str | None = None
    hf_token: str | None = None
    export_dir: Path = field(default_factory=lambda: Path("output"))
    inference_precision: str | None = None
    cuda_graph: bool = False
    device: str = "cuda:0"
    preset_name: str | None = None
    gpu_guard: bool = True
    gpu_guard_timeout: float = 3600.0
    gpu_rollout_steps: int = 1
    overlap_ic_load: bool = True
    async_export: bool = False
    export_pool_size: int = 2
    export_max_inflight: int | None = None
    export_use_egress_stream: bool = True
    ic_cache: bool = False
    forward_warmup_iters: int = 2

    def hub_download_options(self) -> "HubDownloadOptions":
        from flash_aurora.engine.core.hub import HubDownloadOptions

        return HubDownloadOptions(
            endpoint=self.hf_endpoint,
            revision=self.hf_revision,
            token=self.hf_token,
        )
