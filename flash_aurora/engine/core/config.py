from __future__ import annotations

from dataclasses import dataclass
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

V1P5_SURF_INPUT: tuple[str, ...] = (
    "2t",
    "10u",
    "10v",
    "msl",
    "2d",
    "tcwv",
    "tcc",
    "100u",
    "100v",
    "sp",
    "lcc",
    "mcc",
    "hcc",
    "skt",
    "stl1",
    "swvl1",
    "ci",
    "scaled_sd",
    "insolation",
)
V1P5_SURF_OUTPUT_ONLY: tuple[str, ...] = (
    "i10fg",
    "blh",
    "uvb_1h",
    "ssrd_1h",
    "ttr_1h",
    "scaled_tp_1h",
    "scaled_sf_1h",
)
V1P5_SURF: tuple[str, ...] = V1P5_SURF_INPUT + V1P5_SURF_OUTPUT_ONLY
V1P5_STATIC: tuple[str, ...] = (
    "lsm",
    "z",
    "anor",
    "isor",
    "cvh",
    "cl",
    "dl",
    "cvl",
    "slor",
    "slt_0",
    "slt_1",
    "slt_2",
    "slt_3",
    "slt_4",
    "slt_5",
    "slt_6",
    "slt_7",
    "sdfor",
    "sdor",
    "tvh_0",
    "tvh_18",
    "tvh_19",
    "tvh_3",
    "tvh_4",
    "tvh_5",
    "tvh_6",
    "tvl_0",
    "tvl_1",
    "tvl_10",
    "tvl_11",
    "tvl_13",
    "tvl_16",
    "tvl_17",
    "tvl_2",
    "tvl_7",
    "tvl_9",
)
V1P5_ATMOS: tuple[str, ...] = ("z", "u", "v", "t", "q")
V1P5_CHECKPOINT_REVISION: str = "9751bb56e8e4a0f0a780e3cbe978f4c721e12bc7"


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
    output_only_surf_vars: tuple[str, ...] = ()
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
    export_dir: Path | None = None
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
    distributed: "DistributedConfig | None" = None

    def hub_download_options(self) -> "HubDownloadOptions":
        from flash_aurora.engine.core.hub import HubDownloadOptions

        return HubDownloadOptions(
            endpoint=self.hf_endpoint,
            revision=self.hf_revision,
            token=self.hf_token,
        )
