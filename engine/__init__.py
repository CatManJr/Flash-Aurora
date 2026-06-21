from engine.bootstrap import ensure_repo_paths

ensure_repo_paths()

from engine.core.config import EngineConfig
from engine.core.engine import AuroraEngine
from engine.core.presets import DEFAULT_PRESETS, PresetRegistry
from engine.ingress.adapters import IngestRequest
from engine.ingress.build_ic import InitialConditionBuilder
from engine.ingress.download import DataDownloader, DownloadBackendError, DownloadRequest, DownloadResult
from engine.ingress.download.credentials import DownloadCredentials

__all__ = [
    "AuroraEngine",
    "DataDownloader",
    "DEFAULT_PRESETS",
    "DownloadBackendError",
    "DownloadCredentials",
    "DownloadRequest",
    "DownloadResult",
    "EngineConfig",
    "InitialConditionBuilder",
    "IngestRequest",
    "PresetRegistry",
]
