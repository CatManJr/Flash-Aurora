from __future__ import annotations

from flash_aurora.engine.core.config import SourceProfile
from flash_aurora.engine.ingress.adapters.base import DataSourceAdapter, StubAdapter
from flash_aurora.engine.ingress.adapters.cams import CamsAdapter
from flash_aurora.engine.ingress.adapters.era5 import CdsEra5Adapter
from flash_aurora.engine.ingress.adapters.hres_analysis import GribHresAnalysisAdapter
from flash_aurora.engine.ingress.adapters.hres_t0 import Wb2HresT0Adapter
from flash_aurora.engine.ingress.adapters.wave import Wb2WamWaveAdapter


class AdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, DataSourceAdapter] = {
            "cds_era5": CdsEra5Adapter(),
            "wb2_hres": Wb2HresT0Adapter(),
            "grib_ifs_0.1": GribHresAnalysisAdapter(),
            "cams": CamsAdapter(),
            "wb2_wam": Wb2WamWaveAdapter(),
        }

    def get(self, source: SourceProfile) -> DataSourceAdapter:
        adapter = self._adapters.get(source.name)
        if adapter is None:
            raise KeyError(f"No adapter registered for source {source.name!r}")
        return adapter

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))


DEFAULT_ADAPTERS = AdapterRegistry()


def get_adapter(source: SourceProfile) -> DataSourceAdapter:
    return DEFAULT_ADAPTERS.get(source)
