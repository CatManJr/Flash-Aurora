from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from flash_aurora.engine.egress.crs import BATCH_CRS, DEFAULT_EXPORT_CRS
from flash_aurora.engine.egress.mask import Mask
from flash_aurora.engine.egress.roi_batch import RoiBatch

ExportFormat = Literal["netcdf", "geotiff"]


@dataclass(frozen=True)
class RoiBounds:
    """Deprecated: use :class:`Mask` via ``Mask.from_bounds``."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    @classmethod
    def pacific_tc_panel(cls) -> Mask:
        return Mask.pacific_tc_panel()

    def to_mask(self, *, export_crs: str | int = DEFAULT_EXPORT_CRS) -> Mask:
        return Mask.from_bounds(
            lat_min=self.lat_min,
            lat_max=self.lat_max,
            lon_min=self.lon_min,
            lon_max=self.lon_max,
            export_crs=export_crs,
        )


@dataclass(frozen=True)
class RoiGeoJson:
    """Deprecated: use :class:`Mask` via ``Mask.from_geojson``."""

    path: Path

    def to_mask(self, *, export_crs: str | int = DEFAULT_EXPORT_CRS) -> Mask:
        return Mask.from_geojson(self.path, export_crs=export_crs)


RoiSpec = Mask | RoiBounds | RoiGeoJson


def coerce_mask(spec: RoiSpec | None, *, export_crs: str | int = DEFAULT_EXPORT_CRS) -> Mask | None:
    if spec is None:
        return None
    if isinstance(spec, Mask):
        return spec
    if isinstance(spec, RoiBounds):
        return spec.to_mask(export_crs=export_crs)
    if isinstance(spec, RoiGeoJson):
        return spec.to_mask(export_crs=export_crs)
    raise TypeError(f"unsupported mask spec: {type(spec)!r}")


@dataclass(frozen=True)
class ExportOptions:
    """Egress settings for :meth:`AuroraEngine.rollout_and_export`."""

    format: ExportFormat = "netcdf"
    mask: Mask | None = None
    roi: RoiSpec | None = None
    roi_batch: RoiBatch | None = None
    variables: tuple[str, ...] | None = None
    nodata: float = -9999.0
    export_crs: str = DEFAULT_EXPORT_CRS

    def _reject_mixed_roi_args(self) -> None:
        provided = sum(
            1
            for value in (self.mask, self.roi, self.roi_batch)
            if value is not None
        )
        if provided > 1:
            raise ValueError("pass only one of mask=, roi=, or roi_batch= in ExportOptions")

    def resolved_mask(self) -> Mask | None:
        self._reject_mixed_roi_args()
        if self.roi_batch is not None:
            return None
        if self.mask is not None:
            return self.mask
        return coerce_mask(self.roi, export_crs=self.export_crs)

    def resolved_regions(self) -> tuple[tuple[str, Mask], ...] | None:
        """Named regions to export per step. ``None`` means global (no clip)."""
        self._reject_mixed_roi_args()
        if self.roi_batch is not None:
            return self.roi_batch.regions
        single = self.resolved_mask()
        if single is None:
            return None
        return (("", single),)

    def resolved_export_crs(self) -> str:
        resolved = self.resolved_mask()
        if resolved is not None:
            return resolved.export_crs
        return self.export_crs
