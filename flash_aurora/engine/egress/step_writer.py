from __future__ import annotations

from pathlib import Path

from flash_aurora.aurora import Batch

from flash_aurora.engine.egress.export_options import ExportOptions
from flash_aurora.engine.egress.geotiff_codec import geotiff_variables, write_surface_geotiff
from flash_aurora.engine.egress.naming import PredictionNaming
from flash_aurora.engine.egress.offload import owned_cpu_copy
from flash_aurora.engine.egress.roi import apply_mask
from flash_aurora.engine.egress.serialize import BatchExporter


class RolloutStepWriter:
    """Write one rollout step to NetCDF and/or GeoTIFF with optional ROI clipping."""

    def __init__(
        self,
        export_dir: Path,
        options: ExportOptions | None = None,
        naming: PredictionNaming | None = None,
    ) -> None:
        self._export_dir = export_dir
        self._options = options or ExportOptions()
        self._naming = naming or PredictionNaming()
        self._netcdf = BatchExporter(export_dir)

    @property
    def options(self) -> ExportOptions:
        return self._options

    def prepare_batch(self, batch: Batch) -> Batch:
        cpu_batch = owned_cpu_copy(batch)
        regions = self._options.resolved_regions()
        if regions is None:
            return cpu_batch
        if len(regions) == 1 and not regions[0][0]:
            return apply_mask(cpu_batch, regions[0][1])
        return cpu_batch

    def write_step(self, step_index: int, batch: Batch) -> list[Path]:
        cpu_batch = owned_cpu_copy(batch)
        return self._write_regions(step_index, cpu_batch)

    def write_owned_step(self, step_index: int, batch: Batch) -> list[Path]:
        return self._write_regions(step_index, batch)

    def _write_regions(self, step_index: int, batch: Batch) -> list[Path]:
        regions = self._options.resolved_regions()
        if regions is None:
            return self._write_clipped(step_index, batch, region="", export_crs=self._options.export_crs)

        paths: list[Path] = []
        for region_name, mask in regions:
            clipped = apply_mask(batch, mask)
            paths.extend(
                self._write_clipped(
                    step_index,
                    clipped,
                    region=region_name,
                    export_crs=mask.export_crs,
                )
            )
        return paths

    def _write_clipped(
        self,
        step_index: int,
        batch: Batch,
        *,
        region: str,
        export_crs: str,
    ) -> list[Path]:
        region_dir = self._naming.regional_dir(self._export_dir, region)
        region_dir.mkdir(parents=True, exist_ok=True)

        if self._options.format == "netcdf":
            path = self._naming.regional_path(self._export_dir, region, step_index)
            self._netcdf.write_netcdf(batch, path)
            return [path]

        variables = geotiff_variables(batch, self._options.variables)
        paths: list[Path] = []
        for variable in variables:
            path = self._naming.regional_geotiff_path(self._export_dir, region, step_index, variable)
            write_surface_geotiff(
                batch,
                path,
                variable,
                nodata=self._options.nodata,
                crs=export_crs,
            )
            paths.append(path)
        return paths
