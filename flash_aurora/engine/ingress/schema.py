from __future__ import annotations


class SchemaRegistry:
    """Maps external field names to Aurora variable names."""

    def __init__(self) -> None:
        self._tables: dict[str, dict[str, str]] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register(
            "cds_era5",
            {
                "t2m": "2t",
                "u10": "10u",
                "v10": "10v",
                "msl": "msl",
            },
        )
        self.register(
            "wb2_hres",
            {
                "2m_temperature": "2t",
                "10m_u_component_of_wind": "10u",
                "10m_v_component_of_wind": "10v",
                "mean_sea_level_pressure": "msl",
            },
        )
        self.register(
            "grib_ifs_analysis",
            {
                "2t": "2t",
                "10u": "10u",
                "10v": "10v",
                "msl": "msl",
            },
        )
        self.register(
            "cams",
            {
                "t2m": "2t",
                "u10": "10u",
                "v10": "10v",
                "msl": "msl",
            },
        )
        self.register(
            "wb2_wam",
            {
                "significant_height_of_combined_wind_waves_and_swell": "swh",
                "mean_wave_direction": "mwd",
            },
        )

    def register(self, preset: str, mapping: dict[str, str]) -> None:
        self._tables[preset] = dict(mapping)

    def map_name(self, preset: str, external: str) -> str:
        table = self._tables.get(preset)
        if table is None:
            raise KeyError(f"Unknown schema preset: {preset}")
        if external not in table:
            raise KeyError(f"Unknown field {external!r} for preset {preset!r}")
        return table[external]

    def presets(self) -> tuple[str, ...]:
        return tuple(sorted(self._tables))


DEFAULT_SCHEMAS = SchemaRegistry()
