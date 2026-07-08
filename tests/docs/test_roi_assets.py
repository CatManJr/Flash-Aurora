from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ROI_ROOT = REPO_ROOT / "docs" / "roi"
EXPECTED_STATES = ("california", "georgia", "michigan", "texas")


def test_roi_manifest_lists_four_states() -> None:
    manifest = json.loads((ROI_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest) == set(EXPECTED_STATES)


def test_roi_assets_exist_for_each_state() -> None:
    manifest = json.loads((ROI_ROOT / "manifest.json").read_text(encoding="utf-8"))
    for slug in EXPECTED_STATES:
        entry = manifest[slug]
        assert (ROI_ROOT / entry["geojson"]).is_file()
        assert (ROI_ROOT / entry["raster"]).is_file()
        assert (ROI_ROOT / entry["shapefile"]).is_file()
