from __future__ import annotations

from unittest.mock import patch

from flash_aurora.aurora import Batch, Metadata
from flash_aurora.engine.egress.export_options import ExportOptions
from flash_aurora.engine.egress.mask import Mask
from flash_aurora.engine.egress.roi_batch import RoiBatch
from flash_aurora.engine.egress.step_writer import RolloutStepWriter
from tests.engine.test_egress_roi_geotiff import _grid_batch


def test_roi_batch_example_batch_writes_per_region_subdirs(tmp_path: Path) -> None:
    batch = _grid_batch()
    writer = RolloutStepWriter(
        tmp_path,
        ExportOptions(format="netcdf", roi_batch=RoiBatch.example_batch()),
    )
    paths = writer.write_step(0, batch)
    assert len(paths) == 4
    assert paths[0].parent.name == "china"
    assert paths[1].parent.name == "usa"
    assert paths[2].parent.name == "western_europe"
    assert paths[3].parent.name == "north_africa"
    assert all(path.name == "prediction-000.nc" for path in paths)
    assert all(path.is_file() for path in paths)


def test_roi_batch_single_gpu_copy_per_step(tmp_path: Path) -> None:
    batch = _grid_batch()
    writer = RolloutStepWriter(
        tmp_path,
        ExportOptions(
            format="netcdf",
            roi_batch=RoiBatch.from_mapping(
                {
                    "china": Mask.china(),
                    "usa": Mask.usa(),
                }
            ),
        ),
    )
    with patch(
        "flash_aurora.engine.egress.step_writer.owned_cpu_copy",
        side_effect=lambda value: value,
    ) as owned_copy:
        writer.write_step(1, batch)
    owned_copy.assert_called_once()


def test_roi_batch_rejects_duplicate_region_names() -> None:
    mask = Mask.california()
    try:
        RoiBatch(regions=(("china", mask), ("china", mask)))
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("expected duplicate region error")
