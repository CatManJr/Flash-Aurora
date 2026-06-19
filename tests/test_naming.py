from __future__ import annotations

from engine.egress.naming import PredictionNaming


def test_prediction_filename() -> None:
    naming = PredictionNaming()
    assert naming.filename(0) == "prediction-000.nc"
    assert naming.filename(12) == "prediction-012.nc"
