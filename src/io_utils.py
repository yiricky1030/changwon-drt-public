"""Read GeoJSON attribute tables without a geopandas/pyarrow dependency.

Some analysis environments cannot load the native GDAL/Arrow extensions that
geopandas and pyarrow require. Every release dataset that carries geometry is
also published as GeoJSON, and GeoJSON is plain JSON, so its ``properties``
can be read with the standard library and turned into a DataFrame directly.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def read_geojson_properties(path: Path) -> pd.DataFrame:
    """Return one row per feature using only its non-geometry properties."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    frame = pd.DataFrame([feature["properties"] for feature in payload["features"]])
    # Legacy exporter wrote nullable object numerics as strings, including <NA>.
    numeric = [c for c in frame if c.endswith(("_share", "_ridership", "_ratio", "_pct")) or c in ("top3_hour_share", "ridership_total")]
    for column in numeric:
        values = frame[column].replace({"<NA>": None, "NaN": None})
        frame[column] = pd.to_numeric(values, errors="raise")
    return frame
