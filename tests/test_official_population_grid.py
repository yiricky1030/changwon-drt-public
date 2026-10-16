"""Unit tests for official-population-grid ingestion using tiny synthetic fixtures.

These fixtures are NOT real NGII data - they exist only to pin down the
encoding and NULL-vs-masked-vs-missing parsing contract in isolation, without
needing the real (large, license-restricted, gitignored) shapefiles on disk.
"""
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from src.analysis.spatial_validation import (
    anchor_population_context,
    merge_population_variables,
    official_grid_dong_comparison,
    read_official_population_grid,
    select_population_weighted_origins,
)


def _write_grid_shp(path, rows):
    frame = gpd.GeoDataFrame(rows, crs=5179)
    frame.to_file(path, driver="ESRI Shapefile", encoding="utf-8")
    return path


def _cell(x, y, size, gid, lbl, val):
    return {"gid": gid, "lbl": lbl, "val": val, "geometry": box(x, y, x + size, y + size)}


def test_read_official_population_grid_separates_known_masked_and_missing(tmp_path):
    rows = [
        _cell(0, 0, 500, "known1", "42.00", 42.0),
        _cell(500, 0, 500, "masked1", "N/A", 0.0),
        _cell(0, 500, 500, "nodata1", None, None),
    ]
    path = _write_grid_shp(tmp_path / "grid.shp", rows)
    grid = read_official_population_grid(path)
    by_id = grid.set_index("grid_id")
    assert by_id.loc["known1", "status"] == "KNOWN"
    assert by_id.loc["known1", "value"] == 42.0
    assert by_id.loc["masked1", "status"] == "MASKED_SUPPRESSED"
    assert pd.isna(by_id.loc["masked1", "value"])  # the placeholder 0 must not survive as a real value
    assert by_id.loc["nodata1", "status"] == "NO_DATA"
    assert pd.isna(by_id.loc["nodata1", "value"])


def test_read_official_population_grid_rejects_wrong_crs(tmp_path):
    rows = [_cell(0, 0, 500, "a", "1.00", 1.0)]
    path = _write_grid_shp(tmp_path / "grid.shp", rows)
    with pytest.raises(ValueError):
        read_official_population_grid(path, source_crs=5186)


def test_merge_population_variables_outer_join_keeps_mismatched_cells(tmp_path):
    total_rows = [_cell(0, 0, 500, "a", "10.00", 10.0), _cell(500, 0, 500, "b", "20.00", 20.0)]
    elderly_rows = [_cell(0, 0, 500, "a", "3.00", 3.0)]  # cell "b" absent from elderly vintage
    total = read_official_population_grid(_write_grid_shp(tmp_path / "t.shp", total_rows))
    elderly = read_official_population_grid(_write_grid_shp(tmp_path / "e.shp", elderly_rows))
    merged = merge_population_variables(total, elderly)
    by_id = merged.set_index("grid_id")
    assert by_id.loc["a", "population_total"] == 10.0
    assert by_id.loc["a", "population_65_plus"] == 3.0
    assert by_id.loc["b", "population_total"] == 20.0
    assert by_id.loc["b", "population_65_plus_status"] == "NO_DATA"
    assert pd.isna(by_id.loc["b", "population_65_plus"])
    assert by_id.geometry.notna().all()  # geometry backfilled even when one source lacked the cell


def _make_dongs(entries):
    return gpd.GeoDataFrame(entries, crs=5179)


def test_official_grid_dong_comparison_excludes_masked_from_sum_and_reports_counts(tmp_path):
    dongs = _make_dongs([
        {"adm_cd": "D1", "adm_nm": "동1", "population_total": 100, "population_65_plus": 20,
         "geometry": box(0, 0, 1000, 1000)},
    ])
    total_rows = [
        _cell(100, 100, 500, "a", "60.00", 60.0),
        _cell(600, 100, 500, "b", "N/A", 0.0),
    ]
    elderly_rows = [
        _cell(100, 100, 500, "a", "15.00", 15.0),
        _cell(600, 100, 500, "b", "N/A", 0.0),
    ]
    total = read_official_population_grid(_write_grid_shp(tmp_path / "t.shp", total_rows))
    elderly = read_official_population_grid(_write_grid_shp(tmp_path / "e.shp", elderly_rows))
    grid = merge_population_variables(total, elderly)
    comparison = official_grid_dong_comparison(grid, dongs).set_index("adm_cd")
    row = comparison.loc["D1"]
    assert row.grid_sum_population_total == 60.0  # masked cell "b" excluded from the sum
    assert row.population_total_cells_known == 1
    assert row.population_total_cells_masked_suppressed == 1
    assert row.population_total_ratio_grid_over_canonical == pytest.approx(0.6)


def test_select_population_weighted_origins_only_uses_known_positive_cells(tmp_path):
    dongs = _make_dongs([
        {"adm_cd": "D1", "adm_nm": "동1", "geometry": box(0, 0, 2000, 2000)},
    ])
    rows = [
        _cell(100, 100, 500, "a", "5.00", 5.0),
        _cell(1200, 100, 500, "b", "0.00", 0.0),  # KNOWN but zero population: not a residential origin
        _cell(100, 1200, 500, "c", "N/A", 0.0),
    ]
    total = read_official_population_grid(_write_grid_shp(tmp_path / "t.shp", rows))
    total["population_total"] = total.value
    total["population_total_status"] = total.status
    origins = select_population_weighted_origins(total, dongs, count=5)
    assert set(origins.grid_id) == {"a"}


def test_anchor_population_context_reports_status_counts_not_zero(tmp_path):
    anchors = gpd.GeoDataFrame(
        [{"stop_id": "S1", "stop_name": "테스트", "adm_cd": "D1", "adm_nm": "동1", "geometry": Point(250, 250)}],
        crs=5179)
    total_rows = [_cell(0, 0, 500, "a", "8.00", 8.0), _cell(500, 0, 500, "b", "N/A", 0.0)]
    elderly_rows = [_cell(0, 0, 500, "a", "2.00", 2.0), _cell(500, 0, 500, "b", "N/A", 0.0)]
    total = read_official_population_grid(_write_grid_shp(tmp_path / "t.shp", total_rows))
    elderly = read_official_population_grid(_write_grid_shp(tmp_path / "e.shp", elderly_rows))
    grid = merge_population_variables(total, elderly)
    context = anchor_population_context(anchors, grid, radii_m=(800,))
    row = context.iloc[0]
    assert row.population_total_context_sum == 8.0
    assert row.population_total_cells_masked_suppressed >= 1
