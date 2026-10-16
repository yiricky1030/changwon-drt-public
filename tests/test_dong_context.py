import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box

from src.export.release_manifest import add_context_manifest_entry
from src.quality.analyze_dong_lineage import classify_outside_regions
from src.transform.build_dong_context import (
    build_transit_context_by_dong,
    safe_per_1000,
    spatial_join_stops_to_dong,
)


def fixtures():
    projected = gpd.GeoDataFrame(
        {"dong_code":["38111001", "38111002"], "dong_name":["테스트동", "빈동"], "reference_date":["20250630", "20250630"]},
        geometry=[box(200000, 300000, 201000, 301000), box(201000, 300000, 202000, 301000)],
        crs="EPSG:5179",
    )
    boundary = projected.to_crs("EPSG:4326")
    inside = gpd.GeoSeries([Point(200500, 300500)], crs="EPSG:5179").to_crs("EPSG:4326").iloc[0]
    outside = gpd.GeoSeries([Point(210000, 310000)], crs="EPSG:5179").to_crs("EPSG:4326").iloc[0]
    stops = gpd.GeoDataFrame(
        {"stop_id":["valid", "duplicate", "outside"], "master_dong_value":["테스트동", "테스트동", None]},
        geometry=[inside, inside, outside], crs="EPSG:4326",
    )
    profile = gpd.GeoDataFrame(
        {
            "stop_id":["valid", "duplicate"], "quality_status":["VALID", "DUPLICATE_OBSERVATIONS"],
            "is_analysis_ready":[True, False], "total_ridership":pd.Series([10, pd.NA], dtype="Int64"),
            "weekday_ridership":pd.Series([7, pd.NA], dtype="Int64"), "weekend_ridership":pd.Series([3, pd.NA], dtype="Int64"),
            "morning_ridership":pd.Series([5, pd.NA], dtype="Int64"), "daytime_ridership":pd.Series([3, pd.NA], dtype="Int64"),
            "evening_ridership":pd.Series([2, pd.NA], dtype="Int64"), "late_night_ridership":pd.Series([0, pd.NA], dtype="Int64"),
        }, geometry=[inside, inside], crs="EPSG:4326",
    )
    fact = pd.DataFrame({"stop_id":["valid", "valid", "duplicate"], "hour":[7, 8, 7], "ridership":pd.Series([5, 5, 99], dtype="Int64")})
    join = spatial_join_stops_to_dong(stops, boundary)
    context = build_transit_context_by_dong(boundary, join, profile, fact, "EPSG:5179", "test", "test")
    return boundary, stops, join, profile, fact, context


def test_spatial_join_preserves_unmatched():
    _, stops, join, _, _, _ = fixtures()
    assert len(join) == len(stops)
    assert join.set_index("stop_id").loc["outside", "match_status"] == "OUTSIDE_BOUNDARY"


def test_duplicate_profile_metrics_not_aggregated():
    *_, context = fixtures()
    row = context.set_index("dong_code").loc["38111001"]
    assert row.total_ridership == 10
    assert row.duplicate_observation_stop_count == 1


def test_context_observed_stop_count():
    *_, context = fixtures()
    row = context.set_index("dong_code").loc["38111001"]
    assert row.master_stop_count == 2
    assert row.observed_stop_count == 2
    assert row.analysis_ready_stop_count == 1
    assert row.peak_hours == [7, 8]


def test_area_not_calculated_in_epsg4326():
    *_, context = fixtures()
    area = context.set_index("dong_code").loc["38111001", "area_sq_km"]
    assert 0.99 < area < 1.01


def test_population_division_by_zero_returns_null():
    result = safe_per_1000(pd.Series([10.0, 10.0]), pd.Series([0.0, 100.0]))
    assert pd.isna(result.iloc[0])
    assert result.iloc[1] == 100.0


def test_release_context_geometry():
    *_, context = fixtures()
    assert context.crs.to_epsg() == 4326
    assert set(context.geom_type).issubset({"Polygon", "MultiPolygon"})
    assert context.geometry.is_valid.all()


def test_context_manifest_entry():
    *_, context = fixtures()
    manifest = add_context_manifest_entry({}, context)
    entry = manifest["datasets"]["changwon_transit_context_by_dong"]
    assert entry["row_count"] == 2
    assert entry["sum_master_stop_count"] == 2


def test_master_dong_value_not_used_as_join_key():
    boundary, stops, *_ = fixtures()
    altered = stops.copy()
    altered.loc[altered.stop_id.eq("valid"), "master_dong_value"] = "완전히다른명칭"
    joined = spatial_join_stops_to_dong(altered, boundary).set_index("stop_id")
    assert joined.loc["valid", "dong_code"] == "38111001"
    assert joined.loc["valid", "match_status"] == "MASTER_NAME_MISMATCH"


def test_outside_boundary_count_preserved():
    _, stops, join, *_ = fixtures()
    assert len(stops) - int(join.dong_code.notna().sum()) == int(join.match_status.eq("OUTSIDE_BOUNDARY").sum()) == 1


def test_master_name_status_counts_sum_to_spatial_matches():
    *_, join, _, _, _ = fixtures()
    assigned = int(join.dong_code.notna().sum())
    statuses = int(join.match_status.isin(["MATCH", "MASTER_NAME_MISMATCH", "MISSING_MASTER_VALUE"]).sum())
    assert statuses == assigned


def test_context_uses_official_adm_code():
    *_, context = fixtures()
    assert set(context.dong_code) == {"38111001", "38111002"}
    assert "테스트동" not in set(context.dong_code)


def test_outside_region_classification_preserves_all_rows():
    boundary, _, join, *_ = fixtures()
    outside_polygon = gpd.GeoDataFrame(
        {"dong_code":["38070001"], "dong_name":["외부동"]},
        geometry=[box(209000, 309000, 211000, 311000)], crs="EPSG:5179",
    ).to_crs("EPSG:4326")
    national = pd.concat([boundary, outside_polygon], ignore_index=True)
    national = gpd.GeoDataFrame(national, geometry="geometry", crs="EPSG:4326")
    result = classify_outside_regions(join, national, {"38070":"김해시"})
    assert len(result) == int(join.match_status.eq("OUTSIDE_BOUNDARY").sum()) == 1
    assert result.iloc[0].actual_region == "김해시"
