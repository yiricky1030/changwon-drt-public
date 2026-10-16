import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from src.export.release_manifest import build_release_manifest
from src.transform.build_stop_profile import build_stop_profile


BANDS = {
    "morning": [5, 6, 7, 8, 9],
    "daytime": [10, 11, 12, 13, 14, 15, 16],
    "evening": [17, 18, 19, 20, 21, 22, 23],
    "late_night": [0, 1, 2, 3, 4],
}


def sample_profile():
    master = gpd.GeoDataFrame(
        {
            "stop_id": ["valid", "missing", "duplicate"],
            "stop_name": ["정상", "결측", "중복"],
            "direction": [None, None, None],
            "longitude": [128.6, 128.61, 128.62],
            "latitude": [35.2, 35.21, 35.22],
        },
        geometry=[Point(128.6, 35.2), Point(128.61, 35.21), Point(128.62, 35.22)],
        crs="EPSG:4326",
    )
    date = pd.Timestamp("2025-08-18").date()
    fact = pd.DataFrame(
        {
            "stop_id": ["valid", "missing", "missing", "duplicate", "duplicate"],
            "service_date": [date, date, date, date, date],
            "hour": [5, 3, 5, 6, 6],
            "ridership": pd.Series([10, pd.NA, 3, 4, 7], dtype="Int64"),
        }
    )
    return build_stop_profile(master, fact, BANDS)


def test_duplicate_stop_profile_metrics_are_null():
    row = sample_profile().set_index("stop_id").loc["duplicate"]
    metrics = ["total_ridership", "avg_daily_ridership", "weekday_ridership", "weekend_ridership", "morning_ridership", "daytime_ridership", "evening_ridership", "late_night_ridership", "peak_hour", "peak_hour_ridership"]
    assert row[metrics].isna().all()


def test_duplicate_stop_not_analysis_ready():
    row = sample_profile().set_index("stop_id").loc["duplicate"]
    assert row["is_analysis_ready"] == False
    assert row["duplicate_observation_count"] == 2


def test_missing_observations_preserve_valid_totals():
    row = sample_profile().set_index("stop_id").loc["missing"]
    assert row["quality_status"] == "MISSING_OBSERVATIONS"
    assert row["is_analysis_ready"] == True
    assert row["missing_observation_count"] == 1
    assert row["total_ridership"] == 3


def test_release_manifest_counts():
    manifest = build_release_manifest(sample_profile(), "test", 10)
    assert manifest["row_count"] == 3
    assert manifest["analysis_ready_rows"] == 2
    assert manifest["non_analysis_ready_rows"] == 1
    assert manifest["coverage"]["observed_stop_count"] == 3
    assert manifest["coverage"]["observed_stop_ratio"] == 0.3


def test_profile_quality_status():
    status = sample_profile().set_index("stop_id")["quality_status"].to_dict()
    assert status == {
        "valid": "VALID",
        "missing": "MISSING_OBSERVATIONS",
        "duplicate": "DUPLICATE_OBSERVATIONS",
    }
