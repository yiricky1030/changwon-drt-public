"""Build, publish, validate, and load stop-level DRT feature foundations."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_DLL_HANDLES = []
if os.name == "nt" and hasattr(os, "add_dll_directory"):
    qgis_bin = Path(sys.base_prefix).parents[1] / "bin"
    if qgis_bin.exists():
        _DLL_HANDLES.append(os.add_dll_directory(str(qgis_bin)))

import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import JSONB

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.database.connection import get_engine
from src.transform.build_drt_stop_features import ROUTE_TYPES, build_stop_features, validate_stop_features

RELEASE = ROOT / "data" / "release"
PROCESSED = ROOT / "data" / "processed"
FEATURE_PARQUET = RELEASE / "changwon_drt_stop_features.parquet"
FEATURE_GEOJSON = RELEASE / "changwon_drt_stop_features.geojson"
HOURLY_PARQUET = RELEASE / "changwon_drt_stop_hourly_profile.parquet"
QA_PATH = ROOT / "reports" / "changwon_drt_stop_features_qa.json"


def load_inputs():
    return (
        gpd.read_parquet(RELEASE / "dim_transit_stop.parquet"),
        pd.read_parquet(PROCESSED / "stop_dong_spatial_join_qa.parquet"),
        gpd.read_parquet(RELEASE / "changwon_transit_stop_profile.parquet"),
        pd.read_parquet(RELEASE / "fact_stop_ridership.parquet"),
        pd.read_parquet(RELEASE / "changwon_transit_route_stop.parquet"),
        gpd.read_parquet(RELEASE / "changwon_transit_route.parquet"),
        gpd.read_parquet(RELEASE / "changwon_transit_demographic_context_by_dong.parquet"),
    )


def stats(series: pd.Series) -> dict[str, float]:
    values = series.dropna()
    return {"min": float(values.min()), "median": float(values.median()), "max": float(values.max())}


def build_qa(features, hourly, master, profiles, facts, bridge, routes) -> dict:
    route_type_summary = {column: stats(features[column]) for column in ROUTE_TYPES.values()}
    percentile_columns = [column for column in features if "_pct_" in column]
    zero_distance = features.loc[features["nearest_stop_distance_m"].eq(0), ["stop_id", "nearest_stop_id"]].to_dict("records")
    return {
        "dataset": "changwon_drt_stop_features", "status": "PASS", "errors": [],
        "rows": int(len(features)), "expected_rows": 2926,
        "unique_stop_id": int(features["stop_id"].nunique()), "null_stop_id": int(features["stop_id"].isna().sum()),
        "geometry": {"null": int(features.geometry.isna().sum()), "invalid": int((~features.geometry.is_valid).sum()), "srid": 4326},
        "adm": {"null_adm_cd": int(features["adm_cd"].isna().sum()), "unique_adm_count": int(features["adm_cd"].nunique())},
        "status_distribution": features["quality_status"].value_counts().to_dict(),
        "anchor_data_status_distribution": features["anchor_data_status"].value_counts().to_dict(),
        "route": {"route_count": stats(features["route_count"]), "route_count_zero": int(features["route_count"].eq(0).sum()), "distinct_routes_referenced": int(routes["route_id"].nunique()), "route_type_stop_distribution": route_type_summary},
        "neighborhood": {
            "nearby_stops_300m": stats(features["nearby_stops_300m"]), "nearby_stops_500m": stats(features["nearby_stops_500m"]),
            "nearest_stop_distance_m": stats(features["nearest_stop_distance_m"]), "zero_distance_distinct_stop_pairs": zero_distance,
            "zero_neighbor_300m": int(features["nearby_stops_300m"].eq(0).sum()), "zero_neighbor_500m": int(features["nearby_stops_500m"].eq(0).sum()),
            "nearest_observed_stop_distance_m": stats(features["nearest_observed_stop_distance_m"]),
            "nearby_observed_stops_500m": stats(features["nearby_observed_stops_500m"]),
        },
        "ridership": {
            "non_null_stop_count": int(features["ridership_total"].notna().sum()), "total_sum": int(features["ridership_total"].sum()),
            "duplicate_excluded_count": int(features["is_duplicate_excluded"].sum()),
            "master_only_null_check": bool(features.loc[features["quality_status"].eq("MASTER_ONLY"), "ridership_total"].isna().all()),
            "boardings_available": False, "alightings_available": False,
        },
        "temporal": {"hourly_rows": int(len(hourly)), "expected_hourly_rows": 21264, "peak_tie_count": int(features["peak_hours"].dropna().map(len).gt(1).sum()), "top3_hour_share": stats(features["top3_hour_share"])},
        "percentiles": {column: {**stats(features[column]), "null": int(features[column].isna().sum())} for column in percentile_columns},
        "boundary": {"near_boundary_300m": int(features["near_city_boundary_300m"].sum()), "near_boundary_500m": int(features["near_city_boundary_500m"].sum()), "distance_m": stats(features["distance_to_changwon_boundary_m"])},
        "invariants": {
            "official_master_total": int(len(master)), "changwon_inside_master": int(len(features)),
            "profile_total": int(len(profiles)), "observed_inside": int(features["has_observed_profile"].sum()),
            "analysis_ready_inside": int(features["analysis_ready"].sum()), "routes": int(len(routes)),
            "route_stop_relations": int(len(bridge)), "facts": int(len(facts)), "drt_dong": 55,
        },
        "warnings": [
            "TAGO route metrics use matched relations only; route_count=0 does not prove that no bus service exists.",
            "Neighborhoods are straight-line EPSG:5179 distances and exclude the 600 master stops outside Changwon.",
            "The source fact has a combined ridership field only; boardings_total and alightings_total remain NULL.",
            "Relative scarcity and isolation fields are descriptive features, not DRT suitability scores.",
        ],
    }


def publish(features, hourly, qa, version):
    generated_at = datetime.now(timezone.utc).isoformat()
    features = features.copy(); hourly = hourly.copy()
    features["dataset_version"] = version; features["generated_at"] = generated_at
    hourly["dataset_version"] = version; hourly["generated_at"] = generated_at
    features.to_parquet(FEATURE_PARQUET, index=False); hourly.to_parquet(HOURLY_PARQUET, index=False)
    geojson = features.copy()
    for column in ("peak_hours", "peak_periods"):
        geojson[column] = geojson[column].map(lambda value: None if not isinstance(value, (list, tuple)) else json.dumps(value, ensure_ascii=False))
    geojson.to_file(FEATURE_GEOJSON, driver="GeoJSON", index=False)
    qa.update({"dataset_version": version, "generated_at": generated_at, "outputs": {"parquet": str(FEATURE_PARQUET.relative_to(ROOT)), "geojson": str(FEATURE_GEOJSON.relative_to(ROOT)), "hourly": str(HOURLY_PARQUET.relative_to(ROOT))}})
    QA_PATH.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path = RELEASE / "release_manifest.json"; manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["datasets"]["changwon_drt_stop_features"] = {"row_count": 2926, "crs": "EPSG:4326", "primary_key": "stop_id", "purpose": "DRT anchor feature foundation; no score or recommendation"}
    manifest["datasets"]["changwon_drt_stop_hourly_profile"] = {"row_count": 21264, "primary_key": ["stop_id", "hour"]}
    manifest["generated_at"] = generated_at; manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def load_postgis(features, hourly):
    load_dotenv(ROOT / ".env"); engine = get_engine(); counts = []
    db_features = features.copy()
    for column in ("peak_hours", "peak_periods"):
        db_features[column] = db_features[column].map(lambda value: None if value is None else json.dumps(list(value), default=lambda item: item.item()))
    for _ in range(2):
        with engine.begin() as connection:
            db_features.to_postgis("drt_stop_features", connection, schema="transport", if_exists="replace", index=False, dtype={"peak_hours": JSONB, "peak_periods": JSONB})
            hourly.to_sql("drt_stop_hourly_profile", connection, schema="transport", if_exists="replace", index=False)
            connection.execute(text("ALTER TABLE transport.drt_stop_features ALTER COLUMN stop_id SET NOT NULL"))
            connection.execute(text("ALTER TABLE transport.drt_stop_features ADD PRIMARY KEY (stop_id)"))
            connection.execute(text("CREATE INDEX drt_stop_features_geometry_gix ON transport.drt_stop_features USING GIST (geometry)"))
            connection.execute(text("CREATE INDEX drt_stop_features_adm_cd_idx ON transport.drt_stop_features (adm_cd)"))
            connection.execute(text("CREATE INDEX drt_stop_features_status_idx ON transport.drt_stop_features (anchor_data_status)"))
            connection.execute(text("ALTER TABLE transport.drt_stop_hourly_profile ALTER COLUMN stop_id SET NOT NULL"))
            connection.execute(text("ALTER TABLE transport.drt_stop_hourly_profile ALTER COLUMN hour SET NOT NULL"))
            connection.execute(text("ALTER TABLE transport.drt_stop_hourly_profile ADD PRIMARY KEY (stop_id, hour)"))
            connection.execute(text("CREATE INDEX drt_stop_hourly_adm_cd_idx ON transport.drt_stop_hourly_profile (adm_cd)"))
            counts.append({"features": int(connection.execute(text("SELECT count(*) FROM transport.drt_stop_features")).scalar_one()), "hourly": int(connection.execute(text("SELECT count(*) FROM transport.drt_stop_hourly_profile")).scalar_one())})
    with engine.connect() as connection:
        checks = {"feature_pk": bool(connection.execute(text("SELECT EXISTS(SELECT 1 FROM pg_constraint WHERE conrelid='transport.drt_stop_features'::regclass AND contype='p')")).scalar_one()), "hourly_pk": bool(connection.execute(text("SELECT EXISTS(SELECT 1 FROM pg_constraint WHERE conrelid='transport.drt_stop_hourly_profile'::regclass AND contype='p')")).scalar_one()), "geometry_gist": bool(connection.execute(text("SELECT EXISTS(SELECT 1 FROM pg_indexes WHERE schemaname='transport' AND tablename='drt_stop_features' AND indexdef ILIKE '%USING gist%')")).scalar_one()), "srid": int(connection.execute(text("SELECT DISTINCT ST_SRID(geometry) FROM transport.drt_stop_features")).scalar_one())}
    engine.dispose()
    return {"loaded": True, "refresh_counts": counts, "idempotent": counts == [{"features": 2926, "hourly": 21264}] * 2, **checks}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--skip-db", action="store_true"); parser.add_argument("--db-only", action="store_true"); args = parser.parse_args()
    if args.db_only:
        db = load_postgis(gpd.read_parquet(FEATURE_PARQUET), pd.read_parquet(HOURLY_PARQUET))
        assert db["idempotent"] and db["feature_pk"] and db["hourly_pk"] and db["geometry_gist"] and db["srid"] == 4326
        qa = json.loads(QA_PATH.read_text(encoding="utf-8")); qa["postgis"] = db; QA_PATH.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"DRT stop features: 2926, hourly: 21264, QA: {qa['status']}, DB loaded: True")
        return
    version = str(load_config()["project"]["version"])
    master, spatial, profiles, facts, bridge, routes, city = load_inputs()
    features, hourly = build_stop_features(master, spatial, profiles, facts, bridge, routes, city, version)
    validate_stop_features(features, hourly)
    qa = build_qa(features, hourly, master, profiles, facts, bridge, routes)
    publish(features, hourly, qa, version)
    db = {"loaded": False, "reason": "--skip-db"}
    if not args.skip_db:
        db = load_postgis(gpd.read_parquet(FEATURE_PARQUET), pd.read_parquet(HOURLY_PARQUET))
        assert db["idempotent"] and db["feature_pk"] and db["hourly_pk"] and db["geometry_gist"] and db["srid"] == 4326
    qa = json.loads(QA_PATH.read_text(encoding="utf-8")); qa["postgis"] = db; QA_PATH.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DRT stop features: {len(features)}, hourly: {len(hourly)}, QA: {qa['status']}, DB loaded: {db['loaded']}")


if __name__ == "__main__":
    main()
