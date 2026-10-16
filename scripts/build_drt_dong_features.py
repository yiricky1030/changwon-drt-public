"""Build, validate, publish, and optionally load the DRT feature foundation."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# QGIS-bundled Python keeps sqlite3.dll in the QGIS bin directory rather than
# beside the standard-library extension. Keep the DLL directory handle alive.
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

from src.export.geojson_properties import write_properties_geojson
from src.config import load_config
from src.database.connection import get_engine
from src.transform.build_drt_features import build_drt_features, validate_drt_features


RELEASE = ROOT / "data" / "release"
PROCESSED = ROOT / "data" / "processed"
REPORT = ROOT / "reports" / "changwon_drt_dong_features_qa.json"
FEATURE_PARQUET = RELEASE / "changwon_drt_dong_features.parquet"
FEATURE_GEOJSON = RELEASE / "changwon_drt_dong_features.geojson"
HOURLY_PARQUET = RELEASE / "changwon_drt_dong_hourly_profile.parquet"


def read_inputs() -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    context = gpd.read_parquet(RELEASE / "changwon_transit_demographic_context_by_dong.parquet")
    facts = pd.read_parquet(RELEASE / "fact_stop_ridership.parquet")
    profiles = pd.read_parquet(RELEASE / "changwon_transit_stop_profile.parquet")
    spatial = pd.read_parquet(PROCESSED / "stop_dong_spatial_join_qa.parquet")
    bridge = pd.read_parquet(RELEASE / "changwon_transit_route_stop.parquet")
    routes = gpd.read_parquet(RELEASE / "changwon_transit_route.parquet")
    return context, facts, profiles, spatial, bridge, routes


def route_qa(bridge: pd.DataFrame, spatial: pd.DataFrame) -> tuple[int, dict[str, int]]:
    labels = {
        "DIRECT_ID": "direct_id",
        "OFFICIAL_PREFIX_ID": "official_prefix_id",
        "COORDINATE_FALLBACK": "coordinate_fallback",
        "UNMATCHED": "unmatched",
        "AMBIGUOUS_COORDINATE": "ambiguous",
    }
    counts = bridge["match_method"].value_counts().to_dict()
    summary = {"total_relations": int(len(bridge))}
    summary.update({label: int(counts.get(method, 0)) for method, label in labels.items()})
    inside = set(spatial.loc[spatial["dong_code"].notna(), "stop_id"].astype(str))
    referenced = bridge.loc[
        bridge["is_matched"] & bridge["stop_id"].astype("string").isin(inside), "route_id"
    ].nunique()
    return int(referenced), summary


def publish(features: gpd.GeoDataFrame, hourly: pd.DataFrame, qa: dict, version: str) -> None:
    generated_at = datetime.now(timezone.utc).isoformat()
    features = features.copy()
    hourly = hourly.copy()
    features["dataset_version"] = version
    features["generated_at"] = generated_at
    hourly["dataset_version"] = version
    hourly["generated_at"] = generated_at
    features.to_parquet(FEATURE_PARQUET, index=False)
    hourly.to_parquet(HOURLY_PARQUET, index=False)

    write_properties_geojson(features, FEATURE_GEOJSON)

    qa.update({
        "dataset": "changwon_drt_dong_features",
        "dataset_version": version,
        "generated_at": generated_at,
        "status": "PASS" if not qa["errors"] else "FAIL",
        "outputs": {
            "feature_parquet": str(FEATURE_PARQUET.relative_to(ROOT)),
            "feature_geojson": str(FEATURE_GEOJSON.relative_to(ROOT)),
            "hourly_parquet": str(HOURLY_PARQUET.relative_to(ROOT)),
        },
    })
    REPORT.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_path = RELEASE / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["datasets"]["changwon_drt_dong_features"] = {
        "row_count": int(len(features)), "crs": "EPSG:4326", "primary_key": "adm_cd",
        "purpose": "explainable DRT feature foundation; no score or recommendation",
    }
    manifest["datasets"]["changwon_drt_dong_hourly_profile"] = {
        "row_count": int(len(hourly)), "primary_key": ["adm_cd", "hour"]
    }
    manifest["generated_at"] = generated_at
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def load_postgis(features: gpd.GeoDataFrame, hourly: pd.DataFrame) -> dict[str, object]:
    load_dotenv(ROOT / ".env")
    engine = get_engine()
    db_features = features.copy()
    for column in ("peak_hours", "peak_periods"):
        db_features[column] = db_features[column].map(
            lambda value: json.dumps(list(value), default=lambda item: item.item())
        )
    db_hourly = hourly.copy()
    db_hourly["adm_cd"] = db_hourly["adm_cd"].astype(str)
    counts = []
    with engine.begin() as connection:
        connection.execute(text("CREATE SCHEMA IF NOT EXISTS transport"))
    for _ in range(2):
        with engine.begin() as connection:
            db_features.to_postgis(
                "drt_dong_features", connection, schema="transport", if_exists="replace", index=False,
                dtype={"peak_hours": JSONB, "peak_periods": JSONB},
            )
            db_hourly.to_sql("drt_dong_hourly_profile", connection, schema="transport", if_exists="replace", index=False)
            connection.execute(text("ALTER TABLE transport.drt_dong_features ALTER COLUMN adm_cd SET NOT NULL"))
            connection.execute(text("ALTER TABLE transport.drt_dong_features ADD PRIMARY KEY (adm_cd)"))
            connection.execute(text("CREATE INDEX drt_dong_features_geometry_gix ON transport.drt_dong_features USING GIST (geometry)"))
            connection.execute(text("ALTER TABLE transport.drt_dong_hourly_profile ALTER COLUMN adm_cd SET NOT NULL"))
            connection.execute(text("ALTER TABLE transport.drt_dong_hourly_profile ALTER COLUMN hour SET NOT NULL"))
            connection.execute(text("ALTER TABLE transport.drt_dong_hourly_profile ADD PRIMARY KEY (adm_cd, hour)"))
            counts.append({
                "features": int(connection.execute(text("SELECT count(*) FROM transport.drt_dong_features")).scalar_one()),
                "hourly": int(connection.execute(text("SELECT count(*) FROM transport.drt_dong_hourly_profile")).scalar_one()),
            })
    with engine.connect() as connection:
        srid = int(connection.execute(text("SELECT DISTINCT ST_SRID(geometry) FROM transport.drt_dong_features")).scalar_one())
        invalid = int(connection.execute(text("SELECT count(*) FROM transport.drt_dong_features WHERE NOT ST_IsValid(geometry) OR geometry IS NULL")).scalar_one())
        feature_pk = bool(connection.execute(text("SELECT EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid='transport.drt_dong_features'::regclass AND contype='p')")).scalar_one())
        hourly_pk = bool(connection.execute(text("SELECT EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid='transport.drt_dong_hourly_profile'::regclass AND contype='p')")).scalar_one())
        gist = bool(connection.execute(text("SELECT EXISTS (SELECT 1 FROM pg_indexes WHERE schemaname='transport' AND tablename='drt_dong_features' AND indexdef ILIKE '%USING gist%')")).scalar_one())
    engine.dispose()
    return {
        "loaded": True, "refresh_counts": counts,
        "idempotent": counts == [{"features": 55, "hourly": 1320}] * 2,
        "feature_primary_key": feature_pk, "hourly_primary_key": hourly_pk,
        "geometry_gist_index": gist, "geometry_srid": srid, "invalid_or_null_geometry": invalid,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-db", action="store_true")
    args = parser.parse_args()
    config = load_config()
    context, facts, profiles, spatial, bridge, routes = read_inputs()
    features, hourly = build_drt_features(context, facts, profiles, spatial, bridge, routes)
    global_routes, matching_summary = route_qa(bridge, spatial)
    qa = validate_drt_features(features, hourly, global_routes, matching_summary)
    source_counts = {
        "official_master_stops": int(gpd.read_parquet(RELEASE / "dim_transit_stop.parquet").shape[0]),
        "routes": int(len(routes)), "route_stop_relations": int(len(bridge)),
        "fact_rows": int(len(facts)), "profile_stops": int(len(profiles)), "dongs": int(len(context)),
        "population_total": int(context["population_total"].sum()),
        "population_65_plus": int(context["population_65_plus"].sum()),
        "context_master_stops": int(context["master_stop_count"].sum()),
        "context_observed_stops": int(context["observed_stop_count"].sum()),
        "context_analysis_ready_stops": int(context["analysis_ready_stop_count"].sum()),
    }
    expected_counts = {
        "official_master_stops": 3526, "routes": 170, "route_stop_relations": 17026,
        "fact_rows": 171072, "profile_stops": 889, "dongs": 55,
        "population_total": 1001043, "population_65_plus": 189930,
        "context_master_stops": 2926, "context_observed_stops": 888,
        "context_analysis_ready_stops": 886,
    }
    qa["invariants"] = {"actual": source_counts, "expected": expected_counts, "all_match": source_counts == expected_counts}
    if source_counts != expected_counts:
        raise AssertionError("Source invariant mismatch")

    hourly_total = hourly.groupby("adm_cd")["ridership_total"].sum(min_count=1).astype("Float64")
    feature_total = features.set_index("adm_cd")["ridership_total"].astype("Float64")
    # Nullable `!=` returns NA (dropped by fillna(False)) whenever either side is NA, so a plain
    # comparison would silently pass a NaN-vs-real-number mismatch -- exactly the corruption this
    # invariant exists to catch. Missingness must also match on both sides.
    value_mismatch = (hourly_total != feature_total).fillna(False) & hourly_total.notna() & feature_total.notna()
    missingness_mismatch = hourly_total.isna() != feature_total.isna()
    mismatches = value_mismatch | missingness_mismatch
    qa["ridership"]["hourly_feature_total_mismatch_count"] = int(mismatches.sum())
    if mismatches.any():
        raise AssertionError("Hourly and feature ridership totals differ")

    version = str(config["project"]["version"])
    publish(features, hourly, qa, version)
    features = gpd.read_parquet(FEATURE_PARQUET)
    hourly = pd.read_parquet(HOURLY_PARQUET)
    db_result = {"loaded": False, "reason": "--skip-db"}
    if not args.skip_db:
        db_result = load_postgis(features, hourly)
        if not db_result["idempotent"]:
            raise AssertionError("PostGIS full refresh was not idempotent")
    qa = json.loads(REPORT.read_text(encoding="utf-8"))
    qa["postgis"] = db_result
    REPORT.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DRT features: {len(features)}, hourly: {len(hourly)}, QA: {qa['status']}, DB loaded: {db_result['loaded']}")


if __name__ == "__main__":
    main()
