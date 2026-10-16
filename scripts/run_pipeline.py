"""Command-line entry point for the Changwon transit pipeline."""
from __future__ import annotations
import argparse
import logging
import os
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# A venv created from QGIS Python needs its adjacent binary directory for
# extension-module dependencies such as sqlite3.dll. Retain handles for the
# process lifetime and do nothing for standard Python installations.
DLL_DIRECTORY_HANDLES = []
if os.name == "nt" and hasattr(os, "add_dll_directory"):
    qgis_bin = Path(sys.base_prefix).parents[1] / "bin"
    if qgis_bin.exists():
        DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(qgis_bin)))

from src.config import load_config
from src.database.connection import get_engine
from src.database.load_postgis import load_postgis
from src.database.validate_postgis import expected_from_release, validate_database, write_database_report
from src.export.export_dataset import export_context, export_demographic_context, export_releases
from src.export.release_manifest import add_context_manifest_entry, add_demographic_manifest_entry, build_release_manifest, write_release_manifest
from src.ingest.load_boundary import extract_changwon_boundary, load_dong_boundary
from src.ingest.load_ridership import load_ridership
from src.ingest.load_stop_master import load_stop_master
from src.quality.generate_quality_report import build_quality_report, write_quality_report
from src.quality.generate_context_quality_report import build_context_quality_report, write_context_quality_report
from src.quality.generate_demographic_quality_report import build_demographic_quality_report, write_demographic_quality_report
from src.quality.analyze_dong_lineage import build_lineage_outputs, write_lineage_csvs
from src.quality.validate_ridership import validate_ridership
from src.quality.validate_stop_master import validate_stop_master
from src.transform.build_stop_profile import build_stop_profile
from src.transform.build_dong_context import build_transit_context_by_dong, spatial_join_stops_to_dong
from src.transform.build_demographic_context import build_demographic_context
from src.transform.match_stops import match_stops
from src.transform.normalize_stop import normalize_stop_master
from src.transform.reshape_ridership import annotate_duplicate_observations, wide_to_long

LOGGER = logging.getLogger("changwon_transit")


def parse_args() -> argparse.Namespace:
    """Parse supported execution modes."""
    parser = argparse.ArgumentParser(description="Build trusted Changwon transit datasets")
    parser.add_argument("--skip-db", action="store_true", help="Do not load PostGIS")
    parser.add_argument("--validate-only", action="store_true", help="Stop after quality report")
    parser.add_argument("--db-validate-only", action="store_true", help="Validate current PostGIS release without reloading")
    parser.add_argument("--config", type=Path, help="Alternative settings YAML")
    return parser.parse_args()


def main() -> int:
    """Execute ingestion, validation, matching, reshape, QA and optional release."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    if args.db_validate_only:
        expected_db = expected_from_release(config["paths"]["release_dir"], str(config["project"]["version"]))
        database_report = validate_database(get_engine(), config["database"]["schema"], expected_db)
        write_database_report(database_report, config["paths"]["database_report"])
        LOGGER.info("Database validation status: %s", database_report["status"])
        return 0 if database_report["status"] == "PASS" else 1
    stop_raw, stop_encoding = load_stop_master(config["paths"]["stop_master"])
    LOGGER.info("Loaded stop master: %s rows", len(stop_raw))
    stops = normalize_stop_master(stop_raw)
    stop_metrics = validate_stop_master(stops)
    rider_raw, rider_encoding = load_ridership(config["paths"]["ridership"])
    LOGGER.info("Loaded ridership: %s rows", len(rider_raw))
    matched, match_qa = match_stops(rider_raw, stops, float(config["matching"]["coordinate_tolerance_m"]), config["matching"]["projected_crs"])
    for status, count in match_qa.match_status.value_counts().items():
        LOGGER.info("%s stop matches: %s", status.title(), count)
    long, reshape_metrics = wide_to_long(matched)
    long.insert(0, "record_id", range(1, len(long) + 1))
    long = annotate_duplicate_observations(long)
    long["source"] = "ridership.csv"
    long["loaded_at"] = stops.loaded_at.iloc[0]
    rider_metrics = validate_ridership(rider_raw, long, match_qa)
    processed = config["paths"]["processed_dir"]
    processed.mkdir(parents=True, exist_ok=True)
    match_qa.to_parquet(processed / "stop_matching_qa.parquet", index=False)
    duplicate_mask = long.duplicated(["stop_id", "service_date", "hour"], keep=False) & long["stop_id"].notna()
    duplicate_records = long.loc[duplicate_mask, ["stop_id", "stop_name_normalized", "service_date", "hour", "ridership"]].copy()
    duplicate_records = duplicate_records.rename(columns={"stop_name_normalized": "stop_name"})
    duplicate_records["duplicate_group_size"] = duplicate_records.groupby(["stop_id", "service_date", "hour"])["stop_id"].transform("size")
    duplicate_report_path = config["paths"]["report"].parent / "duplicate_ridership_records.csv"
    duplicate_records.sort_values(["stop_id", "service_date", "hour", "ridership"], na_position="last").to_csv(duplicate_report_path, index=False, encoding="utf-8-sig")
    LOGGER.warning("NULL ridership observations: %s", rider_metrics["null_ridership"])
    if args.validate_only:
        report = build_quality_report(stop_metrics, rider_metrics, match_qa, config, {"stop_master":stop_encoding, "ridership":rider_encoding}, reshape_metrics)
        write_quality_report(report, config["paths"]["report"])
        LOGGER.info("Quality status: %s", report["status"])
        return 1 if report["status"] == "FAIL" else 0
    profile = build_stop_profile(stops, long, config["time_bands"])
    dim_columns = ["stop_id", "service_id", "stop_name", "direction", "longitude", "latitude", "master_dong_value", "source_created_at", "source_modified_at", "geometry", "source", "loaded_at"]
    stops_release = stops[[column for column in dim_columns if column in stops.columns]].copy()
    fact_columns = ["record_id", "stop_id", "service_date", "hour", "ridership", "is_duplicate_observation", "duplicate_group_size", "source", "loaded_at"]
    export_releases(stops_release, long[fact_columns], profile, config["paths"]["release_dir"])
    boundary_config = config["boundary"]
    national_boundary = load_dong_boundary(
        config["paths"]["boundary_zip"], boundary_config["layer"],
        boundary_config["code_field"], boundary_config["name_field"],
        boundary_config["reference_date_field"],
    )
    boundary = extract_changwon_boundary(national_boundary, boundary_config["changwon_sigungu_prefixes"])
    stop_dong_join = spatial_join_stops_to_dong(stops_release, boundary)
    stop_dong_join.drop(columns=["geometry"], errors="ignore").to_parquet(processed / "stop_dong_spatial_join_qa.parquet", index=False)
    outside_stops, dong_mismatch, mismatch_pairs, lineage_summary = build_lineage_outputs(
        stop_dong_join, national_boundary, boundary_config["external_region_prefixes"]
    )
    write_lineage_csvs(
        outside_stops, dong_mismatch, mismatch_pairs,
        config["paths"]["context_report"].parent,
    )
    context = build_transit_context_by_dong(
        boundary, stop_dong_join, profile, long[fact_columns],
        boundary_config["area_crs"], boundary_config["source"],
        str(config["project"]["version"]),
    )
    export_context(context, config["paths"]["release_dir"])
    population = pd.read_parquet(config["paths"]["sgis_population_processed"])
    demographic = build_demographic_context(context, population)
    demographic_report = build_demographic_quality_report(demographic, population)
    write_demographic_quality_report(demographic_report, config["paths"]["demographic_report"])
    if demographic_report["status"] == "FAIL":
        LOGGER.error("Demographic validation failed; release and DB load aborted")
        return 1
    export_demographic_context(demographic, config["paths"]["release_dir"])
    context_report = build_context_quality_report(
        national_boundary, boundary, stop_dong_join, context, profile,
        boundary_config["source"], boundary_config["source_url"],
        lineage_summary,
        boundary_config["authoritative_code_field"],
        boundary_config["authoritative_name_field"],
    )
    write_context_quality_report(context_report, config["paths"]["context_report"])
    manifest = build_release_manifest(profile, str(config["project"]["version"]), len(stops_release))
    manifest = add_context_manifest_entry(manifest, context)
    manifest = add_demographic_manifest_entry(manifest, demographic)
    write_release_manifest(manifest, config["paths"]["release_dir"] / "release_manifest.json")
    release_metrics = {"dim_transit_stop_rows":len(stops_release), "fact_stop_ridership_rows":len(long), "stop_profile_rows":len(profile)}
    report = build_quality_report(stop_metrics, rider_metrics, match_qa, config, {"stop_master":stop_encoding, "ridership":rider_encoding}, reshape_metrics, release_metrics)
    write_quality_report(report, config["paths"]["report"])
    LOGGER.info("Quality status: %s", report["status"])
    if not args.skip_db:
        load_postgis(stops_release, long[fact_columns], profile, get_engine(), config["database"]["schema"], context, str(config["project"]["version"]), demographic)
        expected_db = expected_from_release(config["paths"]["release_dir"], str(config["project"]["version"]))
        database_report = validate_database(get_engine(), config["database"]["schema"], expected_db)
        write_database_report(database_report, config["paths"]["database_report"])
        if database_report["status"] != "PASS":
            LOGGER.error("Database validation failed: %s", database_report["failures"])
            return 1
    return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
