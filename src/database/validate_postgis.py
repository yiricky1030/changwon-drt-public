"""Database-level release validation and report generation."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine


REQUIRED_INDEXES = {
    ("dim_transit_stop", "idx_dim_transit_stop_geometry", "gist"),
    ("curated_stop_profile", "idx_curated_stop_profile_geometry", "gist"),
    ("transit_context_by_dong", "idx_transit_context_dong_geometry", "gist"),
    ("dim_transit_stop", "idx_dim_transit_stop_name", "btree"),
    ("fact_stop_ridership", "idx_fact_stop_ridership_stop_id", "btree"),
    ("fact_stop_ridership", "idx_fact_stop_ridership_service_date", "btree"),
    ("fact_stop_ridership", "idx_fact_stop_ridership_stop_date", "btree"),
    ("curated_stop_profile", "idx_curated_stop_profile_quality_status", "btree"),
    ("curated_stop_profile", "idx_curated_stop_profile_analysis_ready", "btree"),
    ("transit_context_by_dong", "idx_transit_context_dong_code", "btree"),
    ("transit_context_by_dong", "idx_transit_context_dong_name", "btree"),
}


def expected_from_release(release_dir: Path, dataset_version: str) -> dict[str, Any]:
    """Calculate expected DB assertions from current release files, not constants."""
    dim = gpd.read_parquet(release_dir / "dim_transit_stop.parquet")
    fact = pd.read_parquet(release_dir / "fact_stop_ridership.parquet")
    profile = gpd.read_parquet(release_dir / "changwon_transit_stop_profile.parquet")
    context = gpd.read_parquet(release_dir / "changwon_transit_context_by_dong.parquet")
    duplicate_groups = fact.loc[fact.is_duplicate_observation].groupby(
        ["stop_id", "service_date", "hour"]
    ).ngroups
    return {
        "dataset_version": dataset_version,
        "rows": {"dim": len(dim), "fact": len(fact), "profile": len(profile), "context": len(context)},
        "context": {
            "master_stops": int(context.master_stop_count.sum()),
            "observed_stops": int(context.observed_stop_count.sum()),
            "analysis_ready_stops": int(context.analysis_ready_stop_count.sum()),
        },
        "fact": {
            "null_ridership": int(fact.ridership.isna().sum()),
            "negative_ridership": int((fact.ridership < 0).sum()),
            "duplicate_observation_rows": int(fact.is_duplicate_observation.sum()),
            "duplicate_groups": int(duplicate_groups),
        },
        "profile": {
            "quality_status": {str(k): int(v) for k, v in profile.quality_status.value_counts().items()},
            "analysis_ready": int(profile.is_analysis_ready.sum()),
        },
    }


def _rows(connection: Any, sql: str) -> list[dict[str, Any]]:
    return [dict(row._mapping) for row in connection.execute(text(sql))]


def validate_database(engine: Engine, schema: str, expected: dict[str, Any]) -> dict[str, Any]:
    """Run SQL/catalog assertions and return a serializable evidence report."""
    failures: list[str] = []
    with engine.connect() as connection:
        environment = {
            "postgresql": connection.execute(text("select version()" )).scalar(),
            "postgis": connection.execute(text("select PostGIS_Version()" )).scalar(),
            "database": connection.execute(text("select current_database()" )).scalar(),
            "schema": schema,
        }
        counts = dict(_rows(connection, f"""
            SELECT
              (SELECT COUNT(*) FROM {schema}.dim_transit_stop) AS dim,
              (SELECT COUNT(*) FROM {schema}.fact_stop_ridership) AS fact,
              (SELECT COUNT(*) FROM {schema}.curated_stop_profile) AS profile,
              (SELECT COUNT(*) FROM {schema}.transit_context_by_dong) AS context
        """)[0])
        context = dict(_rows(connection, f"""
            SELECT SUM(master_stop_count) AS master_stops,
                   SUM(observed_stop_count) AS observed_stops,
                   SUM(analysis_ready_stop_count) AS analysis_ready_stops
            FROM {schema}.transit_context_by_dong
        """)[0])
        fact = dict(_rows(connection, f"""
            SELECT COUNT(*) AS rows,
                   COUNT(*) FILTER (WHERE ridership IS NULL) AS null_ridership,
                   COUNT(*) FILTER (WHERE ridership < 0) AS negative_ridership,
                   COUNT(*) FILTER (WHERE is_duplicate_observation) AS duplicate_observation_rows
            FROM {schema}.fact_stop_ridership
        """)[0])
        fact["duplicate_groups"] = connection.execute(text(f"""
            SELECT COUNT(*) FROM (
              SELECT stop_id, service_date, hour
              FROM {schema}.fact_stop_ridership
              GROUP BY stop_id, service_date, hour HAVING COUNT(*) > 1
            ) groups
        """)).scalar()
        profile_counts = {
            row["quality_status"]: row["count"] for row in _rows(connection, f"""
                SELECT quality_status, COUNT(*) AS count
                FROM {schema}.curated_stop_profile GROUP BY quality_status
            """)
        }
        profile = {
            "quality_status": profile_counts,
            "analysis_ready": connection.execute(text(
                f"SELECT COUNT(*) FROM {schema}.curated_stop_profile WHERE is_analysis_ready"
            )).scalar(),
        }
        spatial_rows = _rows(connection, f"""
            SELECT 'dim' AS dataset, ARRAY_AGG(DISTINCT ST_SRID(geometry)) AS srids,
                   COUNT(*) FILTER (WHERE geometry IS NULL) AS null_geometry,
                   COUNT(*) FILTER (WHERE geometry IS NOT NULL AND NOT ST_IsValid(geometry)) AS invalid_geometry
            FROM {schema}.dim_transit_stop
            UNION ALL
            SELECT 'profile', ARRAY_AGG(DISTINCT ST_SRID(geometry)),
                   COUNT(*) FILTER (WHERE geometry IS NULL),
                   COUNT(*) FILTER (WHERE geometry IS NOT NULL AND NOT ST_IsValid(geometry))
            FROM {schema}.curated_stop_profile
            UNION ALL
            SELECT 'context', ARRAY_AGG(DISTINCT ST_SRID(geometry)),
                   COUNT(*) FILTER (WHERE geometry IS NULL),
                   COUNT(*) FILTER (WHERE geometry IS NOT NULL AND NOT ST_IsValid(geometry))
            FROM {schema}.transit_context_by_dong
        """)
        spatial = {row.pop("dataset"): row for row in spatial_rows}
        spatial["context_geometry_types"] = _rows(connection, f"""
            SELECT GeometryType(geometry) AS geometry_type, COUNT(*) AS count
            FROM {schema}.transit_context_by_dong GROUP BY GeometryType(geometry)
        """)
        referential = {
            "fact_missing_stop": connection.execute(text(f"""
                SELECT COUNT(*) FROM {schema}.fact_stop_ridership f
                LEFT JOIN {schema}.dim_transit_stop d USING (stop_id)
                WHERE f.stop_id IS NOT NULL AND d.stop_id IS NULL
            """)).scalar(),
            "profile_missing_stop": connection.execute(text(f"""
                SELECT COUNT(*) FROM {schema}.curated_stop_profile p
                LEFT JOIN {schema}.dim_transit_stop d USING (stop_id)
                WHERE d.stop_id IS NULL
            """)).scalar(),
        }
        versions = {
            row["dataset"]: row["versions"] for row in _rows(connection, f"""
                SELECT 'dim' AS dataset, ARRAY_AGG(DISTINCT dataset_version) AS versions FROM {schema}.dim_transit_stop
                UNION ALL SELECT 'fact', ARRAY_AGG(DISTINCT dataset_version) FROM {schema}.fact_stop_ridership
                UNION ALL SELECT 'profile', ARRAY_AGG(DISTINCT dataset_version) FROM {schema}.curated_stop_profile
                UNION ALL SELECT 'context', ARRAY_AGG(DISTINCT dataset_version) FROM {schema}.transit_context_by_dong
            """)
        }
        indexes = _rows(connection, f"""
            SELECT table_class.relname AS table_name, index_class.relname AS name,
                   access_method.amname AS type, index_data.indisvalid AS valid,
                   PG_GET_INDEXDEF(index_data.indexrelid) AS definition
            FROM pg_index index_data
            JOIN pg_class index_class ON index_class.oid = index_data.indexrelid
            JOIN pg_class table_class ON table_class.oid = index_data.indrelid
            JOIN pg_namespace namespace ON namespace.oid = table_class.relnamespace
            JOIN pg_am access_method ON access_method.oid = index_class.relam
            WHERE namespace.nspname = '{schema}' ORDER BY table_name, name
        """)

    if counts != expected["rows"]:
        failures.append(f"row counts differ: {counts} != {expected['rows']}")
    if context != expected["context"]:
        failures.append(f"context invariants differ: {context} != {expected['context']}")
    for key, value in expected["fact"].items():
        if fact[key] != value:
            failures.append(f"fact {key}: {fact[key]} != {value}")
    if profile != expected["profile"]:
        failures.append(f"profile quality differs: {profile} != {expected['profile']}")
    for dataset in ("dim", "profile", "context"):
        if spatial[dataset]["srids"] != [4326] or spatial[dataset]["null_geometry"] or spatial[dataset]["invalid_geometry"]:
            failures.append(f"spatial assertion failed for {dataset}: {spatial[dataset]}")
    if any(referential.values()):
        failures.append(f"referential integrity failed: {referential}")
    expected_version = expected["dataset_version"]
    if any(values != [expected_version] for values in versions.values()):
        failures.append(f"dataset versions differ: {versions}")
    available = {(row["table_name"], row["name"], row["type"]) for row in indexes if row["valid"]}
    missing_indexes = sorted(REQUIRED_INDEXES - available)
    invalid_indexes = [row for row in indexes if not row["valid"]]
    if missing_indexes:
        failures.append(f"required indexes missing: {missing_indexes}")
    if invalid_indexes:
        failures.append("invalid indexes exist")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment,
        "expected_from_release": expected,
        "rows": counts,
        "context": context,
        "fact": fact,
        "profile": profile,
        "spatial": spatial,
        "referential_integrity": referential,
        "dataset_versions": versions,
        "indexes": indexes,
        "missing_required_indexes": [list(value) for value in missing_indexes],
        "invalid_indexes": invalid_indexes,
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }


def write_database_report(report: dict[str, Any], path: Path) -> None:
    """Write the database validation report from actual query results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
