"""Transactional PostGIS loading."""
import geopandas as gpd
import pandas as pd
from sqlalchemy.engine import Engine
from sqlalchemy.dialects.postgresql import JSONB
from pathlib import Path


def load_postgis(stops: gpd.GeoDataFrame, facts: pd.DataFrame, profile: gpd.GeoDataFrame, engine: Engine, schema: str = "transport", context: gpd.GeoDataFrame | None = None, dataset_version: str = "", demographic_context: gpd.GeoDataFrame | None = None) -> None:
    """Create typed tables, clear prior derived rows, append, and index."""
    sql_dir = Path(__file__).resolve().parents[2] / "sql"
    stops_db = stops.copy()
    facts_db = facts.copy()
    profile_db = profile.copy()
    context_db = context.copy() if context is not None else None
    demographic_db = demographic_context.copy() if demographic_context is not None else None
    for frame in (stops_db, facts_db, profile_db, context_db, demographic_db):
        if frame is not None:
            frame["dataset_version"] = dataset_version
    profile_integer_columns = [
        "observation_days", "total_ridership", "weekday_ridership",
        "weekend_ridership", "morning_ridership", "daytime_ridership",
        "evening_ridership", "late_night_ridership", "peak_hour",
        "peak_hour_ridership", "missing_observation_count",
        "duplicate_observation_count",
    ]
    for column in profile_integer_columns:
        profile_db[column] = profile_db[column].astype("Int64")
    if context_db is not None:
        context_integer_columns = [
            "master_stop_count", "observed_stop_count", "analysis_ready_stop_count",
            "valid_stop_count", "missing_observation_stop_count",
            "duplicate_observation_stop_count", "total_ridership",
            "weekday_ridership", "weekend_ridership", "morning_ridership",
            "daytime_ridership", "evening_ridership", "late_night_ridership",
        ]
        for column in context_integer_columns:
            context_db[column] = context_db[column].astype("Int64")
    with engine.begin() as connection:
        for name in ("01_create_schema.sql", "02_create_tables.sql"):
            connection.exec_driver_sql((sql_dir / name).read_text(encoding="utf-8"))
        connection.exec_driver_sql(f"DELETE FROM {schema}.curated_stop_profile")
        if context is not None:
            connection.exec_driver_sql(f"DELETE FROM {schema}.transit_context_by_dong")
        if demographic_context is not None:
            connection.exec_driver_sql(f"DELETE FROM {schema}.transit_demographic_context_by_dong")
        connection.exec_driver_sql(f"DELETE FROM {schema}.fact_stop_ridership")
        connection.exec_driver_sql(f"DELETE FROM {schema}.dim_transit_stop")
        stops_db.to_postgis("dim_transit_stop", connection, schema=schema, if_exists="append", index=False)
        facts_db.to_sql("fact_stop_ridership", connection, schema=schema, if_exists="append", index=False, method="multi", chunksize=5000)
        profile_db.to_postgis("curated_stop_profile", connection, schema=schema, if_exists="append", index=False)
        if context_db is not None:
            context_db.to_postgis("transit_context_by_dong", connection, schema=schema, if_exists="append", index=False, dtype={"peak_hours": JSONB})
        if demographic_db is not None:
            demographic_db.to_postgis("transit_demographic_context_by_dong", connection, schema=schema, if_exists="append", index=False, dtype={"peak_hours": JSONB})
        connection.exec_driver_sql((sql_dir / "03_create_indexes.sql").read_text(encoding="utf-8"))
