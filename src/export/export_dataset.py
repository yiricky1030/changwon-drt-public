"""Parquet, GeoParquet and GeoJSON exporters."""
from pathlib import Path
import geopandas as gpd
import pandas as pd


def export_releases(stops: gpd.GeoDataFrame, facts: pd.DataFrame, profile: gpd.GeoDataFrame, release_dir: Path) -> None:
    """Write reusable release datasets without touching raw sources."""
    release_dir.mkdir(parents=True, exist_ok=True)
    stops.to_parquet(release_dir / "dim_transit_stop.parquet", index=False)
    stops.to_file(release_dir / "dim_transit_stop.geojson", driver="GeoJSON")
    facts.to_parquet(release_dir / "fact_stop_ridership.parquet", index=False)
    profile.to_parquet(release_dir / "changwon_transit_stop_profile.parquet", index=False)
    profile.to_file(release_dir / "changwon_transit_stop_profile.geojson", driver="GeoJSON")


def export_context(context: gpd.GeoDataFrame, release_dir: Path) -> None:
    """Write the administrative-dong context as GeoParquet and GeoJSON."""
    release_dir.mkdir(parents=True, exist_ok=True)
    context.to_parquet(release_dir / "changwon_transit_context_by_dong.parquet", index=False)
    context.to_file(release_dir / "changwon_transit_context_by_dong.geojson", driver="GeoJSON")


def export_demographic_context(context: gpd.GeoDataFrame, release_dir: Path) -> None:
    """Write the demographic context with GeoParquet spatial metadata."""
    context.to_parquet(release_dir / "changwon_transit_demographic_context_by_dong.parquet", index=False)
    context.to_file(release_dir / "changwon_transit_demographic_context_by_dong.geojson", driver="GeoJSON")
