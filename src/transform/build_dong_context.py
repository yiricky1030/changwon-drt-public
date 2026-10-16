"""Spatial join and administrative-dong transit context aggregation."""
from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone

import geopandas as gpd
import numpy as np
import pandas as pd


def normalize_admin_name(value: object) -> str | None:
    """Normalize Unicode, whitespace and dot glyphs without changing place semantics."""
    if pd.isna(value) or not str(value).strip():
        return None
    text = unicodedata.normalize("NFKC", str(value)).strip()
    text = re.sub(r"\s+", "", text)
    return text.replace("ㆍ", "·").replace("・", "·")


def spatial_join_stops_to_dong(
    stops: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Join by within and retain unmatched/ambiguous stops without nearest assignment."""
    points = stops.to_crs(boundary.crs).copy()
    if "master_dong_value" not in points.columns:
        raise ValueError("Canonical stop schema requires master_dong_value")
    polygons = boundary[["dong_code", "dong_name", "geometry"]]
    raw = gpd.sjoin(points, polygons, how="left", predicate="within")
    candidate_counts = raw.groupby("stop_id")["dong_code"].count()
    ambiguous_ids = set(candidate_counts[candidate_counts > 1].index)
    joined = raw.drop_duplicates("stop_id", keep="first").copy()
    joined.loc[joined.stop_id.isin(ambiguous_ids), ["dong_code", "dong_name"]] = pd.NA
    joined["boundary_candidate_count"] = joined.stop_id.map(candidate_counts).fillna(0).astype("int64")

    unmatched = joined["dong_code"].isna() & ~joined.stop_id.isin(ambiguous_ids)
    boundary_hits = gpd.sjoin(
        points.loc[unmatched, ["stop_id", "geometry"]], polygons,
        how="left", predicate="intersects",
    )
    intersect_counts = boundary_hits.groupby("stop_id")["dong_code"].count()
    joined["boundary_intersects_candidate_count"] = joined.stop_id.map(intersect_counts).fillna(0).astype("int64")

    master_norm = joined["master_dong_value"].map(normalize_admin_name)
    spatial_norm = joined["dong_name"].map(normalize_admin_name)
    joined["name_comparison"] = "MISMATCH"
    joined.loc[master_norm.isna(), "name_comparison"] = "MISSING"
    joined.loc[master_norm.eq(spatial_norm) & master_norm.notna(), "name_comparison"] = "EXACT"
    raw_master = joined["master_dong_value"].astype("string").str.strip()
    raw_spatial = joined["dong_name"].astype("string").str.strip()
    normalized_only = master_norm.eq(spatial_norm) & raw_master.ne(raw_spatial) & master_norm.notna()
    joined.loc[normalized_only, "name_comparison"] = "NORMALIZED"

    joined["match_status"] = "MATCH"
    joined.loc[joined["name_comparison"].eq("MISMATCH"), "match_status"] = "MASTER_NAME_MISMATCH"
    joined.loc[joined["name_comparison"].eq("MISSING"), "match_status"] = "MISSING_MASTER_VALUE"
    joined.loc[joined["dong_code"].isna(), "match_status"] = "OUTSIDE_BOUNDARY"
    joined.loc[joined.stop_id.isin(ambiguous_ids), "match_status"] = "BOUNDARY_AMBIGUOUS"
    return joined


def _sum_nullable(series: pd.Series) -> object:
    """Sum observed values, returning NULL when no value is available."""
    return series.sum(min_count=1)


def safe_per_1000(numerator: pd.Series, population: pd.Series) -> pd.Series:
    """Calculate a per-1,000 context metric; zero/missing population stays NULL."""
    denominator = population.where(population.gt(0))
    return numerator / denominator * 1000


def build_transit_context_by_dong(
    boundary: gpd.GeoDataFrame,
    stop_join: gpd.GeoDataFrame,
    profile: gpd.GeoDataFrame,
    fact: pd.DataFrame,
    area_crs: str,
    source: str,
    dataset_version: str,
) -> gpd.GeoDataFrame:
    """Build reusable context metrics without scoring or imputing missing data."""
    assigned = stop_join.dropna(subset=["dong_code"])[["stop_id", "dong_code"]].drop_duplicates()
    master_counts = assigned.groupby("dong_code").stop_id.nunique()
    prof = profile.merge(assigned, on="stop_id", how="inner", validate="one_to_one")
    grouped = prof.groupby("dong_code", dropna=False)

    context = boundary[["dong_code", "dong_name", "reference_date", "geometry"]].copy()
    context["master_stop_count"] = context.dong_code.map(master_counts).fillna(0).astype("int64")
    context["observed_stop_count"] = context.dong_code.map(grouped.stop_id.nunique()).fillna(0).astype("int64")
    context["analysis_ready_stop_count"] = context.dong_code.map(grouped.is_analysis_ready.sum()).fillna(0).astype("int64")
    quality_columns = {
        "VALID": "valid_stop_count",
        "MISSING_OBSERVATIONS": "missing_observation_stop_count",
        "DUPLICATE_OBSERVATIONS": "duplicate_observation_stop_count",
    }
    for status, column in quality_columns.items():
        counts = prof.loc[prof.quality_status.eq(status)].groupby("dong_code").stop_id.nunique()
        context[column] = context.dong_code.map(counts).fillna(0).astype("int64")

    metric_columns = [
        "total_ridership", "weekday_ridership", "weekend_ridership",
        "morning_ridership", "daytime_ridership", "evening_ridership",
        "late_night_ridership",
    ]
    for column in metric_columns:
        values = grouped[column].agg(_sum_nullable)
        context[column] = context.dong_code.map(values)
    average = prof.groupby("dong_code")["total_ridership"].mean()
    context["avg_ridership_per_observed_stop"] = context.dong_code.map(average)

    ready_ids = set(prof.loc[prof.is_analysis_ready, "stop_id"])
    hourly = fact.loc[fact.stop_id.isin(ready_ids)].merge(assigned, on="stop_id", how="inner")
    hourly = hourly.groupby(["dong_code", "hour"], as_index=False).ridership.agg(_sum_nullable)
    peak_map: dict[str, list[int] | None] = {}
    for code, rows in hourly.groupby("dong_code"):
        available = rows.dropna(subset=["ridership"])
        if available.empty:
            peak_map[code] = None
        else:
            maximum = available.ridership.max()
            peak_map[code] = sorted(available.loc[available.ridership.eq(maximum), "hour"].astype(int).tolist())
    context["peak_hours"] = context.dong_code.map(peak_map)

    context["observed_stop_ratio"] = np.where(
        context.master_stop_count.gt(0),
        context.observed_stop_count / context.master_stop_count,
        np.nan,
    )
    context["analysis_ready_ratio"] = np.where(
        context.observed_stop_count.gt(0),
        context.analysis_ready_stop_count / context.observed_stop_count,
        np.nan,
    )
    context["area_sq_km"] = context.to_crs(area_crs).geometry.area / 1_000_000
    context["stops_per_sq_km"] = np.where(
        context.area_sq_km.gt(0),
        context.master_stop_count / context.area_sq_km,
        np.nan,
    )
    context["source"] = source
    context["generated_at"] = datetime.now(timezone.utc)
    context["dataset_version"] = dataset_version
    return gpd.GeoDataFrame(context, geometry="geometry", crs="EPSG:4326")
