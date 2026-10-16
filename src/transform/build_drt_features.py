"""Build explainable administrative-dong features for later DRT scenario analysis."""
from __future__ import annotations

from itertools import product
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd


PERIOD_HOURS = {
    "morning": list(range(6, 9)),
    "daytime": list(range(9, 16)),
    "evening": list(range(16, 20)),
    "night": list(range(20, 24)),
    "late_night": list(range(0, 6)),
}
PERCENTILE_COLUMNS = [
    "elderly_ratio",
    "stops_per_1000_population",
    "routes_per_1000_population",
    "ridership_per_1000_population",
    "ridership_per_observed_stop",
]
ROUTE_TYPE_COLUMNS = {
    "지선버스": "route_branch_count",
    "간선버스": "route_trunk_count",
    "좌석버스": "route_seat_count",
    "마을버스": "route_village_count",
}


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide only where the denominator is known and positive."""
    result = pd.Series(pd.NA, index=numerator.index, dtype="Float64")
    mask = denominator.notna() & denominator.gt(0) & numerator.notna()
    result.loc[mask] = numerator.loc[mask].astype(float) / denominator.loc[mask].astype(float)
    return result


def load_base_context(context: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Select and rename the authoritative 55-dong base context."""
    columns = {
        "dong_code": "adm_cd",
        "dong_name": "adm_nm",
        "area_sq_km": "area_km2",
        "total_ridership": "ridership_total",
    }
    keep = [
        "dong_code", "dong_name", "geometry", "area_sq_km", "population_total",
        "population_65_plus", "elderly_ratio", "master_stop_count", "observed_stop_count",
        "analysis_ready_stop_count", "stops_per_1000_population",
        "observed_stops_per_1000_population", "analysis_ready_stops_per_1000_population",
        "total_ridership", "ridership_per_1000_population", "observed_stop_ratio",
        "analysis_ready_ratio", "dataset_version",
    ]
    base = context[keep].rename(columns=columns).copy().to_crs(4326)
    base["adm_cd"] = base["adm_cd"].astype(str)
    return base


def build_route_supply_features(
    base: gpd.GeoDataFrame,
    spatial: pd.DataFrame,
    bridge: pd.DataFrame,
    routes: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate matched TAGO bridge relations through canonical stop-to-dong assignments."""
    inside = spatial.loc[spatial["dong_code"].notna(), ["stop_id", "dong_code"]].rename(columns={"dong_code": "adm_cd"})
    inside["stop_id"] = inside["stop_id"].astype(str)
    matched = bridge.loc[bridge["is_matched"] & bridge["stop_id"].notna()].copy()
    matched["stop_id"] = matched["stop_id"].astype(str)
    matched = matched.merge(inside, on="stop_id", how="inner").merge(
        routes[["route_id", "route_type"]], on="route_id", how="left", validate="many_to_one"
    )

    route_count = matched.groupby("adm_cd")["route_id"].nunique().rename("route_count")
    relation_count = matched.groupby("adm_cd").size().rename("route_stop_relation_count")
    unique_pairs = matched.drop_duplicates(["adm_cd", "route_id", "stop_id"]).groupby("adm_cd").size().rename("unique_route_stop_pair_count")

    stop_route = matched.groupby(["adm_cd", "stop_id"])["route_id"].nunique().rename("distinct_route_count").reset_index()
    all_stops = inside.merge(stop_route, on=["adm_cd", "stop_id"], how="left")
    all_stops["distinct_route_count"] = all_stops["distinct_route_count"].fillna(0).astype(int)
    per_stop = all_stops.groupby("adm_cd")["distinct_route_count"].agg(
        avg_routes_per_stop="mean", median_routes_per_stop="median", max_routes_per_stop="max"
    )
    zero = all_stops.assign(is_zero=all_stops["distinct_route_count"].eq(0)).groupby("adm_cd")["is_zero"].sum().rename("zero_route_stop_count")

    output = pd.DataFrame({"adm_cd": base["adm_cd"]}).set_index("adm_cd")
    for series in (route_count, relation_count, unique_pairs, zero):
        output = output.join(series)
    output = output.join(per_stop)
    type_counts = matched.drop_duplicates(["adm_cd", "route_id", "route_type"]).groupby(["adm_cd", "route_type"]).size().unstack(fill_value=0)
    for route_type, column in ROUTE_TYPE_COLUMNS.items():
        output[column] = type_counts[route_type] if route_type in type_counts else 0

    count_columns = [
        "route_count", "route_stop_relation_count", "unique_route_stop_pair_count",
        "zero_route_stop_count", "max_routes_per_stop", *ROUTE_TYPE_COLUMNS.values(),
    ]
    output[count_columns] = output[count_columns].fillna(0).astype(int)
    output[["avg_routes_per_stop", "median_routes_per_stop"]] = output[["avg_routes_per_stop", "median_routes_per_stop"]].fillna(0.0)
    output = output.reset_index()

    denominators = base.set_index("adm_cd")
    output = output.set_index("adm_cd")
    output["routes_per_1000_population"] = safe_ratio(output["route_count"], denominators["population_total"]) * 1000
    output["routes_per_sq_km"] = safe_ratio(output["route_count"], denominators["area_km2"])
    output["route_stop_relations_per_sq_km"] = safe_ratio(output["route_stop_relation_count"], denominators["area_km2"])
    output["zero_route_stop_ratio"] = safe_ratio(output["zero_route_stop_count"], denominators["master_stop_count"])
    return output.reset_index()


def build_hourly_profile(
    base: gpd.GeoDataFrame,
    facts: pd.DataFrame,
    profiles: pd.DataFrame,
    spatial: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the complete 55 x 24 skeleton using analysis-ready stops only."""
    ready = profiles.loc[profiles["is_analysis_ready"], ["stop_id"]].copy()
    ready["stop_id"] = ready["stop_id"].astype(str)
    stop_dong = spatial.loc[spatial["dong_code"].notna(), ["stop_id", "dong_code"]].rename(columns={"dong_code": "adm_cd"})
    stop_dong["stop_id"] = stop_dong["stop_id"].astype(str)
    ready = ready.merge(stop_dong, on="stop_id", how="inner", validate="one_to_one")
    facts = facts.copy()
    facts["stop_id"] = facts["stop_id"].astype(str)
    selected = facts.merge(ready, on="stop_id", how="inner", validate="many_to_one").copy()
    selected["service_date"] = pd.to_datetime(selected["service_date"])

    skeleton = pd.DataFrame(product(base["adm_cd"], range(24)), columns=["adm_cd", "hour"])
    names = base[["adm_cd", "adm_nm", "analysis_ready_stop_count"]]
    grouped = selected.groupby(["adm_cd", "hour"], observed=True).agg(
        ridership_total=("ridership", lambda s: s.sum(min_count=1)),
        non_null_observation_count=("ridership", "count"),
        null_observation_count=("ridership", lambda s: int(s.isna().sum())),
        observed_date_count=("service_date", "nunique"),
    ).reset_index()
    hourly = skeleton.merge(names, on="adm_cd", how="left").merge(grouped, on=["adm_cd", "hour"], how="left")
    hourly["non_null_observation_count"] = hourly["non_null_observation_count"].fillna(0).astype(int)
    hourly["null_observation_count"] = hourly["null_observation_count"].fillna(0).astype(int)
    hourly["observed_date_count"] = hourly["observed_date_count"].fillna(0).astype(int)
    hourly["expected_observation_count"] = hourly["analysis_ready_stop_count"] * 8
    hourly["observation_coverage_ratio"] = safe_ratio(hourly["non_null_observation_count"], hourly["expected_observation_count"])
    hourly["ridership_total"] = hourly["ridership_total"].astype("Int64")
    return hourly, selected


def build_temporal_features(base: gpd.GeoDataFrame, hourly: pd.DataFrame, selected_facts: pd.DataFrame) -> pd.DataFrame:
    """Summarize periods, peaks, concentration, and calendar weekday/weekend patterns."""
    output: list[dict[str, Any]] = []
    for row in base[["adm_cd", "adm_nm"]].itertuples(index=False):
        h = hourly.loc[hourly["adm_cd"].eq(row.adm_cd)].set_index("hour")["ridership_total"]
        valid = h.dropna()
        record: dict[str, Any] = {"adm_cd": row.adm_cd}
        total = valid.sum(min_count=1) if not valid.empty else pd.NA
        for period, hours in PERIOD_HOURS.items():
            values = h.reindex(hours).dropna()
            value = values.sum(min_count=1) if not values.empty else pd.NA
            record[f"{period}_ridership"] = value
            record[f"{period}_share"] = float(value / total) if pd.notna(value) and pd.notna(total) and total > 0 else pd.NA
        if valid.empty:
            record.update({"peak_hours": [], "peak_hour": pd.NA, "peak_hour_ridership": pd.NA, "peak_hour_share": pd.NA, "peak_periods": [], "peak_period": pd.NA, "top3_hour_share": pd.NA, "active_hour_count": 0})
        else:
            peak_value = valid.max()
            peak_hours = [int(hour) for hour in valid.index[valid.eq(peak_value)]]
            period_values = {period: record[f"{period}_ridership"] for period in PERIOD_HOURS}
            usable_periods = {key: value for key, value in period_values.items() if pd.notna(value)}
            peak_period_value = max(usable_periods.values()) if usable_periods else pd.NA
            record.update({
                "peak_hours": peak_hours,
                "peak_hour": peak_hours[0] if len(peak_hours) == 1 else pd.NA,
                "peak_hour_ridership": int(peak_value),
                "peak_hour_share": float(peak_value / total) if total > 0 else pd.NA,
                "peak_periods": [key for key, value in usable_periods.items() if value == peak_period_value],
                "top3_hour_share": float(valid.nlargest(3).sum() / total) if total > 0 else pd.NA,
                "active_hour_count": int(valid.gt(0).sum()),
            })
            record["peak_period"] = record["peak_periods"][0] if len(record["peak_periods"]) == 1 else pd.NA
        output.append(record)
    temporal = pd.DataFrame(output)

    daily = selected_facts.copy()
    daily["is_weekend"] = daily["service_date"].dt.dayofweek.ge(5)
    daily = daily.groupby(["adm_cd", "service_date", "is_weekend"], observed=True)["ridership"].sum(min_count=1).reset_index()
    day_features = []
    for adm_cd in base["adm_cd"]:
        subset = daily.loc[daily["adm_cd"].eq(adm_cd)]
        rec: dict[str, Any] = {"adm_cd": adm_cd}
        for weekend, prefix in ((False, "weekday"), (True, "weekend")):
            values = subset.loc[subset["is_weekend"].eq(weekend), "ridership"].dropna()
            rec[f"{prefix}_day_count"] = int(values.size)
            rec[f"{prefix}_ridership_total"] = values.sum(min_count=1) if not values.empty else pd.NA
            rec[f"{prefix}_daily_avg_ridership"] = float(values.mean()) if not values.empty else pd.NA
        rec["weekend_weekday_ratio"] = (
            rec["weekend_daily_avg_ridership"] / rec["weekday_daily_avg_ridership"]
            if pd.notna(rec["weekend_daily_avg_ridership"]) and pd.notna(rec["weekday_daily_avg_ridership"]) and rec["weekday_daily_avg_ridership"] > 0 else pd.NA
        )
        day_features.append(rec)
    return temporal.merge(pd.DataFrame(day_features), on="adm_cd", how="left")


def build_quality_features(
    base: gpd.GeoDataFrame,
    facts: pd.DataFrame,
    profiles: pd.DataFrame,
    spatial: pd.DataFrame,
) -> pd.DataFrame:
    """Keep data confidence measures separate from any future need score."""
    stop_dong = spatial.loc[spatial["dong_code"].notna(), ["stop_id", "dong_code"]].rename(columns={"dong_code": "adm_cd"})
    stop_dong["stop_id"] = stop_dong["stop_id"].astype(str)
    p = profiles.copy(); p["stop_id"] = p["stop_id"].astype(str)
    p = p.merge(stop_dong, on="stop_id", how="inner")
    counts = p.groupby("adm_cd").agg(
        profile_stop_count=("stop_id", "nunique"),
        valid_stop_count=("quality_status", lambda s: int(s.eq("VALID").sum())),
        missing_observation_stop_count=("quality_status", lambda s: int(s.eq("MISSING_OBSERVATIONS").sum())),
        duplicate_observation_stop_count=("quality_status", lambda s: int(s.eq("DUPLICATE_OBSERVATIONS").sum())),
    ).reset_index()
    fact_geo = facts.copy()
    fact_geo["stop_id"] = fact_geo["stop_id"].astype(str)
    fact_geo = fact_geo.merge(stop_dong, on="stop_id", how="inner")
    duplicate_counts = fact_geo.groupby("adm_cd")["is_duplicate_observation"].agg(
        lambda s: int(s.fillna(False).sum())
    ).rename("duplicate_fact_count").reset_index()
    ready_ids = set(p.loc[p["is_analysis_ready"], "stop_id"])
    ready_facts = fact_geo.loc[fact_geo["stop_id"].isin(ready_ids)]
    fact_counts = ready_facts.groupby("adm_cd").agg(
        non_null_fact_count=("ridership", "count"),
        null_fact_count=("ridership", lambda s: int(s.isna().sum())),
    ).reset_index()
    output = base[["adm_cd", "analysis_ready_stop_count"]].merge(counts, on="adm_cd", how="left").merge(fact_counts, on="adm_cd", how="left").merge(duplicate_counts, on="adm_cd", how="left")
    count_cols = ["profile_stop_count", "valid_stop_count", "missing_observation_stop_count", "duplicate_observation_stop_count", "non_null_fact_count", "null_fact_count", "duplicate_fact_count"]
    output[count_cols] = output[count_cols].fillna(0).astype(int)
    output["has_missing_observations"] = output["missing_observation_stop_count"].gt(0)
    output["has_duplicate_excluded"] = output["duplicate_observation_stop_count"].gt(0)
    output["has_no_ridership_observation"] = output["analysis_ready_stop_count"].eq(0)
    expected = output["analysis_ready_stop_count"] * 8 * 24
    output["ridership_observation_coverage_ratio"] = safe_ratio(output["non_null_fact_count"], expected)
    return output.drop(columns="analysis_ready_stop_count")


def build_percentile_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add average-tie percentiles on a consistent 0-100 scale."""
    output = features.copy()
    for column in PERCENTILE_COLUMNS:
        output[f"{column}_pct"] = output[column].rank(method="average", pct=True) * 100
    output["stop_supply_scarcity_pct"] = 100 - output["stops_per_1000_population_pct"]
    output["route_supply_scarcity_pct"] = 100 - output["routes_per_1000_population_pct"]
    return output


def build_drt_features(
    context: gpd.GeoDataFrame,
    facts: pd.DataFrame,
    profiles: pd.DataFrame,
    spatial: pd.DataFrame,
    bridge: pd.DataFrame,
    routes: pd.DataFrame,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Compose the final feature table and hourly source table."""
    base = load_base_context(context)
    route_features = build_route_supply_features(base, spatial, bridge, routes)
    hourly, selected_facts = build_hourly_profile(base, facts, profiles, spatial)
    temporal = build_temporal_features(base, hourly, selected_facts)
    quality = build_quality_features(base, facts, profiles, spatial)
    features = base.merge(route_features, on="adm_cd", how="left", validate="one_to_one")
    features = features.merge(temporal, on="adm_cd", how="left", validate="one_to_one")
    features = features.merge(quality, on="adm_cd", how="left", validate="one_to_one")
    features["ridership_per_observed_stop"] = safe_ratio(features["ridership_total"], features["observed_stop_count"])
    features["ridership_per_analysis_ready_stop"] = safe_ratio(features["ridership_total"], features["analysis_ready_stop_count"])
    features["ridership_per_route"] = safe_ratio(features["ridership_total"], features["route_count"])
    features["ridership_per_unique_route_stop_pair"] = safe_ratio(features["ridership_total"], features["unique_route_stop_pair_count"])
    features = build_percentile_features(features)
    return gpd.GeoDataFrame(features, geometry="geometry", crs=4326), hourly


def validate_drt_features(
    features: gpd.GeoDataFrame,
    hourly: pd.DataFrame,
    global_routes_referenced: int,
    matching_summary: dict[str, int],
) -> dict[str, Any]:
    """Return a serializable QA report and fail on core invariant violations."""
    assert len(features) == 55 and features["adm_cd"].is_unique
    assert features.geometry.notna().all() and features.geometry.is_valid.all()
    assert features.crs.to_epsg() == 4326
    assert len(hourly) == 1320 and set(hourly["hour"]) == set(range(24))
    assert not hourly.duplicated(["adm_cd", "hour"]).any()
    assert features["observed_stop_count"].le(features["master_stop_count"]).all()
    assert features["analysis_ready_stop_count"].le(features["observed_stop_count"]).all()
    assert features["ridership_observation_coverage_ratio"].dropna().between(0, 1).all()
    for column in ("morning_share", "daytime_share", "evening_share", "night_share", "late_night_share", "peak_hour_share", "top3_hour_share"):
        assert features[column].dropna().between(0, 1).all()
    percentile_columns = [f"{column}_pct" for column in PERCENTILE_COLUMNS] + ["stop_supply_scarcity_pct", "route_supply_scarcity_pct"]
    peak_ties = int(features["peak_hours"].map(len).gt(1).sum())
    report = {
        "row_count": len(features),
        "unique_adm_cd": int(features["adm_cd"].nunique()),
        "unique_adm_nm": int(features["adm_nm"].nunique()),
        "null_adm_cd": int(features["adm_cd"].isna().sum()),
        "duplicate_adm_cd": int(features["adm_cd"].duplicated().sum()),
        "geometry": {"null": int(features.geometry.isna().sum()), "invalid": int((~features.geometry.is_valid).sum()), "srid": 4326},
        "population": {"null": int(features["population_total"].isna().sum()), "negative": int(features["population_total"].lt(0).sum()), "elderly_gt_total": int(features["population_65_plus"].gt(features["population_total"]).sum()), "total": int(features["population_total"].sum()), "elderly_total": int(features["population_65_plus"].sum())},
        "stops": {"sum_master": int(features["master_stop_count"].sum()), "sum_observed": int(features["observed_stop_count"].sum()), "sum_analysis_ready": int(features["analysis_ready_stop_count"].sum())},
        "routes": {"distinct_global_routes_referenced": int(global_routes_referenced), "route_count_min": int(features["route_count"].min()), "route_count_max": int(features["route_count"].max()), "route_count_median": float(features["route_count"].median()), "zero_route_stop_count_total": int(features["zero_route_stop_count"].sum()), "matching_summary": matching_summary},
        "ridership": {"global_total": int(features["ridership_total"].dropna().sum()), "null_observation_count": int(features["null_fact_count"].sum()), "duplicate_excluded_stop_count": int(features["duplicate_observation_stop_count"].sum()), "dong_with_no_ridership": features.loc[features["ridership_total"].isna(), "adm_nm"].tolist()},
        "hourly": {"expected_rows": 1320, "actual_rows": len(hourly), "hours_complete": set(hourly["hour"]) == set(range(24)), "peak_tie_count": peak_ties},
        "percentiles": {column: {"min": float(features[column].min()), "max": float(features[column].max()), "null": int(features[column].isna().sum())} for column in percentile_columns},
        "warnings": ["Route supply uses matched TAGO bridge relations only; unmatched and ambiguous relations are not force-matched.", "QA and coverage features describe confidence and must not be used automatically as DRT need."],
        "errors": [],
    }
    return report
