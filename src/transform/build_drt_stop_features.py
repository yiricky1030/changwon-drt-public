"""Build trustworthy stop-level features for later DRT anchor analysis."""
from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.strtree import STRtree

from src.transform.build_drt_features import PERIOD_HOURS, safe_ratio


ROUTE_TYPES = {
    "지선버스": "branch_route_count", "간선버스": "trunk_route_count",
    "좌석버스": "seat_route_count", "마을버스": "village_route_count",
}


def build_base(master: gpd.GeoDataFrame, spatial: pd.DataFrame, version: str) -> gpd.GeoDataFrame:
    """Keep every official stop assigned to exactly one official Changwon dong."""
    assignment = spatial.loc[spatial["dong_code"].notna(), ["stop_id", "dong_code", "dong_name"]].copy()
    assignment["stop_id"] = assignment["stop_id"].astype(str)
    base = master.copy()
    base["stop_id"] = base["stop_id"].astype(str)
    base = base.merge(assignment, on="stop_id", how="inner", validate="one_to_one")
    base = base.rename(columns={"dong_code": "adm_cd", "dong_name": "adm_nm"}).to_crs(4326)
    base["source_version"] = version
    return base[["stop_id", "service_id", "stop_name", "adm_cd", "adm_nm", "geometry", "master_dong_value", "direction", "source", "source_version"]]


def build_status(base: gpd.GeoDataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    profile = profiles[["stop_id", "quality_status", "is_analysis_ready"]].copy()
    profile["stop_id"] = profile["stop_id"].astype(str)
    output = base[["stop_id"]].merge(profile, on="stop_id", how="left", validate="one_to_one")
    output["has_observed_profile"] = output["quality_status"].notna()
    output["quality_status"] = output["quality_status"].fillna("MASTER_ONLY")
    output["analysis_ready"] = output["is_analysis_ready"].fillna(False).astype(bool)
    output["is_duplicate_excluded"] = output["quality_status"].eq("DUPLICATE_OBSERVATIONS")
    output["has_missing_observations"] = output["quality_status"].eq("MISSING_OBSERVATIONS")
    status_map = {
        "MASTER_ONLY": "SUPPLY_ONLY", "VALID": "DEMAND_READY",
        "MISSING_OBSERVATIONS": "DEMAND_WITH_MISSING",
        "DUPLICATE_OBSERVATIONS": "DEMAND_EXCLUDED_DUPLICATE",
    }
    output["anchor_data_status"] = output["quality_status"].map(status_map)
    return output.drop(columns="is_analysis_ready")


def build_route_features(base: gpd.GeoDataFrame, bridge: pd.DataFrame, routes: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    matched = bridge.loc[bridge["is_matched"] & bridge["stop_id"].notna()].copy()
    matched["stop_id"] = matched["stop_id"].astype(str)
    matched = matched[matched["stop_id"].isin(set(base["stop_id"]))]
    matched = matched.merge(routes[["route_id", "route_type"]], on="route_id", how="left", validate="many_to_one")
    route_sets = matched.groupby("stop_id")["route_id"].agg(lambda values: set(values)).to_dict()
    output = pd.DataFrame({"stop_id": base["stop_id"]})
    output["route_count"] = output["stop_id"].map(lambda stop: len(route_sets.get(stop, set())))
    relation = matched.groupby("stop_id").size()
    output["route_stop_relation_count"] = output["stop_id"].map(relation).fillna(0).astype(int)
    output["route_relation_match_count"] = output["route_stop_relation_count"]
    type_sets = matched.groupby(["stop_id", "route_type"])["route_id"].nunique().unstack(fill_value=0)
    indexed = output.set_index("stop_id")
    for route_type, column in ROUTE_TYPES.items():
        indexed[column] = type_sets[route_type] if route_type in type_sets else 0
    indexed[list(ROUTE_TYPES.values())] = indexed[list(ROUTE_TYPES.values())].fillna(0).astype(int)
    return indexed.reset_index(), route_sets


def _neighbor_pairs(points: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    pairs = STRtree(points).query(points, predicate="dwithin", distance=radius)
    keep = pairs[0] != pairs[1]
    return pairs[0][keep], pairs[1][keep]


def _nearest(points: np.ndarray, target_indices: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    target_indices = np.arange(len(points)) if target_indices is None else target_indices
    targets = points[target_indices]
    tree = STRtree(targets)
    nearest_local, distances = tree.query_nearest(points, exclusive=True, all_matches=False, return_distance=True)
    result_id = target_indices[nearest_local[1]].astype(int)
    result_distance = distances.astype(float)
    coordinate_groups: dict[bytes, list[int]] = defaultdict(list)
    for index in target_indices:
        coordinate_groups[shapely.to_wkb(points[index])].append(int(index))
    for source in range(len(points)):
        coincident = [candidate for candidate in coordinate_groups.get(shapely.to_wkb(points[source]), []) if candidate != source]
        if coincident:
            result_id[source] = coincident[0]
            result_distance[source] = 0.0
    return result_id, result_distance


def build_spatial_features(
    base: gpd.GeoDataFrame,
    observed_ids: set[str],
    ready_ids: set[str],
    route_sets: dict[str, set[str]],
    ridership: dict[str, float],
    city_geometry: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Use an STRtree in EPSG:5179; no degree distance or quadratic scan."""
    metric = base.to_crs(5179).reset_index(drop=True)
    points = metric.geometry.to_numpy()
    pairs300 = _neighbor_pairs(points, 300)
    pairs500 = _neighbor_pairs(points, 500)
    neighbors300: dict[int, list[int]] = defaultdict(list)
    neighbors500: dict[int, list[int]] = defaultdict(list)
    for source, target in zip(*pairs300): neighbors300[int(source)].append(int(target))
    for source, target in zip(*pairs500): neighbors500[int(source)].append(int(target))
    stop_ids = metric["stop_id"].tolist()
    observed_index = np.array([i for i, stop in enumerate(stop_ids) if stop in observed_ids], dtype=int)
    ready_index = np.array([i for i, stop in enumerate(stop_ids) if stop in ready_ids], dtype=int)
    nearest_index, nearest_distance = _nearest(points)
    nearest_observed_index, nearest_observed_distance = _nearest(points, observed_index)
    nearest_ready_index, nearest_ready_distance = _nearest(points, ready_index)
    boundary = city_geometry.to_crs(5179).geometry.union_all().boundary
    boundary_distance = shapely.distance(points, boundary)
    rows = []
    for index, stop_id in enumerate(stop_ids):
        n300, n500 = neighbors300[index], neighbors500[index]
        route_union = set().union(*(route_sets.get(stop_ids[j], set()) for j in n500)) if n500 else set()
        local_ridership_300 = [ridership[stop_ids[j]] for j in n300 if stop_ids[j] in ridership]
        local_ridership_500 = [ridership[stop_ids[j]] for j in n500 if stop_ids[j] in ridership]
        nearest = int(nearest_index[index])
        rows.append({
            "stop_id": stop_id,
            "nearby_stops_300m": len(n300), "nearby_stops_500m": len(n500),
            "nearby_observed_stops_300m": sum(stop_ids[j] in observed_ids for j in n300),
            "nearby_observed_stops_500m": sum(stop_ids[j] in observed_ids for j in n500),
            "nearby_analysis_ready_stops_300m": sum(stop_ids[j] in ready_ids for j in n300),
            "nearby_analysis_ready_stops_500m": sum(stop_ids[j] in ready_ids for j in n500),
            "nearest_stop_id": stop_ids[nearest], "nearest_stop_distance_m": float(nearest_distance[index]),
            "nearest_stop_same_dong": bool(metric.loc[nearest, "adm_cd"] == metric.loc[index, "adm_cd"]),
            "nearest_observed_stop_distance_m": float(nearest_observed_distance[index]),
            "nearest_analysis_ready_stop_distance_m": float(nearest_ready_distance[index]),
            "nearby_route_count_500m": len(route_union),
            "nearby_route_stop_pair_count_500m": sum(len(route_sets.get(stop_ids[j], set())) for j in n500),
            "nearby_ridership_total_300m_excluding_self": sum(local_ridership_300) if local_ridership_300 else 0,
            "nearby_ridership_total_500m_excluding_self": sum(local_ridership_500) if local_ridership_500 else 0,
            "distance_to_changwon_boundary_m": float(boundary_distance[index]),
            "near_city_boundary_300m": bool(boundary_distance[index] <= 300),
            "near_city_boundary_500m": bool(boundary_distance[index] <= 500),
        })
    return pd.DataFrame(rows)


def build_hourly_and_temporal(
    base: gpd.GeoDataFrame, facts: pd.DataFrame, profiles: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ready = profiles.loc[profiles["is_analysis_ready"], ["stop_id"]].copy()
    ready["stop_id"] = ready["stop_id"].astype(str)
    ready = ready[ready["stop_id"].isin(set(base["stop_id"]))]
    fact = facts.copy(); fact["stop_id"] = fact["stop_id"].astype(str); fact["service_date"] = pd.to_datetime(fact["service_date"])
    selected = fact.merge(ready, on="stop_id", how="inner", validate="many_to_one")
    identity = base[["stop_id", "adm_cd", "adm_nm"]]
    skeleton = pd.DataFrame(product(ready["stop_id"], range(24)), columns=["stop_id", "hour"]).merge(identity, on="stop_id", how="left")
    grouped = selected.groupby(["stop_id", "hour"]).agg(
        ridership_total=("ridership", lambda values: values.sum(min_count=1)),
        non_null_observation_count=("ridership", "count"),
        null_observation_count=("ridership", lambda values: int(values.isna().sum())),
        observed_date_count=("service_date", "nunique"),
    ).reset_index()
    hourly = skeleton.merge(grouped, on=["stop_id", "hour"], how="left")
    for column in ["non_null_observation_count", "null_observation_count", "observed_date_count"]:
        hourly[column] = hourly[column].fillna(0).astype(int)
    hourly["ridership_total"] = hourly["ridership_total"].astype("Int64")

    observation = selected.groupby("stop_id").agg(
        ridership_total=("ridership", lambda values: values.sum(min_count=1)),
        ridership_daily_avg=("ridership", lambda values: values.groupby(selected.loc[values.index, "service_date"]).sum(min_count=1).mean()),
        non_null_observation_count=("ridership", "count"),
        null_observation_count=("ridership", lambda values: int(values.isna().sum())),
        observed_date_count=("service_date", "nunique"),
    ).reset_index()
    observation["expected_observation_count"] = observation["observed_date_count"] * 24
    observation["observation_coverage_ratio"] = safe_ratio(observation["non_null_observation_count"], observation["expected_observation_count"])
    observation["boardings_total"] = pd.Series(pd.NA, index=observation.index, dtype="Float64")
    observation["alightings_total"] = pd.Series(pd.NA, index=observation.index, dtype="Float64")

    temporal_rows = []
    hourly_groups = {stop_id: group.set_index("hour")["ridership_total"] for stop_id, group in hourly.groupby("stop_id", sort=False)}
    selected_groups = {stop_id: group for stop_id, group in selected.groupby("stop_id", sort=False)}
    for stop_id in ready["stop_id"]:
        h = hourly_groups[stop_id]
        valid = h.dropna(); total = valid.sum(min_count=1) if not valid.empty else pd.NA
        row: dict[str, Any] = {"stop_id": stop_id}
        for period, hours in PERIOD_HOURS.items():
            value = h.reindex(hours).dropna().sum(min_count=1)
            row[f"{period}_share"] = float(value / total) if pd.notna(value) and pd.notna(total) and total > 0 else pd.NA
        if valid.empty:
            row.update({"peak_hours": [], "peak_periods": [], "top3_hour_share": pd.NA})
        else:
            peak = valid.max(); row["peak_hours"] = [int(x) for x in valid.index[valid.eq(peak)]]
            period_values = {period: h.reindex(hours).dropna().sum(min_count=1) for period, hours in PERIOD_HOURS.items()}
            period_peak = max(period_values.values())
            row["peak_periods"] = [period for period, value in period_values.items() if value == period_peak]
            row["top3_hour_share"] = float(valid.nlargest(3).sum() / total) if total > 0 else pd.NA
        daily = selected_groups[stop_id].groupby("service_date")["ridership"].sum(min_count=1)
        weekday = daily[daily.index.dayofweek < 5]; weekend = daily[daily.index.dayofweek >= 5]
        row["weekday_daily_avg_ridership"] = float(weekday.mean()) if not weekday.empty else pd.NA
        row["weekend_daily_avg_ridership"] = float(weekend.mean()) if not weekend.empty else pd.NA
        row["weekend_weekday_ratio"] = float(weekend.mean() / weekday.mean()) if not weekend.empty and not weekday.empty and weekday.mean() > 0 else pd.NA
        temporal_rows.append(row)
    return hourly, observation, pd.DataFrame(temporal_rows)


def add_relative_features(features: pd.DataFrame) -> pd.DataFrame:
    output = features.copy()
    metrics = {
        "ridership": "ridership_total", "route_count": "route_count",
        "nearby_stops_500m": "nearby_stops_500m", "nearest_stop_distance": "nearest_stop_distance_m",
        "nearby_route_count_500m": "nearby_route_count_500m", "top3_hour_share": "top3_hour_share",
    }
    for name, column in metrics.items():
        output[f"{name}_pct_citywide"] = output[column].rank(method="average", pct=True) * 100
        output[f"{name}_pct_within_dong"] = output.groupby("adm_cd")[column].rank(method="average", pct=True) * 100
    for scope in ("citywide", "within_dong"):
        output[f"route_supply_scarcity_pct_{scope}"] = 100 - output[f"route_count_pct_{scope}"]
        output[f"local_stop_scarcity_pct_{scope}"] = 100 - output[f"nearby_stops_500m_pct_{scope}"]
        output[f"local_route_scarcity_pct_{scope}"] = 100 - output[f"nearby_route_count_500m_pct_{scope}"]
        output[f"spatial_isolation_pct_{scope}"] = output[f"nearest_stop_distance_pct_{scope}"]
    return output


def build_stop_features(master, spatial, profiles, facts, bridge, routes, city_geometry, version):
    base = build_base(master, spatial, version)
    status = build_status(base, profiles)
    route, route_sets = build_route_features(base, bridge, routes)
    hourly, observation, temporal = build_hourly_and_temporal(base, facts, profiles)
    observed_ids = set(status.loc[status["has_observed_profile"], "stop_id"])
    ready_ids = set(status.loc[status["analysis_ready"], "stop_id"])
    ridership = observation.set_index("stop_id")["ridership_total"].dropna().astype(float).to_dict()
    spatial_features = build_spatial_features(base, observed_ids, ready_ids, route_sets, ridership, city_geometry)
    features = base.merge(status, on="stop_id", validate="one_to_one").merge(route, on="stop_id", validate="one_to_one")
    features = features.merge(spatial_features, on="stop_id", validate="one_to_one").merge(observation, on="stop_id", how="left", validate="one_to_one").merge(temporal, on="stop_id", how="left", validate="one_to_one")
    features = add_relative_features(features)
    return gpd.GeoDataFrame(features, geometry="geometry", crs=4326), hourly


def validate_stop_features(features: gpd.GeoDataFrame, hourly: pd.DataFrame) -> None:
    assert len(features) == 2926 and features["stop_id"].is_unique and features["stop_id"].notna().all()
    assert features["adm_cd"].notna().all() and features["adm_cd"].nunique() == 55
    assert features.geometry.notna().all() and features.geometry.is_valid.all() and features.crs.to_epsg() == 4326
    assert features["nearby_stops_300m"].le(features["nearby_stops_500m"]).all()
    assert features["nearest_stop_id"].ne(features["stop_id"]).all()
    assert features["nearest_stop_distance_m"].ge(0).all()
    assert features["route_count"].between(0, 170).all()
    master_only = features["quality_status"].eq("MASTER_ONLY")
    duplicate = features["quality_status"].eq("DUPLICATE_OBSERVATIONS")
    analytic = ["ridership_total", "ridership_daily_avg", "morning_share", "top3_hour_share", "ridership_pct_citywide"]
    assert features.loc[master_only | duplicate, analytic].isna().all().all()
    assert len(hourly) == 886 * 24 and not hourly.duplicated(["stop_id", "hour"]).any()
    assert set(hourly["hour"]) == set(range(24))
