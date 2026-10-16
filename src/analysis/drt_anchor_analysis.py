"""Diagnostic analysis for stop-level DRT anchor feature design.

This module deliberately produces no combined score, rank, recommendation, or
service zone.  It validates feature behaviour before weights are considered.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.strtree import STRtree


QA_FEATURES = {
    "observation_coverage_ratio", "quality_status", "anchor_data_status",
    "near_city_boundary_500m", "has_missing_observations", "is_duplicate_excluded",
}

DEMAND_FEATURES = [
    "ridership_total", "ridership_daily_avg", "ridership_pct_within_dong",
    "ridership_pct_citywide", "route_count", "route_count_pct_within_dong",
    "route_count_pct_citywide", "route_supply_scarcity_pct_within_dong",
    "route_supply_scarcity_pct_citywide", "nearby_stops_300m",
    "nearby_stops_500m", "nearby_stops_500m_pct_within_dong",
    "nearby_stops_500m_pct_citywide", "local_stop_scarcity_pct_within_dong",
    "local_stop_scarcity_pct_citywide", "nearby_route_count_500m",
    "nearby_route_count_500m_pct_within_dong", "nearby_route_count_500m_pct_citywide",
    "local_route_scarcity_pct_within_dong", "local_route_scarcity_pct_citywide",
    "nearest_stop_distance_m", "spatial_isolation_pct_within_dong",
    "spatial_isolation_pct_citywide", "top3_hour_share",
    "top3_hour_share_pct_within_dong", "top3_hour_share_pct_citywide",
]

SUPPLY_FEATURES = [column for column in DEMAND_FEATURES if not (
    column.startswith("ridership") or column.startswith("top3")
)]

SCENARIO_FEATURES = {
    "demand_supported_anchor": [
        "ridership_pct_within_dong", "route_supply_scarcity_pct_within_dong",
        "local_stop_scarcity_pct_within_dong", "spatial_isolation_pct_within_dong",
    ],
    "supply_gap_anchor": [
        "route_supply_scarcity_pct_within_dong", "local_stop_scarcity_pct_within_dong",
        "local_route_scarcity_pct_within_dong", "spatial_isolation_pct_within_dong",
    ],
}


def populations(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    demand = features.loc[features["analysis_ready"]].copy()
    supply = features.copy()
    assert len(demand) == 886 and len(supply) == 2926
    assert supply.loc[supply["anchor_data_status"].eq("SUPPLY_ONLY"), "ridership_total"].isna().all()
    return demand, supply


def correlation_pairs(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    pearson = frame[columns].corr(method="pearson")
    spearman = frame[columns].corr(method="spearman")
    rows = []
    for left, right in combinations(columns, 2):
        rho = spearman.loc[left, right]
        r = pearson.loc[left, right]
        level = "VERY_HIGH" if pd.notna(rho) and abs(rho) >= .90 else "HIGH" if pd.notna(rho) and abs(rho) >= .80 else None
        rows.append({"feature_1": left, "feature_2": right,
                     "pearson": float(r) if pd.notna(r) else np.nan,
                     "spearman": float(rho) if pd.notna(rho) else np.nan,
                     "redundancy_level": level})
    return pd.DataFrame(rows)


def pair_value(pairs: pd.DataFrame, left: str, right: str) -> dict[str, float]:
    row = pairs.loc[((pairs.feature_1 == left) & (pairs.feature_2 == right)) |
                    ((pairs.feature_1 == right) & (pairs.feature_2 == left))].iloc[0]
    return {"pearson": float(row.pearson), "spearman": float(row.spearman)}


def describe(values: pd.Series) -> dict[str, Any]:
    clean = values.dropna().astype(float)
    q1, median, q3 = clean.quantile([.25, .5, .75])
    return {"count": int(clean.size), "null": int(values.isna().sum()),
            "min": float(clean.min()), "q1": float(q1), "median": float(median),
            "q3": float(q3), "max": float(clean.max()), "mean": float(clean.mean()),
            "std": float(clean.std()), "iqr": float(q3 - q1)}


def distribution_analysis(features: pd.DataFrame) -> dict[str, Any]:
    demand, _ = populations(features)
    master_only = features.loc[features.anchor_data_status.eq("SUPPLY_ONLY")]
    key = list(dict.fromkeys(DEMAND_FEATURES + ["observation_coverage_ratio"]))
    groups = {"all": features, "observed_analysis_ready": demand, "supply_only": master_only}
    return {name: {column: describe(group[column]) for column in key if group[column].notna().any()}
            for name, group in groups.items()}


def temporal_analysis(demand: pd.DataFrame, hourly: pd.DataFrame) -> dict[str, Any]:
    hourly = hourly.copy()
    hourly["ridership_total"] = pd.to_numeric(hourly["ridership_total"], errors="coerce")
    by_stop = hourly.groupby("stop_id")["ridership_total"]
    alt = pd.DataFrame({
        "peak_hour_share": by_stop.max() / by_stop.sum(min_count=1),
        "active_hour_count": by_stop.apply(lambda s: int(s.fillna(0).gt(0).sum())),
    })
    alt["inactive_hour_concentration"] = 1 - alt.active_hour_count / 24
    work = demand.set_index("stop_id").join(alt).reset_index()
    work["demand_quartile"] = pd.qcut(work.ridership_total.rank(method="first"), 4,
                                      labels=["Q1_LOW", "Q2", "Q3", "Q4_HIGH"])
    quartiles = {}
    for label, group in work.groupby("demand_quartile", observed=True):
        q1, q3 = group.top3_hour_share.quantile([.25, .75])
        quartiles[str(label)] = {"count": int(len(group)), "ridership_min": float(group.ridership_total.min()),
            "ridership_max": float(group.ridership_total.max()), "top3_median": float(group.top3_hour_share.median()),
            "top3_iqr": float(q3-q1), "top3_max": float(group.top3_hour_share.max()),
            "top3_equal_1_count": int(group.top3_hour_share.eq(1).sum())}
    thresholds = {}
    for threshold in [10, 25, 50, 100, 200]:
        eligible = work.loc[work.ridership_total.ge(threshold)]
        eligible_rank = eligible.top3_hour_share.rank(pct=True).to_numpy(dtype=float)
        full_subset_rank = work.set_index("stop_id").loc[eligible.stop_id].top3_hour_share.rank(pct=True).to_numpy(dtype=float)
        thresholds[str(threshold)] = {"eligible_stop_count": int(len(eligible)),
            "top3_hour_share": describe(eligible.top3_hour_share),
            "extreme_1_count": int(eligible.top3_hour_share.eq(1).sum()),
            "rank_stability_vs_full": float(np.corrcoef(eligible_rank, full_subset_rank)[0, 1])}
    metrics = ["top3_hour_share", "peak_hour_share", "active_hour_count", "inactive_hour_concentration", "ridership_total"]
    return {"quartiles": quartiles, "thresholds": thresholds,
            "metric_spearman": work[metrics].corr(method="spearman").to_dict(),
            "recommended_threshold": 25,
            "recommendation_basis": "The lowest tested threshold that removes all exact top3=1 patterns while retaining 862 of 886 observed stops; use only as a provisional sensitivity threshold.",
            "temporal_role": "diagnostic_until_longer_observation_window"}


def boundary_analysis(features: gpd.GeoDataFrame, master: gpd.GeoDataFrame) -> dict[str, Any]:
    metric_all = master[["stop_id", "geometry"]].copy().to_crs(5179).reset_index(drop=True)
    metric_all["stop_id"] = metric_all.stop_id.astype(str)
    tree = STRtree(metric_all.geometry.to_numpy())
    all_ids = metric_all.stop_id.to_numpy()
    rows = []
    for stop_id, point in zip(features.stop_id.astype(str), features.to_crs(5179).geometry):
        indices = tree.query(point, predicate="dwithin", distance=500)
        rows.append((stop_id, int(sum(all_ids[indices] != stop_id))))
    extended = pd.DataFrame(rows, columns=["stop_id", "nearby_stops_500m_with_external_context"])
    work = features.merge(extended, on="stop_id", validate="one_to_one")
    work["delta_nearby_stops_500m"] = work.nearby_stops_500m_with_external_context - work.nearby_stops_500m
    work["extended_local_scarcity_pct_citywide"] = 100 - work.nearby_stops_500m_with_external_context.rank(method="average", pct=True) * 100
    work["delta_local_scarcity_rank_pct"] = work.extended_local_scarcity_pct_citywide - work.local_stop_scarcity_pct_citywide
    metrics = ["nearby_stops_500m", "nearby_route_count_500m", "nearest_stop_distance_m",
               "local_stop_scarcity_pct_citywide", "local_route_scarcity_pct_citywide"]
    boundary = work.near_city_boundary_500m
    return {"boundary_stop_count": int(boundary.sum()), "interior_stop_count": int((~boundary).sum()),
            "group_statistics": {"boundary": {m: describe(work.loc[boundary, m]) for m in metrics},
                                 "interior": {m: describe(work.loc[~boundary, m]) for m in metrics}},
            "external_stop_context": {"available": True, "official_master_count": int(len(master)),
                "outside_master_count": int(len(master)-len(features)),
                "boundary_delta_stop_count": describe(work.loc[boundary, "delta_nearby_stops_500m"]),
                "boundary_delta_scarcity_pct": describe(work.loc[boundary, "delta_local_scarcity_rank_pct"]),
                "boundary_stops_affected": int(work.loc[boundary, "delta_nearby_stops_500m"].gt(0).sum())},
            "external_route_context": {"available": False,
                "reason": "Completeness of matched route relations for the 600 outside-city master stops is not guaranteed; no route value was imputed."}}


def zero_route_analysis(features: pd.DataFrame) -> dict[str, Any]:
    zero = features.loc[features.route_count.eq(0)]
    return {"count": int(len(zero)), "observed": int(zero.has_observed_profile.sum()),
            "analysis_ready": int(zero.analysis_ready.sum()), "master_only": int(zero.anchor_data_status.eq("SUPPLY_ONLY").sum()),
            "near_boundary": int(zero.near_city_boundary_500m.sum()), "interior": int((~zero.near_city_boundary_500m).sum()),
            "nearby_routes_positive": int(zero.nearby_route_count_500m.gt(0).sum()),
            "nearby_routes_positive_ratio": float(zero.nearby_route_count_500m.gt(0).mean()),
            "observed_ridership_non_null": int(zero.ridership_total.notna().sum())}


def identical_coordinates(features: gpd.GeoDataFrame) -> dict[str, Any]:
    metric = features.to_crs(5179).copy()
    metric["coordinate_key"] = metric.geometry.to_wkb().map(bytes.hex)
    duplicated = metric.loc[metric.coordinate_key.duplicated(keep=False)]
    groups = []
    for key, group in duplicated.groupby("coordinate_key"):
        groups.append({"coordinate_key": key, "stop_count": int(len(group)), "stops": group[[
            "stop_id", "stop_name", "direction", "route_count", "quality_status", "anchor_data_status",
            "analysis_ready", "ridership_total", "nearest_stop_distance_m", "spatial_isolation_pct_citywide"
        ]].replace({np.nan: None}).to_dict("records")})
    return {"groups": groups, "group_count": len(groups), "stop_count": int(len(duplicated)),
            "self_exclusion_valid": bool(duplicated.nearest_stop_id.ne(duplicated.stop_id).all()),
            "interpretation": "Distinct official stop IDs at identical coordinates correctly have zero nearest distance; retain IDs and treat isolation as location co-presence, not entity duplication."}


def quadrant(frame: pd.DataFrame, x: str, y: str) -> dict[str, Any]:
    xmid, ymid = float(frame[x].median()), float(frame[y].median())
    labels = np.select([frame[x].ge(xmid)&frame[y].ge(ymid), frame[x].lt(xmid)&frame[y].ge(ymid),
                        frame[x].ge(xmid)&frame[y].lt(ymid)],
                       ["high_demand_high_scarcity", "high_demand_low_scarcity", "low_demand_high_scarcity"],
                       default="low_demand_low_scarcity")
    work = frame.assign(quadrant=labels)
    output = {"x_median": xmid, "y_median": ymid, "counts": work.quadrant.value_counts().to_dict(), "examples": {}}
    for label, group in work.groupby("quadrant"):
        output["examples"][label] = group.nlargest(5, y)[["stop_id", "stop_name", "adm_nm", x, y]].to_dict("records")
    return output


def scenario_validation(features: gpd.GeoDataFrame, demand_pairs: pd.DataFrame) -> dict[str, Any]:
    demand, _ = populations(features)
    direct = quadrant(demand, "route_supply_scarcity_pct_within_dong", "ridership_pct_within_dong")
    local = quadrant(demand, "local_stop_scarcity_pct_within_dong", "ridership_pct_within_dong")
    return {"guardrails": {"combined_score_generated": False, "final_rank_generated": False,
                           "qa_features_excluded": sorted(QA_FEATURES), "qa_overlap": sorted(set().union(*map(set, SCENARIO_FEATURES.values())) & QA_FEATURES)},
            "feature_directions": {"ridership_percentile": "higher demand", "route_scarcity": "lower direct route supply",
                "local_stop_scarcity": "lower surrounding stop supply", "local_route_scarcity": "lower surrounding route supply",
                "spatial_isolation": "larger nearest-stop distance", "temporal_concentration": "more concentrated temporal pattern"},
            "scenario_concepts": {"demand_supported_anchor": {"population": 886, "features": SCENARIO_FEATURES["demand_supported_anchor"],
                "excluded": ["quality/coverage fields", "top3_hour_share pending stability threshold", "citywide variants as primary inputs"]},
                "supply_gap_anchor": {"population": 2926, "features": SCENARIO_FEATURES["supply_gap_anchor"],
                "excluded": ["all ridership features", "all temporal demand features", "quality/coverage fields"]}},
            "quadrants": {"direct_route_scarcity": direct, "local_stop_scarcity": local,
                "classification_agreement_ratio": float(pd.Series(
                    np.select([demand.route_supply_scarcity_pct_within_dong.ge(demand.route_supply_scarcity_pct_within_dong.median()),], [1], default=0)
                    == np.select([demand.local_stop_scarcity_pct_within_dong.ge(demand.local_stop_scarcity_pct_within_dong.median()),], [1], default=0)).mean())},
            "key_pairs": {
                "300m_vs_500m": pair_value(demand_pairs, "nearby_stops_300m", "nearby_stops_500m"),
                "direct_vs_local_route": pair_value(demand_pairs, "route_count", "nearby_route_count_500m"),
                "stop_scarcity_vs_local_route_scarcity": pair_value(demand_pairs, "local_stop_scarcity_pct_within_dong", "local_route_scarcity_pct_within_dong"),
                "density_vs_isolation": pair_value(demand_pairs, "local_stop_scarcity_pct_within_dong", "spatial_isolation_pct_within_dong"),
                "ridership_vs_top3": pair_value(demand_pairs, "ridership_total", "top3_hour_share"),
                "ridership_within_vs_citywide": pair_value(demand_pairs, "ridership_pct_within_dong", "ridership_pct_citywide"),
                "route_scarcity_within_vs_citywide": pair_value(demand_pairs, "route_supply_scarcity_pct_within_dong", "route_supply_scarcity_pct_citywide"),
                "isolation_within_vs_citywide": pair_value(demand_pairs, "spatial_isolation_pct_within_dong", "spatial_isolation_pct_citywide"),
                "direct_vs_local_route_scarcity": pair_value(demand_pairs, "route_supply_scarcity_pct_within_dong", "local_route_scarcity_pct_within_dong")}}
