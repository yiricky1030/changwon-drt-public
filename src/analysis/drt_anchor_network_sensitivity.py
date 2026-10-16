"""Network-variant exploratory rank sensitivity for DRT anchor scenarios.

Reuses the Euclidean sensitivity machinery (weight sampling, rank matrices)
from drt_anchor_sensitivity, swapping straight-line local-supply and
isolation features for OSM walk-network equivalents. Local stop/route supply
counts stay populated even for stops the walk network cannot reach, but
isolation (nearest-walk-distance based) is NULL for the 2 unreachable stops;
NULL never becomes 0 and weights are never renormalized around it.
"""
from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from src.analysis.drt_anchor_sensitivity import AnchorScenario, _rank_matrices, _spearman

NETWORK_DEMAND = AnchorScenario(
    "network_demand_supported",
    ("ridership_pct_within_dong", "route_supply_scarcity_pct_within_dong",
     "local_stop_walk_scarcity_pct_within_dong", "walk_isolation_pct_within_dong"),
    ("ridership_pct_citywide", "route_supply_scarcity_pct_citywide",
     "local_stop_walk_scarcity_pct_citywide", "walk_isolation_pct_citywide"),
    ("DEMAND", "DIRECT_ROUTE_SCARCITY", "NETWORK_LOCAL_STOP_SCARCITY", "NETWORK_ISOLATION"),
    (.35, .20, .25, .20), ((.30, .40), (.15, .25), (.20, .30), (.15, .25)), "demand",
)
NETWORK_SUPPLY = AnchorScenario(
    "network_supply_gap",
    ("route_supply_scarcity_pct_within_dong", "local_stop_walk_scarcity_pct_within_dong",
     "local_route_walk_scarcity_pct_within_dong", "walk_isolation_pct_within_dong"),
    ("route_supply_scarcity_pct_citywide", "local_stop_walk_scarcity_pct_citywide",
     "local_route_walk_scarcity_pct_citywide", "walk_isolation_pct_citywide"),
    ("DIRECT_ROUTE_SCARCITY", "NETWORK_LOCAL_STOP_SCARCITY", "NETWORK_LOCAL_ROUTE_SCARCITY", "NETWORK_ISOLATION"),
    (.25, .25, .25, .25), ((.15, .35),) * 4, "supply",
)
NETWORK_SCENARIOS = (NETWORK_DEMAND, NETWORK_SUPPLY)

_EUCLIDEAN_TO_NETWORK_FEATURE = {
    "demand_supported": "network_demand_supported",
    "supply_gap": "network_supply_gap",
}

NETWORK_COLUMNS = [
    "stop_id", "snap_distance_m", "nearest_stop_walk_distance_m", "walk_detour_ratio",
    "walk_isolation_pct_within_dong", "walk_isolation_pct_citywide",
    "local_stop_walk_scarcity_pct_within_dong", "local_stop_walk_scarcity_pct_citywide",
    "local_route_walk_scarcity_pct_within_dong", "local_route_walk_scarcity_pct_citywide",
    "network_snap_flag", "network_reachable_flag",
]


def merge_network_features(features: gpd.GeoDataFrame, network: gpd.GeoDataFrame) -> pd.DataFrame:
    merged = features.merge(pd.DataFrame(network[NETWORK_COLUMNS]), on="stop_id", validate="one_to_one")
    assert len(merged) == len(features) == 2926
    return merged


def network_eligible_population(merged: pd.DataFrame, scenario: AnchorScenario) -> pd.DataFrame:
    output = merged.loc[merged.analysis_ready].copy() if scenario.population == "demand" else merged.copy()
    return output.reset_index(drop=True)


def missing_feature_report(frame: pd.DataFrame, scenario: AnchorScenario) -> dict[str, Any]:
    row_nan = frame[list(scenario.features)].isna().any(axis=1)
    return {"eligible": int(len(frame)), "eligible_scored": int((~row_nan).sum()),
            "excluded_missing_network_feature": int(row_nan.sum()),
            "excluded_stop_ids": frame.loc[row_nan, "stop_id"].tolist()}


def network_base_outputs(frame: pd.DataFrame, scenario: AnchorScenario) -> tuple[pd.DataFrame, pd.DataFrame]:
    values, weights = frame[list(scenario.features)].to_numpy(float), np.array(scenario.base_weights)
    row_nan = np.isnan(values).any(axis=1)
    contributions = values * weights
    score = contributions.sum(axis=1)
    base = frame[["stop_id", "stop_name", "adm_cd", "adm_nm", "anchor_data_status", "route_count",
                  "nearby_route_count_500m", "ridership_total", "ridership_daily_avg",
                  "snap_distance_m", "walk_detour_ratio", "network_reachable_flag"]].copy()
    base["scenario"] = scenario.key
    base["exploratory_anchor_score"] = score
    base["rank_within_dong"] = base.groupby("adm_cd")["exploratory_anchor_score"].rank(method="min", ascending=False)
    base["rank_citywide"] = base.exploratory_anchor_score.rank(method="min", ascending=False)
    contribution = base[["stop_id", "scenario", "exploratory_anchor_score"]].copy()
    for index, label in enumerate(scenario.labels):
        contribution[f"contribution_{label.lower()}"] = contributions[:, index]
    contribution_columns = [c for c in contribution if c.startswith("contribution_")]
    contribution["contribution_sum"] = contributions.sum(axis=1)
    dominant = np.array(scenario.labels)[np.nanargmax(np.where(np.isnan(contributions), -np.inf, contributions), axis=1)]
    maximum = contribution[contribution_columns].max(axis=1)
    contribution["dominant_feature"] = np.where(row_nan, None, dominant)
    contribution["max_contribution_share"] = np.where(row_nan, np.nan, maximum / contribution.exploratory_anchor_score.replace(0, np.nan))
    contribution["single_feature_dominated"] = np.where(row_nan, False, contribution.max_contribution_share.ge(.60))
    return base, contribution


def network_sensitivity(frame: pd.DataFrame, scenario: AnchorScenario, weights: pd.DataFrame) -> pd.DataFrame:
    values = frame[list(scenario.features)].to_numpy(float)
    row_nan = np.isnan(values).any(axis=1)
    scores = values @ weights.to_numpy(float).T
    within, percentile, city = _rank_matrices(scores, frame.adm_cd)
    result = frame[["stop_id", "stop_name", "adm_cd", "adm_nm", "anchor_data_status", "route_count",
                    "nearby_route_count_500m", "ridership_total", "ridership_daily_avg",
                    "snap_distance_m", "walk_detour_ratio", "network_reachable_flag"]].copy()
    result["scenario"] = scenario.key
    result["candidate_stop_count"] = result.adm_cd.map(result.adm_cd.value_counts()).astype(int)
    result["mean_rank_within_dong"] = within.mean(axis=1)
    result["median_rank_within_dong"] = np.median(within, axis=1)
    result["best_rank_within_dong"] = within.min(axis=1)
    result["worst_rank_within_dong"] = within.max(axis=1)
    result["rank_std_within_dong"] = within.std(axis=1)
    result["top1_frequency_within_dong"] = (within <= 1).mean(axis=1)
    result["top3_frequency_within_dong"] = (within <= 3).mean(axis=1)
    result["top5_frequency_within_dong"] = (within <= 5).mean(axis=1)
    result["top10pct_frequency_within_dong"] = (percentile <= .10).mean(axis=1)
    result["top20pct_frequency_within_dong"] = (percentile <= .20).mean(axis=1)
    result["mean_rank_citywide"] = city.mean(axis=1)
    result["median_rank_citywide"] = np.median(city, axis=1)
    result["top20_frequency_citywide"] = (city <= 20).mean(axis=1)
    result["top50_frequency_citywide"] = (city <= 50).mean(axis=1)
    result["top100_frequency_citywide"] = (city <= 100).mean(axis=1)
    stat_columns = ["mean_rank_within_dong", "median_rank_within_dong", "best_rank_within_dong",
        "worst_rank_within_dong", "rank_std_within_dong", "top1_frequency_within_dong",
        "top3_frequency_within_dong", "top5_frequency_within_dong", "top10pct_frequency_within_dong",
        "top20pct_frequency_within_dong", "mean_rank_citywide", "median_rank_citywide",
        "top20_frequency_citywide", "top50_frequency_citywide", "top100_frequency_citywide"]
    result.loc[row_nan, stat_columns] = np.nan
    standard = np.select([result.top3_frequency_within_dong.ge(.8), result.top3_frequency_within_dong.ge(.5),
                          result.top3_frequency_within_dong.ge(.2)],
                         ["VERY_STABLE", "STABLE", "WEIGHT_SENSITIVE"], default="LOW_FREQUENCY")
    result["stability_label"] = np.select([row_nan, result.candidate_stop_count.le(3)],
                                          ["NETWORK_UNREACHABLE_NO_SCORE", "SMALL_CANDIDATE_SET"], default=standard)
    result["low_sample_dong"] = result.candidate_stop_count.lt(5)
    result["route_zero_direct_match"] = result.route_count.eq(0)
    result["network_score_available"] = ~row_nan
    return result


def score_distribution(values: pd.Series) -> dict[str, float]:
    clean = values.dropna().astype(float)
    q1, median, q3 = clean.quantile([.25, .5, .75])
    return {"count": int(clean.size), "null": int(values.isna().sum()), "min": float(clean.min()), "q1": float(q1),
            "median": float(median), "q3": float(q3), "max": float(clean.max()), "mean": float(clean.mean())}


def stop_comparison(euclidean: pd.DataFrame, network: pd.DataFrame, scenario_key: str) -> pd.DataFrame:
    left = euclidean[["stop_id", "adm_cd", "adm_nm", "mean_rank_within_dong", "top3_frequency_within_dong",
                      "top10pct_frequency_within_dong"]].rename(columns={
        "mean_rank_within_dong": "mean_rank_euclidean", "top3_frequency_within_dong": "top3_frequency_euclidean",
        "top10pct_frequency_within_dong": "top10pct_frequency_euclidean"})
    right = network[["stop_id", "mean_rank_within_dong", "top3_frequency_within_dong",
                     "top10pct_frequency_within_dong", "snap_distance_m", "walk_detour_ratio",
                     "network_score_available"]].rename(columns={
        "mean_rank_within_dong": "mean_rank_network", "top3_frequency_within_dong": "top3_frequency_network",
        "top10pct_frequency_within_dong": "top10pct_frequency_network"})
    merged = left.merge(right, on="stop_id", validate="one_to_one")
    merged["scenario"] = scenario_key
    merged["mean_rank_delta"] = merged.mean_rank_network - merged.mean_rank_euclidean
    merged["top3_frequency_delta"] = merged.top3_frequency_network - merged.top3_frequency_euclidean
    merged["top10pct_frequency_delta"] = merged.top10pct_frequency_network - merged.top10pct_frequency_euclidean
    scored = merged.network_score_available
    merged["robust_both_top3"] = scored & merged.top3_frequency_euclidean.ge(.8) & merged.top3_frequency_network.ge(.8)
    merged["robust_both_top10pct"] = scored & merged.top10pct_frequency_euclidean.ge(.8) & merged.top10pct_frequency_network.ge(.8)
    merged["network_sensitive"] = scored & (merged.mean_rank_delta.abs().ge(10) | merged.top3_frequency_delta.abs().ge(.30))
    merged["snap_over_100m"] = merged.snap_distance_m.gt(100)
    merged["snap_over_200m"] = merged.snap_distance_m.gt(200)
    return merged


def scenario_level_comparison(comparison: pd.DataFrame) -> dict[str, Any]:
    scored = comparison.loc[comparison.network_score_available]
    return {"mean_rank_spearman": _spearman(scored.mean_rank_euclidean, scored.mean_rank_network),
            "top3_frequency_spearman": _spearman(scored.top3_frequency_euclidean, scored.top3_frequency_network),
            "top10pct_frequency_spearman": _spearman(scored.top10pct_frequency_euclidean, scored.top10pct_frequency_network),
            "mean_rank_delta": {"mean": float(scored.mean_rank_delta.abs().mean()),
                "p95": float(scored.mean_rank_delta.abs().quantile(.95)), "max": float(scored.mean_rank_delta.abs().max())},
            "excluded_missing_network_score": int((~comparison.network_score_available).sum())}


def dong_level_summary(comparison: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    euclid_top = comparison.top3_frequency_euclidean.ge(.8)
    network_top = comparison.network_score_available & comparison.top3_frequency_network.ge(.8)
    for adm_cd, group in comparison.groupby("adm_cd"):
        e_ids = set(group.loc[euclid_top.loc[group.index], "stop_id"])
        n_ids = set(group.loc[network_top.loc[group.index], "stop_id"])
        rows.append({"adm_cd": adm_cd, "adm_nm": group.adm_nm.iloc[0], "common": len(e_ids & n_ids),
                     "network_only": len(n_ids - e_ids), "euclidean_only": len(e_ids - n_ids)})
    return rows


def snap_and_detour_diagnostics(network_stability: pd.DataFrame) -> dict[str, Any]:
    top = network_stability.loc[network_stability.network_score_available & network_stability.top3_frequency_within_dong.ge(.8)]
    if top.empty:
        return {"top_region_count": 0}
    detour = top.walk_detour_ratio.dropna()
    return {"top_region_count": int(len(top)),
            "snap_over_100m_count": int(top.snap_distance_m.gt(100).sum()),
            "snap_over_100m_share": float(top.snap_distance_m.gt(100).mean()),
            "snap_over_200m_count": int(top.snap_distance_m.gt(200).sum()),
            "snap_over_200m_share": float(top.snap_distance_m.gt(200).mean()),
            "detour_ratio_median": float(detour.median()) if len(detour) else None,
            "detour_ratio_p95": float(detour.quantile(.95)) if len(detour) else None,
            "detour_ratio_max": float(detour.max()) if len(detour) else None}


def validate_scenario_definition(scenario: AnchorScenario) -> None:
    from src.analysis.drt_anchor_analysis import QA_FEATURES
    assert not (set(scenario.features) & QA_FEATURES)
    assert "top3_hour_share" not in " ".join(scenario.features)
    if scenario.key == "network_supply_gap":
        assert not any("ridership" in feature for feature in scenario.features)
