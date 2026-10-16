"""Deterministic exploratory rank sensitivity for DRT anchor scenarios."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.strtree import STRtree

from src.analysis.drt_anchor_analysis import QA_FEATURES


@dataclass(frozen=True)
class AnchorScenario:
    key: str
    features: tuple[str, ...]
    citywide_features: tuple[str, ...]
    labels: tuple[str, ...]
    base_weights: tuple[float, ...]
    ranges: tuple[tuple[float, float], ...]
    population: str


DEMAND = AnchorScenario(
    "demand_supported",
    ("ridership_pct_within_dong", "route_supply_scarcity_pct_within_dong",
     "local_stop_scarcity_pct_within_dong", "spatial_isolation_pct_within_dong"),
    ("ridership_pct_citywide", "route_supply_scarcity_pct_citywide",
     "local_stop_scarcity_pct_citywide", "spatial_isolation_pct_citywide"),
    ("DEMAND", "DIRECT_ROUTE_SCARCITY", "LOCAL_STOP_SCARCITY", "ISOLATION"),
    (.35, .20, .25, .20), ((.30, .40), (.15, .25), (.20, .30), (.15, .25)), "demand",
)
SUPPLY = AnchorScenario(
    "supply_gap",
    ("route_supply_scarcity_pct_within_dong", "local_stop_scarcity_pct_within_dong",
     "local_route_scarcity_pct_within_dong", "spatial_isolation_pct_within_dong"),
    ("route_supply_scarcity_pct_citywide", "local_stop_scarcity_pct_citywide",
     "local_route_scarcity_pct_citywide", "spatial_isolation_pct_citywide"),
    ("DIRECT_ROUTE_SCARCITY", "LOCAL_STOP_SCARCITY", "LOCAL_ROUTE_SCARCITY", "ISOLATION"),
    (.25, .25, .25, .25), ((.15, .35),) * 4, "supply",
)
SCENARIOS = (DEMAND, SUPPLY)


def generate_weights(scenario: AnchorScenario, count: int = 2000, seed: int = 20260830) -> pd.DataFrame:
    """Rejection sample valid weights; last dimension closes every row to one."""
    rng, accepted = np.random.default_rng(seed), []
    while len(accepted) < count:
        first = np.array([rng.uniform(*bounds) for bounds in scenario.ranges[:-1]])
        last = 1.0 - first.sum()
        if scenario.ranges[-1][0] <= last <= scenario.ranges[-1][1]:
            row = np.append(first, last)
            row[-1] = 1.0 - row[:-1].sum()
            accepted.append(row)
    return pd.DataFrame(accepted, columns=scenario.labels)


def eligible_population(features: pd.DataFrame, scenario: AnchorScenario) -> pd.DataFrame:
    output = features.loc[features.analysis_ready].copy() if scenario.population == "demand" else features.copy()
    assert output[list(scenario.features)].notna().all().all()
    return output.reset_index(drop=True)


def base_outputs(frame: pd.DataFrame, scenario: AnchorScenario) -> tuple[pd.DataFrame, pd.DataFrame]:
    values, weights = frame[list(scenario.features)].to_numpy(float), np.array(scenario.base_weights)
    contributions = values * weights
    score = contributions.sum(axis=1)
    base = frame[["stop_id", "stop_name", "adm_cd", "adm_nm", "anchor_data_status", "route_count",
                  "nearby_route_count_500m", "ridership_total", "ridership_daily_avg"]].copy()
    base["scenario"] = scenario.key
    base["exploratory_anchor_score"] = score
    base["rank_within_dong"] = base.groupby("adm_cd")["exploratory_anchor_score"].rank(method="min", ascending=False).astype(int)
    base["rank_citywide"] = base.exploratory_anchor_score.rank(method="min", ascending=False).astype(int)
    contribution = base[["stop_id", "scenario", "exploratory_anchor_score"]].copy()
    for index, label in enumerate(scenario.labels):
        contribution[f"contribution_{label.lower()}"] = contributions[:, index]
    contribution_columns = [c for c in contribution if c.startswith("contribution_")]
    maximum = contribution[contribution_columns].max(axis=1)
    contribution["contribution_sum"] = contribution[contribution_columns].sum(axis=1)
    contribution["dominant_feature"] = np.array(scenario.labels)[contributions.argmax(axis=1)]
    contribution["max_contribution_share"] = maximum / contribution.exploratory_anchor_score.replace(0, np.nan)
    contribution["single_feature_dominated"] = contribution.max_contribution_share.ge(.60)
    return base, contribution


def _rank_matrices(scores: np.ndarray, adm: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    city = pd.DataFrame(scores).rank(axis=0, method="min", ascending=False).to_numpy()
    within = np.empty_like(scores, dtype=float)
    percentile = np.empty_like(scores, dtype=float)
    for _, indices in adm.groupby(adm).groups.items():
        idx = np.asarray(list(indices), dtype=int)
        local = pd.DataFrame(scores[idx]).rank(axis=0, method="min", ascending=False).to_numpy()
        within[idx] = local
        percentile[idx] = local / len(idx)
    return within, percentile, city


def sensitivity(frame: pd.DataFrame, scenario: AnchorScenario, weights: pd.DataFrame) -> pd.DataFrame:
    scores = frame[list(scenario.features)].to_numpy(float) @ weights.to_numpy(float).T
    within, percentile, city = _rank_matrices(scores, frame.adm_cd)
    result = frame[["stop_id", "stop_name", "adm_cd", "adm_nm", "anchor_data_status", "route_count",
                    "nearby_route_count_500m", "ridership_total", "ridership_daily_avg"]].copy()
    result["scenario"] = scenario.key
    result["candidate_stop_count"] = result.adm_cd.map(result.adm_cd.value_counts()).astype(int)
    result["mean_rank_within_dong"] = within.mean(axis=1)
    result["median_rank_within_dong"] = np.median(within, axis=1)
    result["best_rank_within_dong"] = within.min(axis=1).astype(int)
    result["worst_rank_within_dong"] = within.max(axis=1).astype(int)
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
    standard = np.select([result.top3_frequency_within_dong.ge(.8), result.top3_frequency_within_dong.ge(.5),
                          result.top3_frequency_within_dong.ge(.2)],
                         ["VERY_STABLE", "STABLE", "WEIGHT_SENSITIVE"], default="LOW_FREQUENCY")
    result["stability_label"] = np.where(result.candidate_stop_count.le(3), "SMALL_CANDIDATE_SET", standard)
    result["low_sample_dong"] = result.candidate_stop_count.lt(5)
    result["route_zero_direct_match"] = result.route_count.eq(0)
    return result


def _base_rank(frame: pd.DataFrame, columns: tuple[str, ...], weights: tuple[float, ...]) -> pd.DataFrame:
    temp = frame[["stop_id", "adm_cd"]].copy()
    temp["score"] = frame[list(columns)].to_numpy(float) @ np.array(weights)
    temp["rank"] = temp.groupby("adm_cd").score.rank(method="min", ascending=False)
    temp["city_rank"] = temp.score.rank(method="min", ascending=False)
    return temp


def _spearman(left: pd.Series, right: pd.Series) -> float:
    return float(pd.DataFrame({"left": left, "right": right}).corr(method="spearman").iloc[0, 1])


def external_neighbor_counts(frame: gpd.GeoDataFrame, master: gpd.GeoDataFrame) -> pd.Series:
    all_metric = master[["stop_id", "geometry"]].copy().to_crs(5179).reset_index(drop=True)
    all_metric.stop_id = all_metric.stop_id.astype(str)
    ids, tree = all_metric.stop_id.to_numpy(), STRtree(all_metric.geometry.to_numpy())
    counts = []
    for stop_id, point in zip(frame.stop_id.astype(str), frame.to_crs(5179).geometry):
        found = tree.query(point, predicate="dwithin", distance=500)
        counts.append(int(sum(ids[found] != stop_id)))
    return pd.Series(counts, index=frame.index)


def variants(frame: gpd.GeoDataFrame, scenario: AnchorScenario, master: gpd.GeoDataFrame) -> dict[str, Any]:
    main = _base_rank(frame, scenario.features, scenario.base_weights)
    alternative = frame.copy()
    alternative["local_stop_scarcity_300m_within"] = 100 - alternative.groupby("adm_cd")["nearby_stops_300m"].rank(method="average", pct=True) * 100
    columns300 = tuple("local_stop_scarcity_300m_within" if c == "local_stop_scarcity_pct_within_dong" else c for c in scenario.features)
    rank300 = _base_rank(alternative, columns300, scenario.base_weights)
    city = _base_rank(frame, scenario.citywide_features, scenario.base_weights)

    extended = external_neighbor_counts(frame, master)
    alternative["external_count"] = extended
    alternative["external_scarcity"] = 100 - alternative.groupby("adm_cd").external_count.rank(method="average", pct=True) * 100
    external_columns = tuple("external_scarcity" if c == "local_stop_scarcity_pct_within_dong" else c for c in scenario.features)
    external_rank = _base_rank(alternative, external_columns, scenario.base_weights)
    affected = alternative.near_city_boundary_500m & extended.ne(alternative.nearby_stops_500m)

    route_variant = None
    if scenario.key == "supply_gap":
        route_variant = main.copy()
        mask = frame.route_count.eq(0) & frame.nearby_route_count_500m.gt(0)
        values = frame[list(scenario.features)].to_numpy(float)
        alt_score = main.score.to_numpy(copy=True)
        alt_score[mask] = values[mask, 1:].mean(axis=1)
        route_variant["score"] = alt_score
        route_variant["rank"] = route_variant.groupby("adm_cd").score.rank(method="min", ascending=False)

    def comparison(other: pd.DataFrame) -> dict[str, Any]:
        delta = other["rank"] - main["rank"]
        return {"rank_spearman": _spearman(main["rank"], other["rank"]), "mean_abs_rank_delta": float(delta.abs().mean()),
                "max_abs_rank_delta": float(delta.abs().max()), "changed_stop_count": int(delta.ne(0).sum())}

    boundary_delta = external_rank["rank"] - main["rank"]
    return {"radius_300m_vs_500m": comparison(rank300),
            "within_vs_citywide": {"score_spearman": _spearman(main.score, city.score),
                "within_rank_vs_city_rank_spearman": _spearman(main["rank"], city.city_rank),
                "mean_abs_rank_context_difference": float((main["rank"] - city.city_rank).abs().mean())},
            "boundary_adjusted": {"affected_stop_count": int(affected.sum()),
                "affected_mean_abs_rank_delta": float(boundary_delta[affected].abs().mean()) if affected.any() else 0.0,
                "affected_max_abs_rank_delta": float(boundary_delta[affected].abs().max()) if affected.any() else 0.0,
                "affected_max_abs_score_delta": float((external_rank.score-main.score)[affected].abs().max()) if affected.any() else 0.0},
            "route_zero_alternative": comparison(route_variant) if route_variant is not None else {"applicable": False}}


def score_distribution(values: pd.Series) -> dict[str, float]:
    q1, median, q3 = values.quantile([.25, .5, .75])
    return {"min": float(values.min()), "q1": float(q1), "median": float(median), "q3": float(q3),
            "max": float(values.max()), "mean": float(values.mean()), "std": float(values.std())}


def validate_scenario_definition(scenario: AnchorScenario) -> None:
    assert not (set(scenario.features) & QA_FEATURES)
    assert "top3_hour_share" not in " ".join(scenario.features)
    if scenario.key == "supply_gap":
        assert not any("ridership" in feature for feature in scenario.features)
