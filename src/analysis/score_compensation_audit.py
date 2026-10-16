"""Audit C: does the weighted-sum score let one feature dominate or compensate for others?

Reuses the production scenario functions (base_scores, sensitivity_analysis,
correlation_analysis) unchanged. This module only adds diagnostics on top of
their output: it does not redesign the score or change any weight.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.analysis.drt_scenario_analysis import CANDIDATE_FEATURES, Scenario, base_scores


def contribution_dominance_table(contributions: pd.DataFrame) -> pd.DataFrame:
    """Per dong/scenario: which single feature contributes the most to the total score."""
    contribution_columns = [column for column in contributions.columns if column.startswith("contribution__")]
    values = contributions[contribution_columns]
    result = contributions[["adm_cd", "adm_nm", "scenario", "base_rank", "exploratory_score", "contribution_sum"]].copy()
    result["max_single_contribution"] = values.max(axis=1)
    result["dominant_feature"] = values.idxmax(axis=1).str.removeprefix("contribution__")
    result["max_single_contribution_share"] = np.where(
        result["contribution_sum"] > 0, result["max_single_contribution"] / result["contribution_sum"], np.nan,
    )
    top2_sum = pd.DataFrame(np.sort(values.to_numpy(), axis=1)[:, -2:].sum(axis=1), index=values.index, columns=["top2_sum"])
    result["top2_contribution_share"] = np.where(
        result["contribution_sum"] > 0, top2_sum["top2_sum"] / result["contribution_sum"], np.nan,
    )
    return result


def dominance_distribution(dominance: pd.DataFrame, top_n: int = 10) -> dict[str, Any]:
    """Distribution of max_single_contribution_share, all eligible dongs vs. TOP N by scenario."""
    output: dict[str, Any] = {}
    for scenario_key, group in dominance.groupby("scenario"):
        top = group.nsmallest(top_n, "base_rank")

        def describe(series: pd.Series) -> dict[str, float] | None:
            values = series.dropna()
            if values.empty:
                return None
            quantiles = values.quantile([0.0, 0.25, 0.5, 0.75, 1.0])
            return {
                "n": int(values.size), "min": float(quantiles.loc[0.0]), "p25": float(quantiles.loc[0.25]),
                "median": float(quantiles.loc[0.5]), "p75": float(quantiles.loc[0.75]), "max": float(quantiles.loc[1.0]),
            }

        output[scenario_key] = {
            "all_eligible": describe(group["max_single_contribution_share"]),
            "top10": describe(top["max_single_contribution_share"]),
            "top10_dominant_feature_counts": top["dominant_feature"].value_counts().to_dict(),
        }
    return output


def feature_removal_audit(features: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
    """For each feature in the scenario, drop it, renormalize the rest, and re-rank.

    Diagnostic only: this does not change the production score or weights.
    """
    columns = list(scenario.weights)
    original = base_scores(features, scenario).rename(columns={"exploratory_score": "original_score", "base_rank": "original_rank"})
    rows = [original]
    for dropped in columns:
        remaining = {column: weight for column, weight in scenario.weights.items() if column != dropped}
        total_weight = sum(remaining.values())
        normalized = {column: weight / total_weight for column, weight in remaining.items()}
        eligible = features[list(remaining)].notna().all(axis=1)
        rank = pd.Series(pd.NA, index=features.index, dtype="Int64")
        score = sum(features.loc[eligible, column] * weight for column, weight in normalized.items())
        rank.loc[eligible] = score.rank(method="min", ascending=False).astype("Int64")
        column_name = f"rank_without__{dropped}"
        piece = features[["adm_cd"]].copy()
        piece[column_name] = rank
        rows.append(piece.set_index("adm_cd")[[column_name]])
    merged = rows[0].set_index("adm_cd")
    for piece in rows[1:]:
        merged = merged.join(piece, how="left")
    merged = merged.reset_index()
    for dropped in columns:
        merged[f"rank_change__{dropped}"] = merged[f"rank_without__{dropped}"].astype("Float64") - merged["original_rank"].astype("Float64")
    return merged


def correlation_redundancy_pairs(correlation_pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pass through the existing redundancy pair list; kept separate for the audit report."""
    return sorted(correlation_pairs, key=lambda item: -abs(item["spearman"]))


def compensation_case_scan(features: pd.DataFrame, dominance: pd.DataFrame) -> dict[str, Any]:
    """Look for data-driven examples of the four compensation patterns; report none as none."""
    pct_columns = [column for column in CANDIDATE_FEATURES if column.endswith("_pct")]
    HIGH, LOW = 75.0, 25.0
    cases: dict[str, list[dict[str, Any]]] = {"case_a_elderly_dominant": [], "case_b_supply_scarce_but_low_rank": [], "case_c_single_ridership_spike": [], "case_d_broadly_mid_upper": []}

    elderly_top = dominance.merge(features[["adm_cd", "elderly_ratio_pct", "stop_supply_scarcity_pct", "route_supply_scarcity_pct"]], on="adm_cd", how="left")
    a = elderly_top.loc[
        elderly_top["scenario"].eq("elderly_support")
        & elderly_top["elderly_ratio_pct"].ge(HIGH)
        & elderly_top["stop_supply_scarcity_pct"].le(LOW)
        & elderly_top["route_supply_scarcity_pct"].le(LOW)
        & elderly_top["base_rank"].le(10)
    ]
    cases["case_a_elderly_dominant"] = a[["adm_cd", "adm_nm", "base_rank", "elderly_ratio_pct", "stop_supply_scarcity_pct", "route_supply_scarcity_pct"]].to_dict(orient="records")

    supply = features.merge(dominance.loc[dominance["scenario"].eq("supply_complement"), ["adm_cd", "base_rank"]], on="adm_cd", how="left")
    b = supply.loc[
        (supply["stop_supply_scarcity_pct"].ge(HIGH) | supply["route_supply_scarcity_pct"].ge(HIGH))
        & supply["elderly_ratio_pct"].le(LOW)
        & supply["ridership_per_1000_population_pct"].le(LOW)
        & supply["base_rank"].gt(10)
    ]
    cases["case_b_supply_scarce_but_low_rank"] = b[["adm_cd", "adm_nm", "base_rank", "stop_supply_scarcity_pct", "route_supply_scarcity_pct", "elderly_ratio_pct", "ridership_per_1000_population_pct"]].to_dict(orient="records")

    demand_dom = dominance.loc[dominance["scenario"].eq("demand_responsive")]
    high_share = demand_dom["max_single_contribution_share"].quantile(0.75)
    c = demand_dom.loc[
        demand_dom["dominant_feature"].eq("ridership_per_1000_population_pct")
        & demand_dom["max_single_contribution_share"].ge(high_share)
        & demand_dom["base_rank"].le(10)
    ]
    cases["case_c_single_ridership_spike"] = c[["adm_cd", "adm_nm", "base_rank", "max_single_contribution_share"]].to_dict(orient="records")

    mid_upper = features[pct_columns].ge(50).all(axis=1) & features[pct_columns].le(85).all(axis=1)
    stable_candidates = features.loc[mid_upper, ["adm_cd", "adm_nm", *pct_columns]]
    top_any_scenario = set(dominance.loc[dominance["base_rank"].le(10), "adm_cd"])
    d = stable_candidates.loc[stable_candidates["adm_cd"].isin(top_any_scenario)]
    cases["case_d_broadly_mid_upper"] = d.to_dict(orient="records")

    return cases
