"""Audit A: how completely STCIS ridership observation covers official stops per dong.

This module answers one question only: how far does observed/analysis-ready
stop coverage reach across the 55 official dongs, and where is coverage thin
enough that ridership-based comparisons need caution. It does not judge DRT
need, and it does not convert missing observation into zero ridership.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

COVERAGE_COLUMNS = [
    "dong_code", "dong_name", "official_stop_count", "observed_stop_count",
    "analysis_ready_stop_count", "missing_observation_stop_count",
    "duplicate_excluded_stop_count", "observation_coverage", "analysis_ready_coverage",
]


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide only where the denominator is known and positive; NULL otherwise."""
    result = pd.Series(pd.NA, index=numerator.index, dtype="Float64")
    mask = denominator.notna() & denominator.gt(0) & numerator.notna()
    result.loc[mask] = numerator.loc[mask].astype(float) / denominator.loc[mask].astype(float)
    return result


def build_dong_coverage_table(context: pd.DataFrame) -> pd.DataFrame:
    """One row per official dong with observation and analysis-ready coverage."""
    table = context.rename(columns={
        "master_stop_count": "official_stop_count",
        "duplicate_observation_stop_count": "duplicate_excluded_stop_count",
    })[[
        "dong_code", "dong_name", "official_stop_count", "observed_stop_count",
        "analysis_ready_stop_count", "missing_observation_stop_count",
        "duplicate_excluded_stop_count",
    ]].copy()
    table["observation_coverage"] = safe_ratio(table["observed_stop_count"], table["official_stop_count"])
    table["analysis_ready_coverage"] = safe_ratio(table["analysis_ready_stop_count"], table["official_stop_count"])
    table["zero_official_stop_dong"] = table["official_stop_count"].eq(0)
    return table


def coverage_distribution(table: pd.DataFrame) -> dict[str, Any]:
    """Distribution of observation/analysis-ready coverage across dongs with official stops."""
    output: dict[str, Any] = {}
    for column in ("observation_coverage", "analysis_ready_coverage"):
        values = table.loc[table["official_stop_count"].gt(0), column].dropna()
        if values.empty:
            output[column] = None
            continue
        quantiles = values.quantile([0.0, 0.25, 0.5, 0.75, 1.0])
        output[column] = {
            "count": int(values.size),
            "min": float(quantiles.loc[0.0]), "p25": float(quantiles.loc[0.25]),
            "median": float(quantiles.loc[0.5]), "p75": float(quantiles.loc[0.75]),
            "max": float(quantiles.loc[1.0]), "mean": float(values.mean()), "std": float(values.std()),
        }
    return output


def top_bottom_coverage(table: pd.DataFrame, n: int = 10) -> dict[str, Any]:
    """Highest- and lowest-coverage dongs by observation_coverage, ties kept explicit."""
    eligible = table.loc[table["official_stop_count"].gt(0)].dropna(subset=["observation_coverage"])
    columns = ["dong_code", "dong_name", "official_stop_count", "observed_stop_count", "observation_coverage", "analysis_ready_coverage"]
    lowest = eligible.sort_values(["observation_coverage", "dong_code"]).head(n)
    highest = eligible.sort_values(["observation_coverage", "dong_code"], ascending=[False, True]).head(n)
    return {
        "lowest_observation_coverage": lowest[columns].to_dict(orient="records"),
        "highest_observation_coverage": highest[columns].to_dict(orient="records"),
    }


def route_supply_bias_proxy(table: pd.DataFrame, dong_features: pd.DataFrame) -> dict[str, Any]:
    """Dong-level correlation between coverage and route supply; NOT a stop-level check.

    A true observed-vs-unobserved stop route-count comparison needs the
    stop-route bridge table (changwon_transit_route_stop), which is not
    available in this environment. This proxy only tests whether dongs with
    more/less route supply differ systematically in observation coverage.
    """
    merged = table.merge(
        dong_features[[
            "adm_cd", "route_count", "zero_route_stop_ratio", "route_branch_count",
            "route_trunk_count", "route_seat_count", "route_village_count",
        ]],
        left_on="dong_code", right_on="adm_cd", how="left",
    )
    route_columns = ["route_count", "zero_route_stop_ratio", "route_branch_count", "route_trunk_count", "route_seat_count", "route_village_count"]
    correlations = {}
    for column in route_columns:
        pair = merged[["observation_coverage", column]].dropna()
        correlations[column] = {
            "n": int(len(pair)),
            "spearman": float(pair["observation_coverage"].corr(pair[column], method="spearman")) if len(pair) >= 3 else None,
        }
    return {
        "note": "dong-level proxy only; stop-level observed-vs-unobserved route_count comparison requires changwon_transit_route_stop.parquet (unavailable in this environment)",
        "correlations_with_observation_coverage": correlations,
    }


def coverage_audit_report(table: pd.DataFrame, dong_features: pd.DataFrame, official_master_total: int, official_inside_boundary_total: int, observed_total: int, analysis_ready_total: int) -> dict[str, Any]:
    """Assemble the full Audit A JSON payload."""
    distribution = coverage_distribution(table)
    extremes = top_bottom_coverage(table)
    bias = route_supply_bias_proxy(table, dong_features)
    return {
        "purpose": "STCIS ridership observation coverage across official Changwon stops per dong; not a DRT need judgment",
        "citywide_counts": {
            "official_master_stops_total": official_master_total,
            "official_stops_inside_boundary": official_inside_boundary_total,
            "observed_stops_inside_boundary": observed_total,
            "analysis_ready_stops_inside_boundary": analysis_ready_total,
            "observation_coverage_citywide": (observed_total / official_inside_boundary_total) if official_inside_boundary_total else None,
            "analysis_ready_coverage_citywide": (analysis_ready_total / official_inside_boundary_total) if official_inside_boundary_total else None,
        },
        "dong_count": int(len(table)),
        "zero_official_stop_dongs": table.loc[table["zero_official_stop_dong"], "dong_name"].tolist(),
        "coverage_distribution": distribution,
        "coverage_extremes": extremes,
        "route_supply_bias_proxy": bias,
        "warnings": [
            "Coverage below the sample median is not, by itself, evidence that a dong is unsuitable for analysis; it only means observed-stop ridership represents a smaller share of that dong's official stops.",
            "Unobserved stops are not assumed to have zero ridership; they are simply absent from the STCIS sample.",
            "Stop-level (as opposed to dong-level) observed-vs-unobserved route_count comparison could not be run: changwon_transit_route_stop.parquet is unavailable in this environment.",
        ],
    }
