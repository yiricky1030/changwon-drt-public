"""Exploratory DRT feature relationships and reproducible weight sensitivity."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


CANDIDATE_FEATURES = [
    "elderly_ratio_pct",
    "stop_supply_scarcity_pct",
    "route_supply_scarcity_pct",
    "ridership_per_1000_population_pct",
    "ridership_per_observed_stop_pct",
    "top3_hour_share_pct",
]
QA_FEATURES = {
    "analysis_ready_ratio", "ridership_observation_coverage_ratio",
    "missing_observation_stop_count", "duplicate_observation_stop_count",
    "has_missing_observations", "has_duplicate_excluded", "has_no_ridership_observation",
}


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    weights: dict[str, float]


SCENARIOS = [
    Scenario("elderly_support", "고령자 이동지원형", {
        "elderly_ratio_pct": 0.40, "stop_supply_scarcity_pct": 0.25,
        "route_supply_scarcity_pct": 0.20, "ridership_per_1000_population_pct": 0.15,
    }),
    Scenario("supply_complement", "교통공급 보완형", {
        "stop_supply_scarcity_pct": 0.40, "route_supply_scarcity_pct": 0.40,
        "ridership_per_1000_population_pct": 0.20,
    }),
    Scenario("demand_responsive", "수요대응형", {
        "ridership_per_1000_population_pct": 0.40,
        "stop_supply_scarcity_pct": 0.20, "route_supply_scarcity_pct": 0.20,
        "top3_hour_share_pct": 0.20,
    }),
]


def prepare_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add a rank percentile for temporal concentration while preserving its raw value."""
    output = features.copy()
    output["top3_hour_share_pct"] = output["top3_hour_share"].rank(method="average", pct=True) * 100
    return output


def correlation_analysis(features: pd.DataFrame) -> dict[str, Any]:
    data = features[CANDIDATE_FEATURES]
    pearson = data.corr(method="pearson")
    spearman = data.corr(method="spearman")
    pairs = []
    for index, left in enumerate(CANDIDATE_FEATURES):
        for right in CANDIDATE_FEATURES[index + 1:]:
            rho = float(spearman.loc[left, right])
            level = "VERY_HIGH" if abs(rho) >= 0.90 else "HIGH" if abs(rho) >= 0.80 else None
            pairs.append({"feature_1": left, "feature_2": right, "pearson": float(pearson.loc[left, right]), "spearman": rho, "redundancy_level": level})
    warnings = [pair for pair in pairs if pair["redundancy_level"]]
    return {"pearson": pearson, "spearman": spearman, "pairs": pairs, "redundancy_warnings": warnings}


def distribution_analysis(features: pd.DataFrame) -> dict[str, Any]:
    output = {}
    for column in CANDIDATE_FEATURES + ["top3_hour_share", "weekend_weekday_ratio"]:
        values = features[column].dropna()
        q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        mask = features[column].notna() & ((features[column] < q1 - 1.5 * iqr) | (features[column] > q3 + 1.5 * iqr))
        output[column] = {
            "count": int(values.size), "null": int(features[column].isna().sum()),
            "min": float(values.min()), "q1": float(q1), "median": float(median),
            "q3": float(q3), "max": float(values.max()), "mean": float(values.mean()),
            "std": float(values.std()), "iqr": float(iqr),
            "iqr_outlier_dongs": features.loc[mask, "adm_nm"].tolist(),
        }
    return output


def generate_weights(base: dict[str, float], combinations: int = 1000, seed: int = 20260830) -> pd.DataFrame:
    """Generate bounded Monte Carlo perturbations, then normalize each row to one."""
    columns = list(base)
    center = np.array([base[column] for column in columns], dtype=float)
    rng = np.random.default_rng(seed)
    raw = center + rng.uniform(-0.10, 0.10, size=(combinations, len(columns)))
    raw = np.clip(raw, 0.01, None)
    normalized = raw / raw.sum(axis=1, keepdims=True)
    return pd.DataFrame(normalized, columns=columns)


def base_scores(features: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
    columns = list(scenario.weights)
    eligible = features[columns].notna().all(axis=1)
    result = features[["adm_cd", "adm_nm"]].copy()
    result["scenario"] = scenario.key
    result["exploratory_score"] = pd.NA
    score = sum(features.loc[eligible, column] * weight for column, weight in scenario.weights.items())
    result.loc[eligible, "exploratory_score"] = score
    result["exploratory_score"] = result["exploratory_score"].astype("Float64")
    result["base_rank"] = result["exploratory_score"].rank(method="min", ascending=False).astype("Int64")
    return result


def sensitivity_analysis(features: pd.DataFrame, scenario: Scenario, combinations: int = 1000, seed: int = 20260830) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    columns = list(scenario.weights)
    eligible = features[columns].notna().all(axis=1)
    subset = features.loc[eligible, ["adm_cd", "adm_nm", *columns]].reset_index(drop=True)
    weights = generate_weights(scenario.weights, combinations, seed)
    scores = subset[columns].to_numpy(dtype=float) @ weights.to_numpy(dtype=float).T
    ranks = pd.DataFrame(scores).rank(axis=0, method="min", ascending=False).to_numpy()
    stability = subset[["adm_cd", "adm_nm"]].copy()
    stability["scenario"] = scenario.key
    stability["mean_rank"] = ranks.mean(axis=1)
    stability["median_rank"] = np.median(ranks, axis=1)
    stability["best_rank"] = ranks.min(axis=1).astype(int)
    stability["worst_rank"] = ranks.max(axis=1).astype(int)
    stability["rank_std"] = ranks.std(axis=1)
    stability["top5_frequency"] = (ranks <= 5).mean(axis=1)
    stability["top10_frequency"] = (ranks <= 10).mean(axis=1)
    stability["stability_class"] = pd.cut(
        stability["top10_frequency"], bins=[-0.001, 0.20, 0.50, 0.80, 1.001],
        labels=["LOW_FREQUENCY", "WEIGHT_SENSITIVE", "MODERATELY_STABLE", "STABLE_HIGH"], right=False,
    ).astype(str)

    base = base_scores(features, scenario).loc[eligible].reset_index(drop=True)
    contributions = base.copy()
    for column, weight in scenario.weights.items():
        contributions[f"contribution__{column}"] = subset[column] * weight
    contribution_columns = [column for column in contributions if column.startswith("contribution__")]
    contributions["contribution_sum"] = contributions[contribution_columns].sum(axis=1)
    return stability, contributions, weights


def scenario_differentiation(base_results: pd.DataFrame) -> dict[str, float]:
    wide = base_results.pivot(index="adm_cd", columns="scenario", values="base_rank")
    correlation = wide.corr(method="spearman")
    output = {}
    for index, left in enumerate(correlation.columns):
        for right in correlation.columns[index + 1:]:
            output[f"{left}__{right}"] = float(correlation.loc[left, right])
    return output


def group_profiles(features: pd.DataFrame, base: pd.DataFrame, scenario: Scenario) -> dict[str, Any]:
    merged = features.merge(base[["adm_cd", "base_rank"]], on="adm_cd", how="left")
    eligible = merged["base_rank"].notna()
    top = merged.loc[eligible].nsmallest(10, "base_rank")
    columns = list(scenario.weights)
    return {
        "top10_median": {column: float(top[column].median()) for column in columns},
        "eligible_all_median": {column: float(merged.loc[eligible, column].median()) for column in columns},
    }
