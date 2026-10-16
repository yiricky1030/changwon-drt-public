"""Confidence diagnostic layer for the walk-network Anchor sensitivity results.

This module does not produce a new Anchor score and does not change any
existing sensitivity or contribution output. It only reads the existing
network sensitivity, Euclidean-vs-network comparison, and contribution
reports and attaches independent, explainable caution/robustness flags per
stop. No flag causes automatic exclusion; NULL network values (unreachable
stops) stay NULL.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import yaml

REPEATED_HIGH_TOP3 = 0.8  # same threshold analyze_drt_anchor_network_sensitivity.py uses for "top region"
CAUTION_COLUMNS = ["network_model_sensitive", "snap_caution", "boundary_context_caution",
                   "route_zero_context", "low_sample_dong", "single_feature_dominated", "high_detour_caution"]


def snap_thresholds(metadata_path) -> dict[str, float]:
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    thresholds = metadata["snap_qa_thresholds"]
    return {"ok_max_m": float(thresholds["ok_max_m"]), "far_max_m": float(thresholds["far_max_m"])}


def citywide_detour_p95(network: pd.DataFrame) -> float:
    return float(network.walk_detour_ratio.dropna().quantile(.95))


def build_confidence(scenario_key: str, sensitivity: pd.DataFrame, euclidean: pd.DataFrame,
                      comparison: pd.DataFrame, contributions: pd.DataFrame, features: pd.DataFrame,
                      detour_p95: float) -> pd.DataFrame:
    base = sensitivity[["stop_id", "stop_name", "adm_cd", "adm_nm", "snap_distance_m", "walk_detour_ratio",
                        "network_reachable_flag", "route_count", "nearby_route_count_500m",
                        "mean_rank_within_dong", "top3_frequency_within_dong", "top10pct_frequency_within_dong",
                        "low_sample_dong", "route_zero_direct_match", "network_score_available"]].copy()
    base = base.rename(columns={"mean_rank_within_dong": "network_mean_rank",
                                 "top3_frequency_within_dong": "network_top3_frequency",
                                 "top10pct_frequency_within_dong": "network_top10pct_frequency"})
    base["scenario"] = scenario_key

    euclid = euclidean[["stop_id", "mean_rank_within_dong", "top3_frequency_within_dong",
                        "top10pct_frequency_within_dong"]].rename(columns={
        "mean_rank_within_dong": "euclidean_mean_rank", "top3_frequency_within_dong": "euclidean_top3_frequency",
        "top10pct_frequency_within_dong": "euclidean_top10pct_frequency"})
    frame = base.merge(euclid, on="stop_id", validate="one_to_one")

    comp = comparison.loc[comparison.scenario.eq(scenario_key),
                          ["stop_id", "robust_both_top3", "robust_both_top10pct", "network_sensitive",
                           "snap_over_100m", "snap_over_200m"]]
    frame = frame.merge(comp, on="stop_id", validate="one_to_one")

    contrib = contributions.loc[contributions.scenario.eq(scenario_key),
                                ["stop_id", "single_feature_dominated", "dominant_feature"]]
    frame = frame.merge(contrib, on="stop_id", validate="one_to_one")

    boundary = features[["stop_id", "near_city_boundary_500m"]]
    frame = frame.merge(boundary, on="stop_id", validate="one_to_one")

    frame["robust_both"] = frame.robust_both_top3 | frame.robust_both_top10pct
    frame["network_model_sensitive"] = frame.network_sensitive
    frame["snap_caution"] = frame.snap_over_100m
    frame["snap_high_caution"] = frame.snap_over_200m
    frame["boundary_context_caution"] = frame.near_city_boundary_500m
    frame["route_zero_context"] = frame.route_zero_direct_match & frame.nearby_route_count_500m.gt(0)
    frame["network_unreachable"] = ~frame.network_reachable_flag
    frame["high_detour_caution"] = frame.walk_detour_ratio.ge(detour_p95).fillna(False)
    frame["repeated_high"] = frame.network_score_available & frame.network_top3_frequency.ge(REPEATED_HIGH_TOP3)

    frame["caution_count"] = frame[CAUTION_COLUMNS].sum(axis=1).astype(int)
    frame["confidence_class"] = _confidence_class(frame)
    frame["robust_clean"] = (frame.robust_both & ~frame.snap_caution & ~frame.high_detour_caution &
                             ~frame.network_model_sensitive & frame.network_reachable_flag)

    columns = ["scenario", "stop_id", "stop_name", "adm_cd", "adm_nm",
               "network_mean_rank", "network_top3_frequency", "network_top10pct_frequency",
               "euclidean_mean_rank", "euclidean_top3_frequency", "euclidean_top10pct_frequency",
               "robust_both_top3", "robust_both_top10pct", "robust_both", "network_model_sensitive",
               "snap_distance_m", "snap_caution", "snap_high_caution", "walk_detour_ratio", "high_detour_caution",
               "boundary_context_caution", "route_zero_context", "low_sample_dong",
               "single_feature_dominated", "dominant_feature", "network_reachable_flag", "network_unreachable",
               "network_score_available", "repeated_high", "caution_count", "confidence_class", "robust_clean"]
    return frame[columns]


def _confidence_class(frame: pd.DataFrame) -> np.ndarray:
    high_caution = (frame.snap_high_caution | frame.caution_count.ge(3) |
                    (frame.network_model_sensitive & frame.snap_caution))
    high_confidence = (frame.robust_both & ~frame.snap_caution & ~frame.high_detour_caution &
                       ~frame.network_model_sensitive)
    return np.select(
        [frame.network_unreachable, high_caution, high_confidence],
        ["UNSCORABLE_NETWORK", "HIGH_CAUTION", "HIGH_CONFIDENCE_CONTEXT"],
        default="MODERATE_CAUTION")


def scenario_summary(frame: pd.DataFrame) -> dict[str, Any]:
    scored = frame.loc[frame.network_score_available]
    repeated = frame.loc[frame.repeated_high]
    return {"total": int(len(frame)), "scored": int(len(scored)), "repeated_high": int(len(repeated)),
            "robust_both_top3": int(frame.robust_both_top3.sum()), "robust_both_top10pct": int(frame.robust_both_top10pct.sum()),
            "network_model_sensitive": int(frame.network_model_sensitive.sum()),
            "snap_caution": int(frame.snap_caution.sum()), "snap_high_caution": int(frame.snap_high_caution.sum()),
            "high_detour_caution": int(frame.high_detour_caution.sum()),
            "boundary_context_caution": int(frame.boundary_context_caution.sum()),
            "route_zero_context": int(frame.route_zero_context.sum()),
            "low_sample_dong": int(frame.low_sample_dong.sum()),
            "single_feature_dominated": int(frame.single_feature_dominated.sum()),
            "network_unreachable": int(frame.network_unreachable.sum()),
            "confidence_class_distribution": frame.confidence_class.value_counts().to_dict(),
            "repeated_high_confidence_class_distribution": repeated.confidence_class.value_counts().to_dict(),
            "robust_clean": int(frame.robust_clean.sum())}


def caution_combinations(frame: pd.DataFrame, top: int = 10) -> list[dict[str, Any]]:
    active = frame.loc[frame.caution_count.gt(0)].copy()
    if active.empty:
        return []
    active["combination"] = active[CAUTION_COLUMNS].apply(
        lambda row: "+".join(sorted(col.upper() for col in CAUTION_COLUMNS if row[col])), axis=1)
    counts = active.combination.value_counts().head(top)
    return [{"combination": key, "count": int(value)} for key, value in counts.items()]


def dong_pattern(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for adm_cd, group in frame.groupby("adm_cd"):
        rows.append({"adm_cd": adm_cd, "adm_nm": group.adm_nm.iloc[0],
                     "repeated_high_count": int(group.repeated_high.sum()),
                     "robust_clean_count": int(group.robust_clean.sum()),
                     "high_caution_count": int(group.confidence_class.eq("HIGH_CAUTION").sum())})
    return rows
