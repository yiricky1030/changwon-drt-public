"""Run exploratory DRT anchor weight sensitivity using walk-network features,
then compare against the existing Euclidean sensitivity outputs."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.drt_anchor_sensitivity import generate_weights
from src.analysis.drt_anchor_network_sensitivity import (
    NETWORK_SUPPLY, NETWORK_SCENARIOS, dong_level_summary,
    merge_network_features, missing_feature_report, network_base_outputs,
    network_eligible_population, network_sensitivity, scenario_level_comparison,
    score_distribution, snap_and_detour_diagnostics, stop_comparison,
    validate_scenario_definition,
)

REPORTS = ROOT / "reports"
EUCLIDEAN_KEY = {"network_demand_supported": "demand_supported", "network_supply_gap": "supply_gap"}


def main() -> None:
    features = gpd.read_parquet(ROOT / "data/release/changwon_drt_stop_features.parquet")
    network = gpd.read_parquet(ROOT / "data/release/changwon_drt_stop_network_accessibility.parquet")
    merged = merge_network_features(features, network)

    euclidean = {
        "demand_supported": pd.read_parquet(REPORTS / "drt_anchor_sensitivity_demand.parquet"),
        "supply_gap": pd.read_parquet(REPORTS / "drt_anchor_sensitivity_supply.parquet"),
    }

    stability_parts, contribution_parts, comparison_parts = [], [], []
    report: dict = {}
    for offset, scenario in enumerate(NETWORK_SCENARIOS):
        validate_scenario_definition(scenario)
        frame = network_eligible_population(merged, scenario)
        missing = missing_feature_report(frame, scenario)
        weights = generate_weights(scenario, 2000, 20260830 + offset)
        stable = network_sensitivity(frame, scenario, weights)
        base, contributions = network_base_outputs(frame, scenario)
        stability_parts.append(stable)
        contribution_parts.append(contributions)

        comparison = stop_comparison(euclidean[EUCLIDEAN_KEY[scenario.key]], stable, scenario.key)
        comparison_parts.append(comparison)
        scenario_comparison = scenario_level_comparison(comparison)
        dong_summary = dong_level_summary(comparison)
        snap_detour = snap_and_detour_diagnostics(stable)

        repeated = stable.sort_values(
            ["top3_frequency_within_dong", "top10pct_frequency_within_dong", "mean_rank_within_dong"],
            ascending=[False, False, True], na_position="last")
        top_region = repeated.loc[repeated.network_score_available & repeated.top3_frequency_within_dong.ge(.8)]

        decision = ("A_EUCLIDEAN_SUFFICIENT" if scenario_comparison["mean_rank_spearman"] >= .90
                    else "B_NETWORK_PRIMARY_EUCLIDEAN_SECONDARY" if scenario_comparison["mean_rank_spearman"] >= .75
                    else "C_NETWORK_PRIMARY")

        report[scenario.key] = {
            "purpose": "exploratory repeated-high-rank analysis using walk-network features; not a final anchor determination",
            "eligible": missing["eligible"], "eligible_scored": missing["eligible_scored"],
            "excluded_missing_network_feature": missing["excluded_missing_network_feature"],
            "combinations": int(len(weights)),
            "base_weights": dict(zip(scenario.labels, scenario.base_weights)),
            "weight_ranges": {label: list(bounds) for label, bounds in zip(scenario.labels, scenario.ranges)},
            "score_distribution": score_distribution(base.exploratory_anchor_score),
            "rank_stability": {"mean_rank_std": float(stable.rank_std_within_dong.mean()),
                "labels": stable.stability_label.value_counts().to_dict()},
            "repeated_high_examples": repeated.head(15)[["stop_id", "stop_name", "adm_nm", "candidate_stop_count",
                "top3_frequency_within_dong", "top10pct_frequency_within_dong", "mean_rank_within_dong",
                "snap_distance_m", "walk_detour_ratio"]].to_dict("records"),
            "dominant_features": contributions.dominant_feature.value_counts(dropna=True).to_dict(),
            "single_feature_dominated_count": int(contributions.single_feature_dominated.sum()),
            "snap_and_detour_top_region": snap_detour,
            "euclidean_vs_network": scenario_comparison,
            "robust_both_top3_count": int(comparison.robust_both_top3.sum()),
            "robust_both_top10pct_count": int(comparison.robust_both_top10pct.sum()),
            "network_sensitive_count": int(comparison.network_sensitive.sum()),
            "largest_rank_changes": comparison.reindex(comparison.mean_rank_delta.abs().sort_values(ascending=False).index)
                .head(10)[["stop_id", "adm_nm", "mean_rank_euclidean", "mean_rank_network", "mean_rank_delta"]]
                .to_dict("records"),
            "dong_summary": dong_summary,
            "decision_rule": decision,
        }
        if scenario is NETWORK_SUPPLY:
            report[scenario.key]["supply_only_share_top_region"] = (
                float(top_region.anchor_data_status.eq("SUPPLY_ONLY").mean()) if len(top_region) else 0.0)

    demand_stability, supply_stability = stability_parts
    demand_stability.to_parquet(REPORTS / "drt_anchor_network_sensitivity_demand.parquet", index=False)
    supply_stability.to_parquet(REPORTS / "drt_anchor_network_sensitivity_supply.parquet", index=False)
    pd.concat(contribution_parts, ignore_index=True).to_parquet(REPORTS / "drt_anchor_network_contributions.parquet", index=False)
    pd.concat(comparison_parts, ignore_index=True).to_parquet(REPORTS / "drt_anchor_euclidean_network_comparison.parquet", index=False)

    analysis = {"generated_at": datetime.now(timezone.utc).isoformat(),
        "interpretation": "Walk-network local supply and isolation features replace straight-line equivalents; "
            "observed DRT demand is not established and no final anchor is produced.",
        "scenarios": report, "warnings": [
            "Weights are sensitivity centers and ranges, not policy weights.",
            "Network isolation is NULL, never 0, for the 2 walk-unreachable stops; weights are not renormalized.",
            "Snap distance and walk detour are diagnostic context, not automatic exclusion criteria.",
            "2,000-combination sensitivity simulation was rerun only for this network variant; full Euclidean-side re-simulation was not repeated."],
        "errors": []}
    (REPORTS / "drt_anchor_network_sensitivity_analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = ["# DRT Anchor Network Sensitivity Analysis", "",
             "직선거리 기반 Anchor sensitivity와 동일한 구조에서 local supply/isolation feature를 보행 네트워크 기반으로 교체한 결과이며 최종 후보 확정이 아니다.", ""]
    for scenario in NETWORK_SCENARIOS:
        item = report[scenario.key]
        lines += [f"## {scenario.key}", "",
                  f"- Eligible stops: {item['eligible']} (scored: {item['eligible_scored']}, excluded for missing network feature: {item['excluded_missing_network_feature']})",
                  f"- Weight combinations: {item['combinations']}",
                  f"- Score median: {item['score_distribution']['median']:.3f}",
                  f"- Mean within-dong rank std: {item['rank_stability']['mean_rank_std']:.3f}",
                  f"- Euclidean vs Network mean-rank Spearman: {item['euclidean_vs_network']['mean_rank_spearman']:.4f}",
                  f"- Robust in both (top3 within-dong): {item['robust_both_top3_count']}",
                  f"- Network-sensitive stops: {item['network_sensitive_count']}",
                  f"- Decision rule: {item['decision_rule']}", ""]
    lines += ["## Guardrails", "", "- Dashboard와 curated release는 변경하지 않았다.",
              "- 기존 Euclidean sensitivity output은 수정하지 않고 읽기만 했다.",
              "- Service Zone과 실제 배치 판단은 생성하지 않았다.", ""]
    (REPORTS / "drt_anchor_network_sensitivity_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Network anchor sensitivity complete: demand={report['network_demand_supported']['eligible']}, "
          f"supply={report['network_supply_gap']['eligible']}, combinations=2000 each")


if __name__ == "__main__":
    main()
