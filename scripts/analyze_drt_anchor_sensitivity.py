"""Run exploratory DRT anchor weight and context sensitivity."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.drt_anchor_sensitivity import (
    DEMAND, SCENARIOS, base_outputs, eligible_population, generate_weights,
    score_distribution, sensitivity, validate_scenario_definition, variants,
)

REPORTS = ROOT / "reports"


def main() -> None:
    source = gpd.read_parquet(ROOT / "data/release/changwon_drt_stop_features.parquet")
    master = gpd.read_parquet(ROOT / "data/release/dim_transit_stop.parquet")
    stability_parts, base_parts, contribution_parts = [], [], []
    report, variant_report = {}, {}
    for offset, scenario in enumerate(SCENARIOS):
        validate_scenario_definition(scenario)
        frame = eligible_population(source, scenario)
        weights = generate_weights(scenario, 2000, 20260830 + offset)
        stable = sensitivity(frame, scenario, weights)
        base, contributions = base_outputs(frame, scenario)
        variant = variants(frame, scenario, master)
        stability_parts.append(stable); base_parts.append(base); contribution_parts.append(contributions)
        variant_report[scenario.key] = variant
        repeated = stable.sort_values(["top3_frequency_within_dong", "top10pct_frequency_within_dong", "mean_rank_within_dong"], ascending=[False, False, True])
        top_region = repeated.loc[repeated.top3_frequency_within_dong.ge(.8)]
        report[scenario.key] = {
            "purpose": "exploratory repeated-high-rank analysis; not a final anchor determination",
            "eligible": int(len(frame)), "combinations": int(len(weights)),
            "base_weights": dict(zip(scenario.labels, scenario.base_weights)),
            "weight_ranges": {label: list(bounds) for label, bounds in zip(scenario.labels, scenario.ranges)},
            "score_distribution": score_distribution(base.exploratory_anchor_score),
            "rank_stability": {"mean_rank_std": float(stable.rank_std_within_dong.mean()),
                "max_rank_span": int((stable.worst_rank_within_dong-stable.best_rank_within_dong).max()),
                "labels": stable.stability_label.value_counts().to_dict()},
            "repeated_high_examples": repeated.head(15)[["stop_id", "stop_name", "adm_nm", "candidate_stop_count",
                "top3_frequency_within_dong", "top10pct_frequency_within_dong", "mean_rank_within_dong",
                "route_zero_direct_match", "nearby_route_count_500m", "ridership_total"]].to_dict("records"),
            "low_sample_dong_count": int(stable.loc[stable.low_sample_dong, "adm_cd"].nunique()),
            "low_sample_dongs": stable.loc[stable.low_sample_dong, ["adm_cd", "adm_nm", "candidate_stop_count"]].drop_duplicates().to_dict("records"),
            "dominant_features": contributions.dominant_feature.value_counts().to_dict(),
            "single_feature_dominated_count": int(contributions.single_feature_dominated.sum()),
            "route_zero": {"count": int(stable.route_zero_direct_match.sum()),
                "top3_frequency_ge_80_count": int((stable.route_zero_direct_match & stable.top3_frequency_within_dong.ge(.8)).sum()),
                "mean_top3_frequency": float(stable.loc[stable.route_zero_direct_match, "top3_frequency_within_dong"].mean())},
            "top_region_status": top_region.anchor_data_status.value_counts().to_dict(),
        }
        if scenario is DEMAND:
            report[scenario.key]["low_ridership_top_region"] = {str(limit): {
                "count": int(top_region.ridership_total.lt(limit).sum()),
                "share": float(top_region.ridership_total.lt(limit).mean()) if len(top_region) else 0.0} for limit in [25, 50, 100]}
        else:
            report[scenario.key]["supply_only_share_top_region"] = float(top_region.anchor_data_status.eq("SUPPLY_ONLY").mean()) if len(top_region) else 0.0

    demand_stability, supply_stability = stability_parts
    demand_stability.to_parquet(REPORTS / "drt_anchor_sensitivity_demand.parquet", index=False)
    supply_stability.to_parquet(REPORTS / "drt_anchor_sensitivity_supply.parquet", index=False)
    pd.concat(base_parts, ignore_index=True).to_parquet(REPORTS / "drt_anchor_base_scores.parquet", index=False)
    pd.concat(contribution_parts, ignore_index=True).to_parquet(REPORTS / "drt_anchor_contributions.parquet", index=False)
    variant_report["generated_at"] = datetime.now(timezone.utc).isoformat()
    (REPORTS / "drt_anchor_variant_comparison.json").write_text(json.dumps(variant_report, ensure_ascii=False, indent=2), encoding="utf-8")
    analysis = {"generated_at": datetime.now(timezone.utc).isoformat(), "interpretation":
        "Supply-gap results indicate relative supply and spatial context only; observed DRT demand is not established.",
        "scenarios": report, "variants": variant_report, "warnings": [
            "Weights are sensitivity centers and ranges, not policy weights.",
            "Within-dong frequency is interpretation-limited for small candidate sets.",
            "TAGO route_count=0 can reflect unmatched direct relations.",
            "Boundary adjustment covers outside-stop counts, not outside route completeness."], "errors": []}
    (REPORTS / "drt_anchor_sensitivity_analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# DRT Anchor Sensitivity Analysis", "", "탐색적 반복 상위권과 순위 안정성을 검토한 결과이며 최종 후보 확정이 아니다.", ""]
    for scenario in SCENARIOS:
        item = report[scenario.key]
        lines += [f"## {scenario.key}", "", f"- Eligible stops: {item['eligible']}", f"- Weight combinations: {item['combinations']}",
                  f"- Score median: {item['score_distribution']['median']:.3f}", f"- Mean within-dong rank std: {item['rank_stability']['mean_rank_std']:.3f}", ""]
    lines += ["## Guardrails", "", "- Dashboard와 curated release는 변경하지 않았다.", "- Service Zone과 실제 배치 판단은 생성하지 않았다.", ""]
    (REPORTS / "drt_anchor_sensitivity_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print("Anchor sensitivity complete: demand=886, supply=2926, combinations=2000 each")


if __name__ == "__main__":
    main()
