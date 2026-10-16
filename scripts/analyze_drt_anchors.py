"""Generate stop-level DRT anchor feature diagnostic reports without scoring."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.drt_anchor_analysis import (
    DEMAND_FEATURES, SUPPLY_FEATURES, boundary_analysis, correlation_pairs,
    distribution_analysis, identical_coordinates, populations, scenario_validation,
    temporal_analysis, zero_route_analysis,
)

REPORTS = ROOT / "reports"


def dump(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main() -> None:
    features = gpd.read_parquet(ROOT / "data/release/changwon_drt_stop_features.parquet")
    hourly = pd.read_parquet(ROOT / "data/release/changwon_drt_stop_hourly_profile.parquet")
    master = gpd.read_parquet(ROOT / "data/release/dim_transit_stop.parquet")
    demand, supply = populations(features)
    demand_corr = correlation_pairs(demand, DEMAND_FEATURES)
    supply_corr = correlation_pairs(supply, SUPPLY_FEATURES)
    demand_corr.to_csv(REPORTS / "drt_anchor_feature_correlation_demand.csv", index=False, encoding="utf-8-sig")
    supply_corr.to_csv(REPORTS / "drt_anchor_feature_correlation_supply.csv", index=False, encoding="utf-8-sig")

    high = demand_corr.loc[demand_corr.redundancy_level.notna()].sort_values("spearman", key=lambda s: s.abs(), ascending=False)
    corr_summary = {"generated_at": datetime.now(timezone.utc).isoformat(), "demand_population": len(demand),
        "supply_population": len(supply), "thresholds": {"high": .8, "very_high": .9},
        "demand_high_redundancy": high.to_dict("records"),
        "supply_high_redundancy": supply_corr.loc[supply_corr.redundancy_level.notna()].to_dict("records")}
    dump(REPORTS / "drt_anchor_feature_correlation_summary.json", corr_summary)
    dump(REPORTS / "drt_anchor_feature_distribution.json", distribution_analysis(features))
    boundary = boundary_analysis(features, master)
    dump(REPORTS / "drt_anchor_boundary_analysis.json", boundary)
    temporal = temporal_analysis(demand, hourly)
    dump(REPORTS / "drt_anchor_temporal_stability.json", temporal)
    scenario = scenario_validation(features, demand_corr)
    scenario.update({"generated_at": datetime.now(timezone.utc).isoformat(), "zero_route": zero_route_analysis(features),
                     "identical_coordinates": identical_coordinates(features),
                     "boundary_summary": boundary, "temporal_summary": temporal,
                     "warnings": ["Eight-day ridership is not a long-term demand baseline.",
                         "TAGO matched route_count=0 is not proof that no service exists.",
                         "External boundary sensitivity covers stop counts only; external route context is unavailable."], "errors": []})
    dump(REPORTS / "drt_anchor_scenario_validation.json", scenario)
    direct = scenario["quadrants"]["direct_route_scarcity"]
    lines = ["# DRT Anchor Scenario Validation", "", "이 문서는 후보 추천이나 점수 결과가 아니라 정류소 Feature의 분석 논리 검증이다.", "",
        "## Populations", "", f"- Demand observed: {len(demand)}", f"- Supply context: {len(supply)}", "",
        "## Feature relationships", ""]
    for key, value in scenario["key_pairs"].items():
        lines.append(f"- {key}: Spearman {value['spearman']:.4f}, Pearson {value['pearson']:.4f}")
    lines += ["", "## Temporal stability", "", f"- Provisional sensitivity threshold: ridership_total >= {temporal['recommended_threshold']}",
              f"- Role: {temporal['temporal_role']}", "", "## Boundary", "",
              f"- Boundary stops: {boundary['boundary_stop_count']}",
              f"- Stops affected by external context: {boundary['external_stop_context']['boundary_stops_affected']}", "",
              "## Quadrant", "", f"- High demand + high direct scarcity: {direct['counts'].get('high_demand_high_scarcity', 0)}", "",
              "## Guardrails", "", "- No combined score or final rank was generated.", "- QA fields are confidence filters only.",
              "- Dashboard and release feature tables were not modified.", ""]
    (REPORTS / "drt_anchor_scenario_validation.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"DRT anchor diagnostics complete: demand={len(demand)}, supply={len(supply)}, reports=7")


if __name__ == "__main__":
    main()
