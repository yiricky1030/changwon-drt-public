"""Anchor -> Priority POI walk accessibility context for robust_clean stops.

No POI score, destination score, or Anchor score is produced. Reads only
existing confidence/network outputs and the freshly-built POI master; no
Anchor ranking is recomputed."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.drt_anchor_poi import CATCHMENT_RADII, PRIORITY_CATEGORIES, anchor_poi_context
from src.analysis.drt_network_accessibility import load_walk_graph
from src.analysis.physical_stop_eligibility import attach_physical_eligibility, eligible_robust

REPORTS = ROOT / "reports"
POI_PATH = ROOT / "data/interim/poi/changwon_priority_poi.parquet"
GRAPHML = ROOT / "data/interim/network/changwon_walk_network.graphml"


def robust_clean_union() -> tuple[pd.DataFrame, dict]:
    demand = pd.read_parquet(REPORTS / "drt_anchor_confidence_demand.parquet")
    supply = pd.read_parquet(REPORTS / "drt_anchor_confidence_supply.parquet")
    demand = attach_physical_eligibility(demand, ROOT)
    supply = attach_physical_eligibility(supply, ROOT)
    demand_ids = set(demand.loc[eligible_robust(demand), "stop_id"])
    supply_ids = set(supply.loc[eligible_robust(supply), "stop_id"])
    union_ids = demand_ids | supply_ids
    membership = pd.DataFrame({"stop_id": sorted(union_ids)})
    membership["is_demand_robust_clean"] = membership.stop_id.isin(demand_ids)
    membership["is_supply_robust_clean"] = membership.stop_id.isin(supply_ids)
    counts = {"demand_robust_clean": len(demand_ids), "supply_robust_clean": len(supply_ids),
              "union_robust_clean": len(union_ids), "overlap": len(demand_ids & supply_ids)}
    return membership, counts


def coverage_summary(context: pd.DataFrame) -> dict:
    output = {}
    for category in PRIORITY_CATEGORIES:
        key = category.lower()
        nearest = context[f"nearest_{key}_walk_m"].dropna()
        row = {"nearest_available_count": int(nearest.notna().sum()),
               "nearest_median": float(nearest.median()) if len(nearest) else None,
               "nearest_p95": float(nearest.quantile(.95)) if len(nearest) else None}
        for radius in CATCHMENT_RADII:
            column = context[f"{key}_within_walk_{radius}m"]
            valid = column.notna()
            row[f"within_{radius}_count"] = int((column.fillna(0) > 0).sum())
            row[f"within_{radius}_share"] = float((column.fillna(0) > 0).mean()) if valid.any() else None
        output[category] = row
    return output


def main() -> None:
    membership, counts = robust_clean_union()
    network = gpd.read_parquet(ROOT / "data/release/changwon_drt_stop_network_accessibility.parquet")
    anchors = membership.merge(
        network[["stop_id", "stop_name", "adm_cd", "adm_nm", "network_node_id", "snap_distance_m", "network_reachable_flag"]],
        on="stop_id", validate="one_to_one")

    poi = gpd.read_parquet(POI_PATH)
    graph = load_walk_graph(GRAPHML)
    context_features = anchor_poi_context(anchors, poi, graph)
    context = anchors.merge(context_features, on="stop_id", validate="one_to_one")
    context.to_parquet(REPORTS / "drt_anchor_poi_context.parquet", index=False)

    demand_context = context.loc[context.is_demand_robust_clean]
    supply_context = context.loc[context.is_supply_robust_clean]
    demand_summary = coverage_summary(demand_context)
    supply_summary = coverage_summary(supply_context)

    def low_context_examples(frame: pd.DataFrame) -> list[dict]:
        ranked = frame.sort_values("poi_category_count_walk_800m", ascending=True, na_position="first")
        return ranked.head(5)[["stop_id", "stop_name", "adm_nm", "poi_category_count_walk_800m",
                                "nearest_any_priority_poi_walk_m"]].to_dict("records")

    analysis = {"generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "robust_clean Anchor 주변의 우선순위 POI 보행 접근성 context; POI 점수 미생성, Anchor 순위 재계산 없음",
        "anchor_population": counts,
        "catchment_radii_note": "500m=근거리 접근 context, 800m=약 10분 보행 proxy 규모(실제 보행시간 아님), 1200m=확장 생활권 context. 정책 확정 거리 아님.",
        "demand_supported": {"summary": demand_summary, "low_context_examples": low_context_examples(demand_context),
                              "interpretation": "관측 수요가 있는 반복 상위권(robust_clean) Anchor가 주요 생활 목적지와 연결될 가능성을 설명하는 context이며, POI가 많다고 더 나은 후보라는 의미는 아니다."},
        "supply_gap": {"summary": supply_summary, "low_context_examples": low_context_examples(supply_context),
                        "interpretation": "공급 공백상 반복 상위권(robust_clean) Anchor 주변에 생활 목적지가 존재하는지를 설명하는 context이며, Supply-gap에 실제 수요가 있다는 의미는 아니다."},
        "elderly_scenario_note": "HEALTHCARE, ADMINISTRATION은 기존 dong-level 고령자 이동지원 시나리오와 연결 가능한 explanatory category로 표시 가능하나, 특정 연령층의 실제 통행 목적을 가정하지 않는다.",
        "warnings": ["POI 개수·거리는 raw context feature이며 poi_score, destination_score 등 어떤 형태의 점수도 생성하지 않았다.",
                     "network 재계산 없이 기존 network_node_id/snap_distance_m을 그대로 재사용했다.",
                     "공식 POI도 분석 기준시점과 등록 범위에 따른 한계가 있다."],
        "errors": []}
    (REPORTS / "drt_anchor_poi_analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = ["# DRT Anchor POI Context", "",
             "robust_clean Anchor 주변 우선순위 POI 보행 접근성 context이며, POI 점수나 최종 Anchor 후보를 생성하지 않는다.", "",
             f"- Demand robust_clean: {counts['demand_robust_clean']} / Supply robust_clean: {counts['supply_robust_clean']} / union: {counts['union_robust_clean']} / overlap: {counts['overlap']}", ""]
    for label, summary in [("Demand-supported", demand_summary), ("Supply-gap", supply_summary)]:
        lines += [f"## {label}", ""]
        for category, row in summary.items():
            lines += [f"- {category}: nearest median {row['nearest_median']}, within 800m share {row['within_800_share']}"]
        lines += [""]
    lines += ["## Guardrails", "", "- POI 점수, destination score, 최종 Anchor score를 생성하지 않았다.",
              "- Anchor 순위를 재계산하지 않았다.", "- data/release, release_manifest, Dashboard는 수정하지 않았다.", ""]
    (REPORTS / "drt_anchor_poi_analysis.md").write_text("\n".join(lines), encoding="utf-8")

    qa_path = REPORTS / "changwon_drt_poi_qa.json"
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    qa["anchor"] = counts
    qa["accessibility"] = {"category_coverage": demand_summary, "supply_category_coverage": supply_summary}
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Anchor POI context complete: union={counts['union_robust_clean']}")


if __name__ == "__main__":
    main()
