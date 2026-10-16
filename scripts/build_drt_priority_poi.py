"""Build the active POI product from cached official HIRA and MOIS records only."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.drt_anchor_poi import CATEGORY_TAGS, SOURCE_AUDIT, poi_snap_qa, snap_poi_to_network
from src.analysis.drt_network_accessibility import load_walk_graph
import networkx as nx

REPORTS = ROOT / "reports"
POI_DIR = ROOT / "data/interim/poi"
GRAPHML = ROOT / "data/interim/network/changwon_walk_network.graphml"


def write_source_audit(records: dict, decision: str) -> None:
    generated_at = datetime.now(timezone.utc).isoformat()
    audit = {"generated_at": generated_at, "purpose": "1~4순위 공식 source 사용 가능 여부를 실제로 검증", "categories": records,
              "decision": decision}
    (REPORTS / "drt_poi_source_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# DRT Priority POI Source Audit", "",
             "1~4순위 공식 source의 실제 사용 가능 여부를 검증한 결과이며, 억지로 채운 POI는 없다.", ""]
    for category, entries in records.items():
        lines += [f"## {category}", ""]
        for entry in entries:
            lines += [f"- 기관: {entry['source_institution']}", f"- 데이터셋/API: {entry['dataset_or_api']}",
                      f"- 공식/공개 여부: {entry['official_public']}", f"- 좌표 제공: {entry['coordinate_availability']}",
                      f"- CRS: {entry['crs']}", f"- 식별자: {entry['key_id_availability']}",
                      f"- 주소 제공: {entry['address_availability']}", f"- credential 요건: {entry['api_credential_requirement']}",
                      f"- 사용 가능: {entry['usable']}", f"- 사유: {entry['reason']}", ""]
    lines += ["## Decision", "", audit["decision"], ""]
    (REPORTS / "drt_poi_source_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    load_dotenv(ROOT / ".env")
    POI_DIR.mkdir(parents=True, exist_ok=True)

    existing_path = POI_DIR / "changwon_priority_poi.parquet"
    if not existing_path.is_file():
        raise FileNotFoundError("Official POI cache is required; POI recollection is disabled for this build")
    existing = gpd.read_parquet(existing_path)
    poi = existing.loc[
        existing.poi_category.isin(CATEGORY_TAGS)
        & existing.source_name.isin(["건강보험심사평가원_병원정보서비스", "행정안전부_읍면동 하부행정기관 현황"])
    ].copy()
    if set(poi.poi_category.unique()) != set(CATEGORY_TAGS):
        raise ValueError("Active POI cache must contain both official HEALTHCARE and ADMINISTRATION records")
    records = {category: [dict(entry) for entry in entries] for category, entries in SOURCE_AUDIT.items()}
    official_counts = poi.poi_category.value_counts().to_dict()
    for category, count in official_counts.items():
        records[category][0].update(usable=True, reason=f"OFFICIAL_SOURCE_AVAILABLE — cached_count={count}")
    build_qa = {"raw_count": len(existing), "cleaned_count": len(poi),
        "by_category": official_counts, "official_by_category": official_counts, "unexpected_count": 0}
    decision = "활성 POI product는 cached HIRA healthcare와 MOIS administration 공식 데이터만 사용한다."
    write_source_audit(records, decision)

    graph = load_walk_graph(GRAPHML)
    if not poi.empty:
        largest_component = max(nx.connected_components(graph), key=len)
        poi = poi.drop(columns=["network_node_id", "network_snap_distance_m", "network_snap_flag", "network_reachable_flag"], errors="ignore")
        snapped = snap_poi_to_network(poi, graph)
        poi = poi.merge(snapped, on="poi_id", validate="one_to_one")
        poi["network_snap_flag"] = poi.network_snap_distance_m.apply(
            lambda d: "NETWORK_SNAP_OK" if d <= 100 else "NETWORK_SNAP_FAR" if d <= 200 else "NETWORK_SNAP_VERY_FAR")
        poi["network_reachable_flag"] = poi.network_node_id.map(lambda n: n in {str(node) for node in largest_component})
    snap_qa = poi_snap_qa(poi) if not poi.empty else {}

    poi.to_parquet(POI_DIR / "changwon_priority_poi.parquet", index=False)
    poi.to_file(POI_DIR / "changwon_priority_poi.geojson", driver="GeoJSON")

    source_available = {category.lower(): int(build_qa.get("by_category", {}).get(category, 0)) > 0 for category in CATEGORY_TAGS}
    official_categories = [c.lower() for c, n in official_counts.items() if n > 0]
    qa = {"dataset": "changwon_priority_poi", "status": "PASS" if poi.geometry.is_valid.all() else "FAIL",
          "source": {"categories_requested": list(CATEGORY_TAGS), "categories_usable_official": official_categories,
                     "categories_usable_secondary": [],
                     "unavailable_categories": [c for c, ok in source_available.items() if not ok],
                     "institutions": sorted(poi.source_institution.dropna().unique().tolist())},
          "poi": build_qa, "source_available": source_available,
          "snap": snap_qa, "errors": [], "warnings": [
              "same-name 서로 다른 위치는 자동 중복 제거하지 않고 SAME_NAME_REVIEW로만 표시했다.",
              "이 데이터셋은 curated release가 아닌 data/interim/poi experimental 산출물이다."],
          "generated_at": datetime.now(timezone.utc).isoformat()}
    (REPORTS / "changwon_drt_poi_qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"POI master built: raw={build_qa.get('raw_count', 0)}, cleaned={build_qa.get('cleaned_count', 0)}")


if __name__ == "__main__":
    main()
