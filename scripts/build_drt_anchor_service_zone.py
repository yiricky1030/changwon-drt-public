"""Build and optionally load the experimental Anchor walk-access zones."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_DLL_HANDLES = []
if os.name == "nt" and hasattr(os, "add_dll_directory"):
    qgis_bin = Path(sys.base_prefix).parents[1] / "bin"
    if qgis_bin.exists():
        _DLL_HANDLES.append(os.add_dll_directory(str(qgis_bin)))

import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.drt_anchor_service_zone import (BUFFER_M, DISTANCES, METHOD, SIMPLIFY_M,
    SOURCE, build_service_zones, physical_anchors, validate_service_zones)
from src.analysis.drt_network_accessibility import load_walk_graph
from src.database.connection import get_engine

PARQUET = ROOT / "reports/drt_anchor_service_zone.parquet"
GEOJSON = ROOT / "reports/drt_anchor_service_zone.geojson"
QA = ROOT / "reports/changwon_drt_anchor_service_zone_qa.json"


def load_postgis(zones: gpd.GeoDataFrame) -> dict:
    load_dotenv(ROOT / ".env")
    engine = get_engine()
    with engine.begin() as connection:
        zones.to_postgis("drt_anchor_service_zone", connection, schema="transport", if_exists="replace", index=False)
        connection.execute(text("ALTER TABLE transport.drt_anchor_service_zone ALTER COLUMN stop_id SET NOT NULL"))
        connection.execute(text("ALTER TABLE transport.drt_anchor_service_zone ALTER COLUMN distance_m SET NOT NULL"))
        connection.execute(text("ALTER TABLE transport.drt_anchor_service_zone ADD PRIMARY KEY (stop_id, distance_m)"))
        connection.execute(text("DROP INDEX IF EXISTS transport.idx_drt_anchor_service_zone_geometry"))
        connection.execute(text("CREATE INDEX drt_anchor_service_zone_geometry_gix ON transport.drt_anchor_service_zone USING GIST (geometry)"))
        connection.execute(text("CREATE INDEX drt_anchor_service_zone_adm_idx ON transport.drt_anchor_service_zone (adm_cd)"))
        connection.execute(text("CREATE INDEX drt_anchor_service_zone_distance_idx ON transport.drt_anchor_service_zone (distance_m)"))
    with engine.connect() as connection:
        row = connection.execute(text("SELECT count(*), count(*) FILTER (WHERE NOT ST_IsValid(geometry)), min(ST_SRID(geometry)) FROM transport.drt_anchor_service_zone")).one()
        indexes = connection.execute(text("SELECT indexdef FROM pg_indexes WHERE schemaname='transport' AND tablename='drt_anchor_service_zone'")).scalars().all()
    engine.dispose()
    return {"loaded": True, "table": "transport.drt_anchor_service_zone", "rows": int(row[0]),
            "invalid_geometry": int(row[1]), "srid": int(row[2]),
            "primary_key": any("unique" in value.lower() and "stop_id, distance_m" in value.lower() for value in indexes),
            "geometry_gist": any("using gist" in value.lower() for value in indexes),
            "adm_index": any("(adm_cd)" in value for value in indexes),
            "distance_index": any("(distance_m)" in value for value in indexes)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-db", action="store_true")
    parser.add_argument("--cached-only", action="store_true")
    args = parser.parse_args()
    graph = load_walk_graph(ROOT / "data/interim/network/changwon_walk_network.graphml")
    if args.cached_only and PARQUET.exists():
        zones = gpd.read_parquet(PARQUET)
    else:
        access = gpd.read_parquet(ROOT / "data/release/changwon_drt_stop_network_accessibility.parquet")
        demand = pd.read_parquet(ROOT / "reports/drt_anchor_confidence_demand.parquet")
        supply = pd.read_parquet(ROOT / "reports/drt_anchor_confidence_supply.parquet")
        anchors = physical_anchors(demand, supply, access)
        poi = pd.read_parquet(ROOT / "reports/drt_priority_poi_with_reliability.parquet")
        bridge = pd.read_parquet(ROOT / "data/release/changwon_transit_route_stop.parquet")
        zones = build_service_zones(graph, anchors, access, bridge, poi)
    validate_service_zones(zones)
    zones.to_parquet(PARQUET, index=False)
    zones.to_file(GEOJSON, driver="GeoJSON", index=False)
    metric = zones.to_crs(5179)
    qa = {"dataset": "changwon_drt_anchor_service_zone", "status": "PASS",
          "prototype_status": "experimental_accessibility_prototype",
          "not_a_drt_operating_service_area": True, "anchors": int(zones.stop_id.nunique()),
          "distances_m": list(DISTANCES), "rows": len(zones),
          "geometry": {"invalid": int((~zones.geometry.is_valid).sum()), "empty": int(zones.geometry.is_empty.sum()),
                       "median_area_m2": {str(distance): float(metric.loc[metric.distance_m.eq(distance)].geometry.area.median()) for distance in DISTANCES}},
          "network": {"source": SOURCE, "crs": str(graph.graph.get("crs")), "polygonization": METHOD,
                      "edge_and_node_buffer_m": BUFFER_M, "simplification_m": SIMPLIFY_M},
          "unreachable": int(zones.status.ne("OK").sum()),
          "warnings": ["This is not a DRT operating service area.",
                       "Distance is walk-network metres, not actual walking time.",
                       "Buffered reachable edges are a visualization prototype and do not imply population coverage."]}
    qa["postgis"] = {"loaded": False, "reason": "--skip-db"} if args.skip_db else load_postgis(zones)
    QA.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Anchor service zones: anchors={qa['anchors']}, rows={qa['rows']}, unreachable={qa['unreachable']}, DB={qa['postgis']['loaded']}")


if __name__ == "__main__":
    main()
