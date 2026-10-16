"""Build, publish, compare, and optionally load pedestrian accessibility."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_DLL_HANDLES = []
if os.name == "nt" and hasattr(os, "add_dll_directory"):
    qgis_bin = Path(sys.base_prefix).parents[1] / "bin"
    if qgis_bin.exists():
        _DLL_HANDLES.append(os.add_dll_directory(str(qgis_bin)))

import geopandas as gpd
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.drt_network_accessibility import build_accessibility, load_walk_graph, network_qa, validate_accessibility
from src.database.connection import get_engine

GRAPHML = ROOT / "data/interim/network/changwon_walk_network.graphml"
PARQUET = ROOT / "data/release/changwon_drt_stop_network_accessibility.parquet"
GEOJSON = ROOT / "data/release/changwon_drt_stop_network_accessibility.geojson"
QA = ROOT / "reports/changwon_drt_stop_network_accessibility_qa.json"


def stats(series: pd.Series) -> dict:
    clean = series.dropna().astype(float)
    return {"count": int(len(clean)), "null": int(series.isna().sum()), "min": float(clean.min()),
            "q1": float(clean.quantile(.25)), "median": float(clean.median()), "q3": float(clean.quantile(.75)),
            "p95": float(clean.quantile(.95)), "max": float(clean.max()), "mean": float(clean.mean())}


def correlations(combined: pd.DataFrame) -> dict:
    pairs = {
        "nearest_distance": ("nearest_stop_distance_m", "nearest_stop_walk_distance_m"),
        "nearby_500m": ("nearby_stops_500m", "nearby_stops_walk_500m"),
        "local_stop_scarcity": ("local_stop_scarcity_pct_within_dong", "local_stop_walk_scarcity_pct_within_dong"),
        "local_route_scarcity": ("local_route_scarcity_pct_within_dong", "local_route_walk_scarcity_pct_within_dong"),
    }
    return {name: {"pearson": float(combined[[a,b]].corr(method="pearson").iloc[0,1]),
                   "spearman": float(combined[[a,b]].corr(method="spearman").iloc[0,1])} for name,(a,b) in pairs.items()}


def rank_diagnostic(combined: pd.DataFrame) -> dict:
    scenarios = {
        "demand_supported": (combined.analysis_ready, [.35,.20,.25,.20],
            ["ridership_pct_within_dong", "route_supply_scarcity_pct_within_dong", "local_stop_scarcity_pct_within_dong", "spatial_isolation_pct_within_dong"],
            ["ridership_pct_within_dong", "route_supply_scarcity_pct_within_dong", "local_stop_walk_scarcity_pct_within_dong", "walk_isolation_pct_within_dong"]),
        "supply_gap": (pd.Series(True,index=combined.index), [.25]*4,
            ["route_supply_scarcity_pct_within_dong", "local_stop_scarcity_pct_within_dong", "local_route_scarcity_pct_within_dong", "spatial_isolation_pct_within_dong"],
            ["route_supply_scarcity_pct_within_dong", "local_stop_walk_scarcity_pct_within_dong", "local_route_walk_scarcity_pct_within_dong", "walk_isolation_pct_within_dong"]),
    }
    output = {}
    for key,(mask,weights,old,new) in scenarios.items():
        frame=combined.loc[mask].copy(); weights=np.array(weights)
        frame["old_score"]=frame[old].to_numpy(float)@weights; frame["new_score"]=frame[new].to_numpy(float)@weights
        frame["old_rank"]=frame.groupby("adm_cd").old_score.rank(method="min",ascending=False)
        frame["new_rank"]=frame.groupby("adm_cd").new_score.rank(method="min",ascending=False)
        delta=(frame.new_rank-frame.old_rank).abs()
        output[key]={"rank_spearman":float(frame[["old_rank","new_rank"]].corr(method="spearman").iloc[0,1]),
            "mean_abs_rank_delta":float(delta.mean()),"median_abs_rank_delta":float(delta.median()),
            "p95_abs_rank_delta":float(delta.quantile(.95)),"max_abs_rank_delta":float(delta.max()),
            "over_10":int(delta.gt(10).sum()),"over_25":int(delta.gt(25).sum()),"over_50":int(delta.gt(50).sum())}
    return output


def load_route_sets(base):
    bridge=pd.read_parquet(ROOT/"data/release/changwon_transit_route_stop.parquet")
    bridge=bridge.loc[bridge.is_matched & bridge.stop_id.notna()].copy(); bridge.stop_id=bridge.stop_id.astype(str)
    bridge=bridge[bridge.stop_id.isin(set(base.stop_id.astype(str)))]
    return bridge.groupby("stop_id").route_id.agg(lambda x:set(x)).to_dict()


def load_postgis(access):
    load_dotenv(ROOT/".env"); engine=get_engine()
    with engine.begin() as connection:
        access.to_postgis("drt_stop_network_accessibility",connection,schema="transport",if_exists="replace",index=False)
        connection.execute(text("ALTER TABLE transport.drt_stop_network_accessibility ALTER COLUMN stop_id SET NOT NULL"))
        connection.execute(text("ALTER TABLE transport.drt_stop_network_accessibility ADD PRIMARY KEY (stop_id)"))
        connection.execute(text("CREATE INDEX drt_stop_network_accessibility_geometry_gix ON transport.drt_stop_network_accessibility USING GIST (geometry)"))
        connection.execute(text("CREATE INDEX drt_stop_network_accessibility_adm_idx ON transport.drt_stop_network_accessibility (adm_cd)"))
        count=connection.execute(text("SELECT count(*) FROM transport.drt_stop_network_accessibility")).scalar_one()
    with engine.connect() as connection:
        primary_key=bool(connection.execute(text("SELECT EXISTS(SELECT 1 FROM pg_constraint WHERE conrelid='transport.drt_stop_network_accessibility'::regclass AND contype='p')")).scalar_one())
        geometry_gist=bool(connection.execute(text("SELECT EXISTS(SELECT 1 FROM pg_indexes WHERE schemaname='transport' AND tablename='drt_stop_network_accessibility' AND indexdef ILIKE '%USING gist%')")).scalar_one())
        adm_index=bool(connection.execute(text("SELECT EXISTS(SELECT 1 FROM pg_indexes WHERE schemaname='transport' AND tablename='drt_stop_network_accessibility' AND indexdef LIKE '%(adm_cd)%')")).scalar_one())
        srid=int(connection.execute(text("SELECT DISTINCT ST_SRID(geometry) FROM transport.drt_stop_network_accessibility")).scalar_one())
    engine.dispose(); return {"loaded":True,"table":"transport.drt_stop_network_accessibility","rows":int(count),"primary_key":primary_key,"geometry_gist":geometry_gist,"adm_index":adm_index,"srid":srid}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--skip-db",action="store_true"); parser.add_argument("--cached-only",action="store_true"); args=parser.parse_args()
    started=time.perf_counter(); graph=load_walk_graph(GRAPHML)
    base=gpd.read_parquet(ROOT/"data/release/changwon_drt_stop_features.parquet")
    if args.cached_only and PARQUET.exists(): access=gpd.read_parquet(PARQUET)
    else:
        access=build_accessibility(base,graph,load_route_sets(base)); validate_accessibility(access,graph)
        access["dataset_version"]="1.2.0"; access["generated_at"]=datetime.now(timezone.utc).isoformat()
        access.to_parquet(PARQUET,index=False); access.to_file(GEOJSON,driver="GeoJSON",index=False)
    combined=base.merge(access.drop(columns="geometry"),on=["stop_id","stop_name","adm_cd","adm_nm","anchor_data_status","analysis_ready","has_observed_profile"],validate="one_to_one")
    comparison=correlations(combined); rank=rank_diagnostic(combined); boundary=combined.near_city_boundary_500m
    high=combined.loc[combined.walk_detour_ratio.notna()].nlargest(20,"walk_detour_ratio")[["stop_id","stop_name","adm_nm","walk_detour_ratio","snap_distance_m","nearest_stop_distance_m","nearest_stop_walk_distance_m"]]
    snap=access.snap_distance_m
    qa={"dataset":"changwon_drt_stop_network_accessibility","status":"PASS","source":"OpenStreetMap","network_type":"walk",
        "network":network_qa(graph),"snap":{"total_stops":len(access),"success":int(access.network_node_id.notna().sum()),
            "distance_m":stats(snap),"over_50m":int(snap.gt(50).sum()),"over_100m":int(snap.gt(100).sum()),"over_200m":int(snap.gt(200).sum()),"over_300m":int(snap.gt(300).sum())},
        "accessibility":{"reachable":int(access.network_reachable_flag.sum()),"unreachable":int((~access.network_reachable_flag).sum()),
            "nearest_walk_distance_m":stats(access.nearest_stop_walk_distance_m),
            "observed_nearest_walk_distance_m":stats(access.nearest_observed_stop_walk_distance_m),
            "analysis_ready_nearest_walk_distance_m":stats(access.nearest_analysis_ready_stop_walk_distance_m),
            "nearby_walk_300m":stats(access.nearby_stops_walk_300m),"nearby_walk_500m":stats(access.nearby_stops_walk_500m),
            "nearby_walk_800m":stats(access.nearby_stops_walk_800m),
            "nearby_observed_walk_500m":stats(access.nearby_observed_stops_walk_500m),
            "nearby_analysis_ready_walk_500m":stats(access.nearby_analysis_ready_stops_walk_500m),
            "nearby_routes_walk_500m":stats(access.nearby_route_count_walk_500m),
            "nearby_routes_walk_800m":stats(access.nearby_route_count_walk_800m),
            "detour_ratio":{**stats(access.walk_detour_ratio),"over_2":int(access.walk_detour_ratio.gt(2).sum()),
                "over_5":int(access.walk_detour_ratio.gt(5).sum()),"over_10":int(access.walk_detour_ratio.gt(10).sum())}},
        "comparison":comparison,"high_detour_examples":high.replace({np.nan:None}).to_dict("records"),
        "boundary":{"stops":int(boundary.sum()),"straight_nearby_500_median":float(combined.loc[boundary,"nearby_stops_500m"].median()),
            "walk_nearby_500_median":float(combined.loc[boundary,"nearby_stops_walk_500m"].median()),
            "median_neighbor_delta":float((combined.loc[boundary,"nearby_stops_walk_500m"]-combined.loc[boundary,"nearby_stops_500m"]).median())},
        "anchor_base_rank_diagnostic":rank,"walking_speed_assumption":"4.8 km/h (80 m/min) for interpretation only; stored features are network metres",
        "runtime_seconds":time.perf_counter()-started,"errors":[],"warnings":["OSM completeness varies by place and date.","Snap links are analytical connectors, not mapped walkways.","Walking distance is not vehicle or DRT routing distance."]}
    qa["postgis"]={"loaded":False,"reason":"--skip-db"} if args.skip_db else load_postgis(access)
    QA.write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Network accessibility: rows={len(access)}, reachable={access.network_reachable_flag.sum()}, runtime={qa['runtime_seconds']:.1f}s, DB={qa['postgis']['loaded']}")

if __name__=="__main__": main()
