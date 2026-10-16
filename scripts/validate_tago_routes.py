"""Validate route release artifacts and their PostGIS representation."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
DLL_DIRECTORY_HANDLES=[]
if os.name=="nt" and hasattr(os,"add_dll_directory"):
    qgis_bin=Path(sys.base_prefix).parents[1]/"bin"
    if qgis_bin.exists(): DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(qgis_bin)))
from src.database.connection import get_engine

def main():
    load_dotenv(ROOT/".env")
    route_pq=gpd.read_parquet(ROOT/"data/release/changwon_transit_route.parquet")
    route_json=gpd.read_file(ROOT/"data/release/changwon_transit_route.geojson")
    bridge=pd.read_parquet(ROOT/"data/release/changwon_transit_route_stop.parquet")
    assert len(route_pq)==len(route_json)==170 and len(bridge)==17026
    assert route_pq.geometry.notna().all() and route_pq.geometry.is_valid.all()
    assert not bridge.duplicated(["route_id","stop_order"]).any()
    non_monotonic=sum(not g.stop_order.dropna().is_monotonic_increasing for _,g in bridge.groupby("route_id",sort=False))
    with get_engine().connect() as c:
        counts=c.execute(text("SELECT (SELECT count(*) FROM transport.dim_transit_route), (SELECT count(*) FROM transport.bridge_route_stop)")).one()
        invalid=c.execute(text("SELECT count(*) FROM transport.dim_transit_route WHERE geometry IS NULL OR NOT ST_IsValid(geometry) OR ST_SRID(geometry)<>4326")).scalar_one()
        gist=c.execute(text("SELECT count(*) FROM pg_indexes WHERE schemaname='transport' AND tablename='dim_transit_route' AND indexdef ILIKE '%gist%' ")).scalar_one()
        fks=c.execute(text("SELECT count(*) FROM information_schema.table_constraints WHERE table_schema='transport' AND table_name='bridge_route_stop' AND constraint_type='FOREIGN KEY'")).scalar_one()
        unique=c.execute(text("SELECT count(*) FROM information_schema.table_constraints WHERE table_schema='transport' AND table_name='bridge_route_stop' AND constraint_type='UNIQUE'")).scalar_one()
    result={"release_routes":len(route_pq),"unique_route_no":int(route_pq.route_no.nunique()),"non_monotonic_route_order":non_monotonic,"release_route_stops":len(bridge),"postgis_routes":counts[0],"postgis_route_stops":counts[1],"invalid_geometry":invalid,"gist_indexes":gist,"foreign_keys":fks,"unique_constraints":unique,"status":"PASS"}
    print(json.dumps(result,ensure_ascii=False,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
