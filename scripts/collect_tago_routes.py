"""Collect TAGO routes, create release/QA/metadata, and optionally load PostGIS."""
from __future__ import annotations
import argparse, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
import geopandas as gpd
import pandas as pd
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DLL_DIRECTORY_HANDLES=[]
if os.name=="nt" and hasattr(os,"add_dll_directory"):
    qgis_bin=Path(sys.base_prefix).parents[1]/"bin"
    if qgis_bin.exists(): DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(qgis_bin)))
from src.database.connection import get_engine
from src.route.tago_pipeline import *

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--skip-db",action="store_true"); parser.add_argument("--db-only",action="store_true"); args=parser.parse_args()
    load_dotenv(ROOT/".env"); key=os.environ.get("TAGO_SERVICE_KEY","")
    if not key: raise SystemExit("TAGO_SERVICE_KEY is missing")
    if args.db_only:
        route_geo=gpd.read_parquet(ROOT/"data/release/changwon_transit_route.parquet")
        bridge=pd.read_parquet(ROOT/"data/release/changwon_transit_route_stop.parquet")
        engine=get_engine(); sql=(ROOT/"sql/04_create_route_tables.sql").read_text(encoding="utf-8")
        db_bridge=bridge[["route_id","stop_id","node_id","stop_name","stop_order","match_method","match_distance_m","is_matched","retrieved_at","dataset_version"]].copy()
        with engine.begin() as c:
            c.exec_driver_sql(sql); c.exec_driver_sql("DELETE FROM transport.bridge_route_stop"); c.exec_driver_sql("DELETE FROM transport.dim_transit_route")
            route_geo.to_postgis("dim_transit_route",c,schema="transport",if_exists="append",index=False)
            db_bridge.to_sql("bridge_route_stop",c,schema="transport",if_exists="append",index=False,method="multi",chunksize=2000)
        print(json.dumps({"postgis":{"routes":len(route_geo),"route_stops":len(bridge)}},ensure_ascii=False)); return 0
    now=datetime.now(timezone.utc); stamp=now.strftime("%Y%m%d"); retrieved=now.isoformat(); version="1.2.0"
    cities,city_pages=fetch_paginated(key,"getCtyCodeList",rows=1000); city=validate_city(cities)
    route_records,route_pages=fetch_paginated(key,"getRouteNoList",{"cityCode":"38010"},rows=1000); routes=normalize_routes(route_records)
    all_stops=[]; stop_pages=[]
    for number,route_id in enumerate(routes.route_id,1):
        records,pages=fetch_paginated(key,"getRouteAcctoThrghSttnList",{"cityCode":"38010","routeId":route_id},rows=1000)
        all_stops.extend(records); stop_pages.extend(pages)
        if number%25==0 or number==len(routes): print(f"Collected route stops: {number}/{len(routes)}")
    raw=ROOT/"data/raw/route"; raw.mkdir(parents=True,exist_ok=True)
    (raw/f"tago_bus_routes_{stamp}.json").write_text(json.dumps({"city":city,"pages":route_pages},ensure_ascii=False,indent=2),encoding="utf-8")
    (raw/f"tago_route_stops_{stamp}.json").write_text(json.dumps({"pages":stop_pages},ensure_ascii=False),encoding="utf-8")
    stops=normalize_route_stops(all_stops)
    master=gpd.read_parquet(ROOT/"data/release/dim_transit_stop.parquet")
    stops=match_route_stops(stops,master)
    route_geo=build_route_geometry(routes,stops,retrieved,version)
    qa=build_quality(route_records,routes,stops,route_geo,stamp)
    stops.to_parquet(raw/f"tago_route_stops_{stamp}.parquet",index=False)
    release=ROOT/"data/release"; route_geo.to_parquet(release/"changwon_transit_route.parquet",index=False); route_geo.to_file(release/"changwon_transit_route.geojson",driver="GeoJSON")
    bridge=stops.rename(columns={"node_name":"stop_name"}); bridge["retrieved_at"]=retrieved; bridge["dataset_version"]=version
    bridge.to_parquet(release/"changwon_transit_route_stop.parquet",index=False)
    (ROOT/"reports/transit_route_quality_report.json").write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding="utf-8")
    metadata={"dataset":"changwon_transit_route","source":"국토교통부 TAGO 버스노선정보","cityCode":"38010","cityName":"창원시","retrieval_date":stamp,"crs":"EPSG:4326","geometry_method":"STOP_SEQUENCE_DERIVED","stop_matching_methods":["DIRECT_ID","OFFICIAL_PREFIX_ID","COORDINATE_FALLBACK_50M"],"limitations":["공식 경유정류소 순서를 직선으로 연결한 derived geometry이며, 실제 도로 주행 궤적을 의미하지 않는다.","이름 단독 정합, 도로망 snap, spline, 임의 보간을 사용하지 않는다."]}
    (ROOT/"metadata/changwon_transit_route.yaml").write_text(yaml.safe_dump(metadata,allow_unicode=True,sort_keys=False),encoding="utf-8")
    if not args.skip_db:
        engine=get_engine(); sql=(ROOT/"sql/04_create_route_tables.sql").read_text(encoding="utf-8")
        db_routes=route_geo.copy(); db_bridge=bridge[["route_id","stop_id","node_id","stop_name","stop_order","match_method","match_distance_m","is_matched","retrieved_at","dataset_version"]].copy()
        with engine.begin() as c:
            c.exec_driver_sql(sql); c.exec_driver_sql("DELETE FROM transport.bridge_route_stop"); c.exec_driver_sql("DELETE FROM transport.dim_transit_route")
            db_routes.to_postgis("dim_transit_route",c,schema="transport",if_exists="append",index=False)
            db_bridge.to_sql("bridge_route_stop",c,schema="transport",if_exists="append",index=False,method="multi",chunksize=2000)
    print(json.dumps({"auth":{"resultCode":"00"},"city":city,"qa":qa},ensure_ascii=False,indent=2))
    return 0 if qa["status"]=="PASS" else 1
if __name__=="__main__": raise SystemExit(main())
