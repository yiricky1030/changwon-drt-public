import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from src.route.tago_pipeline import parse_response, fetch_paginated, normalize_route_stops, build_route_geometry, match_route_stops

def response(items,total=1): return {"response":{"header":{"resultCode":"00","resultMsg":"NORMAL SERVICE."},"body":{"items":{"item":items},"totalCount":total}}}
def test_parser_single_item():
    items,total,header=parse_response(response({"routeid":"R1"})); assert items==[{"routeid":"R1"}] and total==1 and header["resultCode"]=="00"
def test_pagination(monkeypatch):
    pages=[response([{"x":1}],2),response([{"x":2}],2)]
    monkeypatch.setattr("src.route.tago_pipeline.request_page",lambda *a,**k:pages.pop(0))
    items,_=fetch_paginated("secret","op",rows=1); assert [x["x"] for x in items]==[1,2]
def test_order_geometry_and_missing_coordinates():
    routes=pd.DataFrame([{"route_id":"R1","route_no":"1","route_type":"일반","start_node_name":"A","end_node_name":"B","start_vehicle_time":None,"end_vehicle_time":None,"interval_time":None,"interval_sat_time":None,"interval_sun_time":None}])
    stops=normalize_route_stops([{"routeid":"R1","nodeid":"N2","nodeord":2,"gpslong":128.2,"gpslati":35.2},{"routeid":"R1","nodeid":"N1","nodeord":1,"gpslong":128.1,"gpslati":35.1},{"routeid":"R1","nodeid":"N3","nodeord":3}])
    geo=build_route_geometry(routes,stops,"2026-08-30T00:00:00+00:00","test")
    assert list(geo.geometry.iloc[0].coords)==[(128.1,35.1),(128.2,35.2)] and geo.valid_stop_count.iloc[0]==2
def test_direct_prefix_unmatched_and_ambiguous():
    master=gpd.GeoDataFrame({"stop_id":["123","999"],"geometry":[Point(128,35),Point(129,36)]},crs=4326)
    stops=normalize_route_stops([{"routeid":"R","nodeid":"123","nodeord":1},{"routeid":"R","nodeid":"CWB999","nodeord":2},{"routeid":"R","nodeid":"X","nodeord":3}])
    out=match_route_stops(stops,master); assert out.match_method.tolist()==["DIRECT_ID","OFFICIAL_PREFIX_ID","UNMATCHED"]
def test_duplicate_order_detection_input():
    stops=normalize_route_stops([{"routeid":"R","nodeid":"A","nodeord":1},{"routeid":"R","nodeid":"B","nodeord":1}])
    assert stops.duplicated(["route_id","stop_order"],keep=False).sum()==2
