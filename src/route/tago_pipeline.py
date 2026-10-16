"""Collect and build verified TAGO stop-sequence route geometry."""
from __future__ import annotations

import json
import time
from urllib.parse import urlencode
from urllib.request import urlopen

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

BASE_URL = "https://apis.data.go.kr/1613000/BusRouteInfoInqireService"


def parse_response(payload: dict) -> tuple[list[dict], int | None, dict]:
    response = payload.get("response", {})
    header = response.get("header", {})
    if str(header.get("resultCode")) != "00":
        raise ValueError(f"TAGO response failed: {header.get('resultCode')} {header.get('resultMsg')}")
    body = response.get("body", {})
    raw = body.get("items", {})
    items = raw.get("item", []) if isinstance(raw, dict) else []
    if isinstance(items, dict):
        items = [items]
    return list(items or []), int(body["totalCount"]) if body.get("totalCount") is not None else None, header


def request_page(service_key: str, operation: str, params: dict, page: int, rows: int = 1000) -> dict:
    query = urlencode({"serviceKey": service_key, "_type": "json", "pageNo": page, "numOfRows": rows, **params})
    for attempt in range(10):
        with urlopen(f"{BASE_URL}/{operation}?{query}", timeout=60) as response:
            payload = json.load(response)
        header = payload.get("response", {}).get("header", {})
        if str(header.get("resultCode")) != "99" or "세션" not in str(header.get("resultMsg")):
            time.sleep(0.15)
            return payload
        time.sleep(min(30, 2 ** attempt))
    return payload


def fetch_paginated(service_key: str, operation: str, params: dict | None = None, rows: int = 1000) -> tuple[list[dict], list[dict]]:
    records, pages, page = [], [], 1
    while True:
        payload = request_page(service_key, operation, params or {}, page, rows)
        items, total, _ = parse_response(payload)
        pages.append(payload)
        records.extend(items)
        if total is None or len(records) >= total or not items:
            break
        page += 1
    return records, pages


def validate_city(cities: list[dict], city_code: str = "38010", city_name: str = "창원시") -> dict:
    matches = [x for x in cities if str(x.get("citycode")) == city_code]
    if len(matches) != 1 or matches[0].get("cityname") != city_name:
        raise ValueError(f"City validation failed for {city_code}/{city_name}")
    return matches[0]


def normalize_routes(records: list[dict]) -> pd.DataFrame:
    aliases = {"routeid":"route_id","routeno":"route_no","routetp":"route_type","startnodenm":"start_node_name","endnodenm":"end_node_name","startvehicletime":"start_vehicle_time","endvehicletime":"end_vehicle_time","intervaltime":"interval_time","intervalsattime":"interval_sat_time","intervalsuntime":"interval_sun_time"}
    frame = pd.DataFrame(records).rename(columns=aliases)
    for column in aliases.values():
        if column not in frame: frame[column] = pd.NA
    for column in aliases.values():
        frame[column] = frame[column].astype("string")
    return frame[list(aliases.values())]


def normalize_route_stops(records: list[dict]) -> pd.DataFrame:
    aliases = {"routeid":"route_id","nodeid":"node_id","nodenm":"node_name","nodeno":"node_no","nodeord":"stop_order","gpslati":"latitude","gpslong":"longitude","updowncd":"up_down_code"}
    frame = pd.DataFrame(records).rename(columns=aliases)
    for column in aliases.values():
        if column not in frame: frame[column] = pd.NA
    frame["route_id"] = frame.route_id.astype("string")
    frame["node_id"] = frame.node_id.astype("string")
    frame["node_no"] = frame.node_no.astype("string")
    frame["node_name"] = frame.node_name.astype("string")
    frame["up_down_code"] = frame.up_down_code.astype("string")
    frame["stop_order"] = pd.to_numeric(frame.stop_order, errors="coerce").astype("Int64")
    frame["latitude"] = pd.to_numeric(frame.latitude, errors="coerce")
    frame["longitude"] = pd.to_numeric(frame.longitude, errors="coerce")
    return frame[list(aliases.values())]


def match_route_stops(route_stops: pd.DataFrame, master: gpd.GeoDataFrame, threshold_m: float = 50.0) -> pd.DataFrame:
    result = route_stops.copy()
    master_ids = set(master.stop_id.astype(str))
    result["stop_id"] = pd.NA
    result["match_method"] = "UNMATCHED"
    result["match_distance_m"] = pd.NA
    exact = result.node_id.astype(str).isin(master_ids)
    result.loc[exact, "stop_id"] = result.loc[exact, "node_id"].astype(str)
    result.loc[exact, "match_method"] = "DIRECT_ID"
    normalized = result.node_id.astype(str).str.removeprefix("CWB")
    official = ~exact & normalized.isin(master_ids)
    result.loc[official, "stop_id"] = normalized[official]
    result.loc[official, "match_method"] = "OFFICIAL_PREFIX_ID"

    candidates = result.index[(result.match_method == "UNMATCHED") & result.longitude.notna() & result.latitude.notna()]
    if len(candidates):
        mg = master[["stop_id", "geometry"]].to_crs(5179).reset_index(drop=True)
        rg = gpd.GeoDataFrame(result.loc[candidates, ["longitude", "latitude"]], geometry=gpd.points_from_xy(result.loc[candidates, "longitude"], result.loc[candidates, "latitude"], crs=4326), crs=4326).to_crs(5179)
        for idx, point in rg.geometry.items():
            nearby = list(mg.sindex.query(point.buffer(threshold_m), predicate="intersects"))
            distances = sorted((float(point.distance(mg.geometry.iloc[i])), i) for i in nearby if point.distance(mg.geometry.iloc[i]) <= threshold_m)
            if len(distances) == 1:
                distance, i = distances[0]
                result.at[idx, "stop_id"] = str(mg.stop_id.iloc[i])
                result.at[idx, "match_method"] = "COORDINATE_FALLBACK"
                result.at[idx, "match_distance_m"] = distance
            elif len(distances) > 1:
                result.at[idx, "match_method"] = "AMBIGUOUS_COORDINATE"
    result["is_matched"] = result.stop_id.notna()
    return result


def build_route_geometry(routes: pd.DataFrame, stops: pd.DataFrame, retrieved_at: str, dataset_version: str) -> gpd.GeoDataFrame:
    output = []
    for route in routes.itertuples(index=False):
        group = stops[stops.route_id == route.route_id].sort_values("stop_order", kind="stable")
        valid = group[group.stop_order.notna() & group.longitude.notna() & group.latitude.notna()]
        geometry = LineString(zip(valid.longitude, valid.latitude)) if len(valid) >= 2 else None
        output.append({**route._asdict(), "stop_count": len(group), "valid_stop_count": len(valid), "geometry_method": "STOP_SEQUENCE_DERIVED" if geometry else None, "geometry": geometry, "source": "국토교통부 TAGO 버스노선정보", "retrieved_at": retrieved_at, "dataset_version": dataset_version})
    return gpd.GeoDataFrame(output, geometry="geometry", crs=4326)


def build_quality(routes_raw: list[dict], routes: pd.DataFrame, stops: pd.DataFrame, route_geo: gpd.GeoDataFrame, retrieval_date: str) -> dict:
    dup_order = stops.duplicated(["route_id", "stop_order"], keep=False) & stops.stop_order.notna()
    route_sequence = stops[stops.stop_order.notna()].route_id.nunique()
    invalid = int((route_geo.geometry.notna() & ~route_geo.geometry.is_valid).sum())
    return {"routes_total": len(routes), "routes_unique": int(routes.route_id.nunique()), "routes_with_stop_sequence": int(route_sequence), "routes_geometry_created": int(route_geo.geometry.notna().sum()), "routes_failed": int(route_geo.geometry.isna().sum()), "duplicate_route_id": int(routes.route_id.duplicated().sum()), "null_route_id": int(routes.route_id.isna().sum()), "route_type_distribution": routes.route_type.fillna("NULL").value_counts().to_dict(), "duplicate_route_stop_order": int(dup_order.sum()), "null_stop_order": int(stops.stop_order.isna().sum()), "duplicate_node_within_route": int(stops.duplicated(["route_id", "node_id"], keep=False).sum()), "routes_with_less_than_2_valid_stops": int((route_geo.valid_stop_count < 2).sum()), "total_route_stop_relations": len(stops), "direct_stop_id_matches": int((stops.match_method == "DIRECT_ID").sum()), "official_prefix_id_matches": int((stops.match_method == "OFFICIAL_PREFIX_ID").sum()), "fallback_matches": int((stops.match_method == "COORDINATE_FALLBACK").sum()), "unmatched_stops": int((stops.match_method == "UNMATCHED").sum()), "ambiguous_matches": int((stops.match_method == "AMBIGUOUS_COORDINATE").sum()), "missing_stop_coordinates": int((stops.longitude.isna() | stops.latitude.isna()).sum()), "invalid_geometry": invalid, "null_geometry": int(route_geo.geometry.isna().sum()), "geometry_method_distribution": route_geo.geometry_method.fillna("NULL").value_counts().to_dict(), "source": "국토교통부 TAGO 버스노선정보", "retrieval_date": retrieval_date, "status": "PASS" if not invalid and routes.route_id.notna().all() and not routes.route_id.duplicated().any() else "FAIL"}
