"""Priority POI Context Foundation for robust_clean network Anchor stops.

No POI score, destination score, or final Anchor score is produced here.
The active product contains only official HIRA healthcare and MOIS
administration records. Legacy OSM collection helpers remain for reproducible
cache inspection, but are not part of the active product pipeline.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.strtree import STRtree


CATCHMENT_RADII = (500, 800, 1200)
PRIORITY_CATEGORIES = ("HEALTHCARE", "ADMINISTRATION")

CATEGORY_TAGS = {
    "HEALTHCARE": {"amenity": ["hospital", "clinic", "doctors"], "healthcare": ["hospital", "clinic", "centre"]},
    "ADMINISTRATION": {"amenity": ["townhall"], "office": ["government"]},
}

ADMINISTRATION_NAME_HINTS = ("행정복지센터", "주민센터", "동사무소", "읍사무소", "면사무소", "시청", "구청", "도청", "민원센터")

SOURCE_AUDIT = {
    "HEALTHCARE": [{
        "category": "HEALTHCARE", "source_institution": "건강보험심사평가원 (HIRA)",
        "dataset_or_api": "병원정보서비스 getHospBasisList (data.go.kr 15001698)",
        "official_public": True, "reference_date": "quarterly (등록 시점 기준)", "download_date": None,
        "coordinate_availability": True, "crs": "EPSG:4326 (WGS84 lat/lon)", "key_id_availability": "요양기관기호 (ykiho)",
        "address_availability": True, "api_credential_requirement": "data.go.kr 개인 인증키 + 서비스별 활용신청 승인 필요",
        "terms_attribution": "공공데이터포털 이용약관", "usable": False,
        "reason": "SOURCE_REQUIRES_CREDENTIAL_OR_APPROVAL — 기존 TAGO_SERVICE_KEY로 apis.data.go.kr/B551182/hospInfoServicev2/getHospBasisList 호출 시 HTTP 403 Forbidden (이 API에 대한 활용신청 승인 없음). 새 승인 없이는 수집 불가."},
    ], "ADMINISTRATION": [{
        "category": "ADMINISTRATION", "source_institution": "행정안전부",
        "dataset_or_api": "읍면동 하부행정기관 현황 (data.go.kr 15059715, 파일데이터)",
        "official_public": True, "reference_date": "2025-12-31", "download_date": None,
        "coordinate_availability": "unknown (주소만 확인, 파일 미다운로드)", "crs": None, "key_id_availability": "unknown",
        "address_availability": True, "api_credential_requirement": "data.go.kr 로그인 세션 기반 파일 다운로드(동적 CSRF/PK 파라미터), serviceKey 방식 아님",
        "terms_attribution": "공공데이터포털 이용약관", "usable": False,
        "reason": "SOURCE_REQUIRES_CREDENTIAL_OR_APPROVAL — fileData.do 다운로드 버튼이 JS로 publicDataPk/publicDataDetailPk를 동적 조합하는 세션 기반 흐름이라 무인증 HTTP 요청으로 안정적으로 재현할 수 없음. 로그인 세션이 있는 사용자 상호작용이 필요."},
    ]}


def changwon_extent(boundary: gpd.GeoDataFrame, buffer_m: int = 2000) -> tuple[Any, Any]:
    official = boundary.to_crs(5179).geometry.union_all()
    buffered_metric = official.buffer(buffer_m)
    official_4326 = gpd.GeoSeries([official], crs=5179).to_crs(4326).iloc[0]
    buffered_4326 = gpd.GeoSeries([buffered_metric], crs=5179).to_crs(4326).iloc[0]
    return official_4326, buffered_4326


def fetch_osm_priority_poi(extent_polygon) -> gpd.GeoDataFrame:
    frames = []
    for category, tags in CATEGORY_TAGS.items():
        try:
            raw = ox.features.features_from_polygon(extent_polygon, tags)
        except Exception:
            continue
        if raw.empty:
            continue
        raw = raw.reset_index()
        raw["poi_category"] = category
        frames.append(raw)
    if not frames:
        return gpd.GeoDataFrame(columns=["element", "id", "geometry", "poi_category"], geometry="geometry", crs=4326)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return gpd.GeoDataFrame(combined, geometry="geometry", crs=4326)


def _subcategory(row: pd.Series) -> str:
    if row.poi_category == "HEALTHCARE":
        amenity = row.get("amenity")
        if amenity == "hospital":
            return "종합병원/병원"
        if amenity in ("clinic",) or row.get("healthcare") == "clinic":
            return "의원"
        if row.get("healthcare") == "centre":
            return "보건시설"
        return "의료기관"
    if row.poi_category == "ADMINISTRATION":
        name = str(row.get("name") or "")
        if any(hint in name for hint in ADMINISTRATION_NAME_HINTS):
            return "행정복지센터/주민센터"
        return "공공행정거점"
    return row.poi_category


def build_poi_master(raw: gpd.GeoDataFrame, official_boundary, buffered_extent, download_date: str) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    if raw.empty:
        return raw, {"raw_count": 0, "cleaned_count": 0}
    work = raw.copy()
    work["geometry"] = work.geometry.apply(lambda geom: geom.centroid if geom.geom_type != "Point" else geom)
    work["source_record_id"] = work.element.astype(str) + "/" + work.id.astype(str)
    work["poi_name"] = work.get("name")
    name_text = work.poi_name.fillna("")
    is_administration_relevant = name_text.str.contains("|".join(ADMINISTRATION_NAME_HINTS))
    work = work.loc[work.poi_category.ne("ADMINISTRATION") | is_administration_relevant].reset_index(drop=True)
    work["poi_subcategory"] = work.apply(_subcategory, axis=1)
    work["source_name"] = "OpenStreetMap"
    work["source_institution"] = "OpenStreetMap contributors"
    work["source_reference_date"] = None
    work["source_download_date"] = download_date
    work["address"] = work.get("addr:full")
    housenumber = work.get("addr:housenumber")
    street = work.get("addr:street")
    if housenumber is not None and street is not None:
        work["road_address"] = np.where(street.notna() & housenumber.notna(), street.fillna("") + " " + housenumber.fillna(""), None)
    else:
        work["road_address"] = None
    work["longitude"] = work.geometry.x
    work["latitude"] = work.geometry.y

    coord_null = work.longitude.isna() | work.latitude.isna()
    valid_range = work.longitude.between(-180, 180) & work.latitude.between(-90, 90)
    valid_geometry = work.geometry.is_valid & work.geometry.notna()
    duplicate_source_id = work.source_record_id.duplicated(keep=False)
    coordinate_key = work.geometry.to_wkb()
    exact_duplicate_coordinate = coordinate_key.duplicated(keep=False)
    name_key = work.poi_name.fillna("").str.strip()
    same_name_duplicate = name_key.ne("") & name_key.duplicated(keep=False)

    quality_flag = np.select(
        [coord_null | ~valid_range | ~valid_geometry, duplicate_source_id, exact_duplicate_coordinate & same_name_duplicate,
         same_name_duplicate], ["GEOMETRY_INVALID", "DUPLICATE_SOURCE_ID", "LIKELY_DUPLICATE", "SAME_NAME_REVIEW"],
        default="OK")
    work["source_quality_flag"] = quality_flag

    inside = work.within(official_boundary)
    inside_buffer = work.within(buffered_extent)
    work["inside_changwon"] = inside
    work["outside_context"] = inside_buffer & ~inside

    keep = valid_range & valid_geometry & ~coord_null & ~duplicate_source_id & (inside | inside_buffer)
    cleaned = work.loc[keep].drop_duplicates(subset="source_record_id").reset_index(drop=True)
    cleaned["poi_id"] = "osm:" + cleaned.source_record_id

    qa = {"raw_count": int(len(work)), "cleaned_count": int(len(cleaned)),
          "geometry_invalid": int((~valid_geometry | ~valid_range | coord_null).sum()),
          "duplicate_source_id": int(duplicate_source_id.sum()),
          "exact_duplicate_coordinate": int(exact_duplicate_coordinate.sum()),
          "same_name_review": int(same_name_duplicate.sum()),
          "inside_changwon": int(cleaned.inside_changwon.sum()), "outside_context": int(cleaned.outside_context.sum()),
          "by_category": cleaned.poi_category.value_counts().to_dict()}
    columns = ["poi_id", "poi_name", "poi_category", "poi_subcategory", "source_name", "source_institution",
               "source_record_id", "source_reference_date", "source_download_date", "address", "road_address",
               "longitude", "latitude", "geometry", "source_quality_flag", "inside_changwon", "outside_context"]
    return gpd.GeoDataFrame(cleaned[columns], geometry="geometry", crs=4326), qa


def snap_poi_to_network(poi: gpd.GeoDataFrame, graph: nx.MultiGraph) -> pd.DataFrame:
    metric = poi.to_crs(5179).reset_index(drop=True)
    node_ids = np.array(list(graph.nodes), dtype=object)
    node_points = np.array([graph.nodes[node]["geometry"] if "geometry" in graph.nodes[node]
                            else None for node in node_ids], dtype=object)
    if any(point is None for point in node_points):
        from shapely.geometry import Point
        node_points = np.array([Point(float(graph.nodes[node]["x"]), float(graph.nodes[node]["y"])) for node in node_ids])
    pairs, distances = STRtree(node_points).query_nearest(metric.geometry.to_numpy(), all_matches=False, return_distance=True)
    nearest = pairs[1]
    return pd.DataFrame({"poi_id": metric.poi_id.to_numpy(), "network_node_id": node_ids[nearest].astype(str),
                         "network_snap_distance_m": distances.astype(float)})


def poi_snap_qa(poi: gpd.GeoDataFrame) -> dict[str, Any]:
    output = {}
    for category, group in poi.groupby("poi_category"):
        distance = group.network_snap_distance_m.dropna()
        output[category] = {"total": int(len(group)), "snap_success": int(distance.notna().sum()) if len(group) else 0,
            "median": float(distance.median()) if len(distance) else None, "p95": float(distance.quantile(.95)) if len(distance) else None,
            "over_100m": int(distance.gt(100).sum()), "over_200m": int(distance.gt(200).sum()),
            "max": float(distance.max()) if len(distance) else None}
    return output


def anchor_poi_context(anchors: pd.DataFrame, poi: gpd.GeoDataFrame, graph: nx.MultiGraph) -> pd.DataFrame:
    node_lookup = {str(node): node for node in graph.nodes}
    poi_by_category: dict[str, pd.DataFrame] = {category: group.reset_index(drop=True) for category, group in poi.groupby("poi_category")}
    node_to_poi: dict[str, dict[Any, list[int]]] = {}
    for category, group in poi_by_category.items():
        mapping: dict[Any, list[int]] = defaultdict(list)
        for index, node_id in enumerate(group.network_node_id):
            if node_id in node_lookup:
                mapping[node_lookup[node_id]].append(index)
        node_to_poi[category] = mapping

    rows = []
    max_radius = max(CATCHMENT_RADII)
    for _, anchor in anchors.iterrows():
        anchor_node = node_lookup.get(anchor.network_node_id)
        row: dict[str, Any] = {"stop_id": anchor.stop_id}
        if anchor_node is None or pd.isna(anchor.snap_distance_m):
            for category in PRIORITY_CATEGORIES:
                key = category.lower()
                row[f"nearest_{key}_walk_m"] = np.nan
                row[f"nearest_{key}_poi_id"] = None
                row[f"nearest_{key}_name"] = None
                for radius in CATCHMENT_RADII:
                    row[f"{key}_within_walk_{radius}m"] = np.nan
            row["nearest_any_priority_poi_walk_m"] = np.nan
            row["poi_category_count_walk_800m"] = np.nan
            rows.append(row)
            continue
        lengths = nx.single_source_dijkstra_path_length(graph, anchor_node, cutoff=max_radius, weight="length")
        best_overall = np.inf
        category_present_800 = 0
        for category in PRIORITY_CATEGORIES:
            key = category.lower()
            group = poi_by_category.get(category)
            mapping = node_to_poi.get(category, {})
            candidates = []
            for node, graph_distance in lengths.items():
                for poi_index in mapping.get(node, []):
                    total = anchor.snap_distance_m + graph_distance + group.loc[poi_index, "network_snap_distance_m"]
                    candidates.append((poi_index, total))
            if group is None or not candidates:
                row[f"nearest_{key}_walk_m"] = np.nan
                row[f"nearest_{key}_poi_id"] = None
                row[f"nearest_{key}_name"] = None
                for radius in CATCHMENT_RADII:
                    row[f"{key}_within_walk_{radius}m"] = 0
                continue
            best_index, best_distance = min(candidates, key=lambda item: item[1])
            row[f"nearest_{key}_walk_m"] = float(best_distance)
            row[f"nearest_{key}_poi_id"] = group.loc[best_index, "poi_id"]
            row[f"nearest_{key}_name"] = group.loc[best_index, "poi_name"]
            for radius in CATCHMENT_RADII:
                row[f"{key}_within_walk_{radius}m"] = int(sum(1 for _, distance in candidates if distance <= radius))
            if row[f"{key}_within_walk_800m"] > 0:
                category_present_800 += 1
            best_overall = min(best_overall, best_distance)
        row["nearest_any_priority_poi_walk_m"] = float(best_overall) if np.isfinite(best_overall) else np.nan
        row["poi_category_count_walk_800m"] = category_present_800
        rows.append(row)
    return pd.DataFrame(rows)


def anchor_poi_nearby_list(anchors: pd.DataFrame, poi: gpd.GeoDataFrame, graph: nx.MultiGraph,
                            radius: int = 1200) -> pd.DataFrame:
    """Long-format (stop_id, poi_id, category, name, walk_m) rows for every POI
    within `radius` network metres of each anchor. Additive helper for map
    display (e.g. Dashboard POI markers); does not change anchor_poi_context's
    aggregate output."""
    node_lookup = {str(node): node for node in graph.nodes}
    poi_by_category: dict[str, pd.DataFrame] = {category: group.reset_index(drop=True) for category, group in poi.groupby("poi_category")}
    node_to_poi: dict[str, dict[Any, list[int]]] = {}
    for category, group in poi_by_category.items():
        mapping: dict[Any, list[int]] = defaultdict(list)
        for index, node_id in enumerate(group.network_node_id):
            if node_id in node_lookup:
                mapping[node_lookup[node_id]].append(index)
        node_to_poi[category] = mapping

    rows = []
    for _, anchor in anchors.iterrows():
        anchor_node = node_lookup.get(anchor.network_node_id)
        if anchor_node is None or pd.isna(anchor.snap_distance_m):
            continue
        lengths = nx.single_source_dijkstra_path_length(graph, anchor_node, cutoff=radius, weight="length")
        for category, group in poi_by_category.items():
            mapping = node_to_poi.get(category, {})
            for node, graph_distance in lengths.items():
                for poi_index in mapping.get(node, []):
                    total = anchor.snap_distance_m + graph_distance + group.loc[poi_index, "network_snap_distance_m"]
                    if total <= radius:
                        rows.append({"stop_id": anchor.stop_id, "poi_id": group.loc[poi_index, "poi_id"],
                                     "poi_category": category, "poi_name": group.loc[poi_index, "poi_name"],
                                     "walk_m": float(total)})
    return pd.DataFrame(rows, columns=["stop_id", "poi_id", "poi_category", "poi_name", "walk_m"])
