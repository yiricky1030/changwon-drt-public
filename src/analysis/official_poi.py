"""Official healthcare and administration POI collectors.

Credentials are read by the caller and are never persisted in outputs.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pandas as pd
import requests
from shapely.geometry import Point

HIRA_ENDPOINT = "https://apis.data.go.kr/B551182/hospInfoServicev2/getHospBasisList"
CHANGWON_SGGU_CODES = ("380701", "380702", "380703", "380704", "380705")
ADMIN_FILENAME = "administration.csv"


def _text(item: ET.Element, name: str) -> str | None:
    value = item.findtext(name)
    return value.strip() if value and value.strip() else None


def fetch_hira_healthcare(service_key: str, timeout: int = 90) -> tuple[gpd.GeoDataFrame, dict]:
    """Fetch all HIRA rows for Gyeongsangnam-do and retain Changwon addresses."""
    if not service_key:
        raise RuntimeError("HIRA_SERVICE_KEY (or DATA_GO_KR_SERVICE_KEY/TAGO_SERVICE_KEY) is required")
    rows: list[dict] = []
    def fetch_district(sggu_code: str) -> ET.Element:
        response = requests.get(HIRA_ENDPOINT, params={"serviceKey": service_key, "pageNo": 1,
            "numOfRows": 1000, "sidoCd": "380000", "sgguCd": sggu_code}, timeout=timeout)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        result_code = root.findtext(".//resultCode")
        if result_code not in ("00", "0"):
            raise RuntimeError(f"HIRA API returned resultCode={result_code}")
        return root

    with ThreadPoolExecutor(max_workers=len(CHANGWON_SGGU_CODES)) as executor:
        roots = list(executor.map(fetch_district, CHANGWON_SGGU_CODES))
    for root in roots:
            for item in root.findall(".//item"):
                address = _text(item, "addr") or ""
                x, y = _text(item, "XPos"), _text(item, "YPos")
                if "창원시" not in address or not x or not y:
                    continue
                try:
                    longitude, latitude = float(x), float(y)
                except (TypeError, ValueError):
                    continue
                rows.append({"poi_id": "hira:" + str(_text(item, "ykiho")), "poi_name": _text(item, "yadmNm"),
                "poi_category": "HEALTHCARE", "poi_subcategory": _text(item, "clCdNm") or "의료기관",
                "source_name": "건강보험심사평가원_병원정보서비스", "source_institution": "건강보험심사평가원",
                "source_record_id": _text(item, "ykiho"), "source_reference_date": None,
                "source_download_date": date.today().isoformat(), "address": address, "road_address": address,
                "longitude": longitude, "latitude": latitude, "geometry": Point(longitude, latitude),
                    "source_quality_flag": "OK", "coordinate_source": "HIRA"})
    frame = gpd.GeoDataFrame(rows, columns=["poi_id", "poi_name", "poi_category", "poi_subcategory",
        "source_name", "source_institution", "source_record_id", "source_reference_date",
        "source_download_date", "address", "road_address", "longitude", "latitude", "geometry",
        "source_quality_flag", "coordinate_source"], geometry="geometry", crs=4326)
    return frame, {"usable": True, "http_status": 200, "result_code": "00", "collected_count": len(frame)}


def _normalise_admin_name(value: object) -> str:
    return re.sub(r"(행정복지센터|주민센터|읍사무소|면사무소|동사무소|출장소|\s)", "", str(value))


def load_official_administration(path: Path, osm_admin: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict]:
    """Load Changwon MOIS rows, using OSM only to resolve coordinates by name."""
    source = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    source = source[source["시군구"].str.contains("창원시", na=False)].copy()
    lookup = osm_admin.copy()
    lookup["_key"] = lookup.poi_name.map(_normalise_admin_name)
    lookup = lookup.drop_duplicates("_key").set_index("_key")
    rows, unmatched = [], []
    for _, item in source.iterrows():
        key = _normalise_admin_name(item["읍면동"])
        if key in lookup.index:
            coordinate = lookup.loc[key]
            lon, lat, coordinate_source = float(coordinate.longitude), float(coordinate.latitude), "OpenStreetMap name match"
        else:
            try:
                lat, lon = ox.geocode(item["주소"])
                coordinate_source = "OpenStreetMap Nominatim address geocode"
            except Exception:
                unmatched.append(item.to_dict())
                continue
        rows.append({"poi_id": "mois-admin:" + str(item["연번"]), "poi_name": item["읍면동"].strip(),
            "poi_category": "ADMINISTRATION", "poi_subcategory": "행정복지센터/주민센터",
            "source_name": "행정안전부_읍면동 하부행정기관 현황", "source_institution": "행정안전부",
            "source_record_id": item["연번"], "source_reference_date": "2025-12-31",
            "source_download_date": date.today().isoformat(), "address": item["주소"], "road_address": item["주소"],
            "longitude": lon, "latitude": lat, "geometry": Point(lon, lat), "source_quality_flag": "OK",
            "coordinate_source": coordinate_source})
    frame = gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)
    return frame, {"usable": bool(len(frame)), "official_rows": len(source), "collected_count": len(frame),
        "coordinate_match_count": len(frame), "unmatched_count": len(unmatched)}
