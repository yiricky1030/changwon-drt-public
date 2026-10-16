"""Collect official TAGO per-route schedule fields (first/last bus, weekday/Sat/Sun interval).

Source: 국토교통부_(TAGO)_버스노선정보 OpenAPI, operation getRouteInfoIem
(data.go.kr dataset 15098529). Only canonical route_id values already present
in data/release/changwon_transit_route.parquet are queried (official exact ID
match; no name-based matching). Raw responses are preserved verbatim under
data/raw/transit_service_frequency/. This script does not modify the existing
canonical changwon_transit_route.parquet; it writes a new, separate table.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import geopandas as gpd
import pandas as pd
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE_URL = "https://apis.data.go.kr/1613000/BusRouteInfoInqireService"
CITY_CODE = "38010"
CITY_NAME = "창원시"
CANDIDATE_OPERATIONS = ["getRouteInfoIem", "getRouteInfoItem"]


def request_item(service_key: str, operation: str, route_id: str) -> dict:
    query = urlencode({"serviceKey": service_key, "_type": "json", "pageNo": 1, "numOfRows": 10, "cityCode": CITY_CODE, "routeId": route_id})
    url = f"{BASE_URL}/{operation}?{query}"
    for attempt in range(5):
        try:
            with urlopen(url, timeout=30) as response:
                return json.load(response)
        except Exception as exc:  # network/HTTP error; retry with backoff
            if attempt == 4:
                return {"_fetch_error": str(exc), "_operation": operation, "_route_id": route_id}
            time.sleep(min(10, 2 ** attempt))
    return {}


def parse_item(payload: dict) -> dict:
    response = payload.get("response", {})
    header = response.get("header", {})
    result_code = str(header.get("resultCode", ""))
    body = response.get("body", {})
    raw_items = body.get("items", "") if isinstance(body, dict) else ""
    item = {}
    if isinstance(raw_items, dict):
        inner = raw_items.get("item", {})
        item = inner[0] if isinstance(inner, list) and inner else (inner if isinstance(inner, dict) else {})
    return {"resultCode": result_code, "resultMsg": header.get("resultMsg", ""), **item}


def probe_operation(service_key: str, sample_route_id: str) -> str:
    for operation in CANDIDATE_OPERATIONS:
        payload = request_item(service_key, operation, sample_route_id)
        parsed = parse_item(payload)
        if parsed.get("resultCode") == "00":
            return operation
    raise SystemExit(f"No working getRouteInfoI(t)em operation found among {CANDIDATE_OPERATIONS}; last resultCode={parsed.get('resultCode')} msg={parsed.get('resultMsg')}")


def to_minutes(value) -> float | None:
    if value is None or value is pd.NA or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_hhmm_minutes_since_midnight(value) -> int | None:
    text = str(value) if value is not None else ""
    if not text or not text.isdigit() or len(text) not in (3, 4):
        return None
    text = text.zfill(4)
    hour, minute = int(text[:2]), int(text[2:])
    return hour * 60 + minute


def main() -> int:
    load_dotenv(ROOT / ".env")
    key = os.environ.get("TAGO_SERVICE_KEY", "")
    if not key:
        raise SystemExit("TAGO_SERVICE_KEY is missing in .env")

    routes = gpd.read_parquet(ROOT / "data/release/changwon_transit_route.parquet")
    route_ids = sorted(routes.route_id.dropna().astype(str).unique())
    if not route_ids:
        raise SystemExit("No canonical route_id values found in changwon_transit_route.parquet")

    operation = probe_operation(key, route_ids[0])

    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%d")
    retrieved_at = now.isoformat()

    raw_dir = ROOT / "data/raw/transit_service_frequency"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"tago_route_schedule_{stamp}.json"

    records = []
    raw_payloads = []
    for number, route_id in enumerate(route_ids, 1):
        payload = request_item(key, operation, route_id)
        raw_payloads.append({"route_id": route_id, "payload": payload})
        parsed = parse_item(payload)
        records.append({
            "route_id": route_id,
            "result_code": parsed.get("resultCode"),
            "result_msg": parsed.get("resultMsg"),
            "route_no": parsed.get("routeno"),
            "route_type": parsed.get("routetp"),
            "start_node_name": parsed.get("startnodenm"),
            "end_node_name": parsed.get("endnodenm"),
            "start_vehicle_time": parsed.get("startvehicletime"),
            "end_vehicle_time": parsed.get("endvehicletime"),
            "interval_weekday_min": to_minutes(parsed.get("intervaltime")),
            "interval_sat_min": to_minutes(parsed.get("intervalsattime")),
            "interval_sun_min": to_minutes(parsed.get("intervalsuntime")),
        })
        if number % 25 == 0 or number == len(route_ids):
            print(f"Collected route schedule: {number}/{len(route_ids)}")
        time.sleep(0.12)

    raw_path.write_text(json.dumps({"operation": operation, "city_code": CITY_CODE, "city_name": CITY_NAME, "retrieved_at": retrieved_at, "route_count": len(route_ids), "responses": raw_payloads}, ensure_ascii=False), encoding="utf-8")
    sha256 = hashlib.sha256(raw_path.read_bytes()).hexdigest()

    frame = pd.DataFrame(records)
    for column in ["route_id", "result_code", "result_msg", "route_no", "route_type", "start_node_name", "end_node_name", "start_vehicle_time", "end_vehicle_time"]:
        frame[column] = frame[column].astype("string")
    frame["start_min"] = frame.start_vehicle_time.map(to_hhmm_minutes_since_midnight)
    frame["end_min"] = frame.end_vehicle_time.map(to_hhmm_minutes_since_midnight)
    span = frame.end_min - frame.start_min
    frame["service_span_minutes"] = span.where(span >= 0, span + 24 * 60)
    frame["estimated_scheduled_trips_weekday"] = (frame.service_span_minutes / frame.interval_weekday_min).round(1)
    frame["estimated_scheduled_trips_saturday"] = (frame.service_span_minutes / frame.interval_sat_min).round(1)
    frame["estimated_scheduled_trips_sunday"] = (frame.service_span_minutes / frame.interval_sun_min).round(1)
    frame["source"] = "국토교통부 TAGO 버스노선정보(getRouteInfoIem)"
    frame["retrieved_at"] = retrieved_at
    frame["dataset_version"] = "1.0.0"

    processed_dir = ROOT / "data/processed/transit_service_frequency"
    processed_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(processed_dir / f"tago_route_schedule_{stamp}.parquet", index=False)

    release_path = ROOT / "data/release/changwon_transit_route_schedule.parquet"
    frame.to_parquet(release_path, index=False)

    qa = {
        "operation_used": operation,
        "routes_queried": len(route_ids),
        "result_code_distribution": frame.result_code.value_counts(dropna=False).to_dict(),
        "routes_with_interval_weekday": int(frame.interval_weekday_min.notna().sum()),
        "routes_with_interval_sat": int(frame.interval_sat_min.notna().sum()),
        "routes_with_interval_sun": int(frame.interval_sun_min.notna().sum()),
        "routes_with_start_end_time": int((frame.start_min.notna() & frame.end_min.notna()).sum()),
        "routes_missing_all_schedule_fields": int((frame.interval_weekday_min.isna() & frame.start_min.isna()).sum()),
        "raw_file": str(raw_path.relative_to(ROOT)),
        "raw_sha256": sha256,
        "retrieval_date": stamp,
        "canonical_route_parquet_modified": False,
    }
    (ROOT / "reports/transit_service_frequency_quality_report.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")

    metadata = {
        "dataset": "changwon_transit_route_schedule",
        "source": "국토교통부 TAGO 버스노선정보 OpenAPI (data.go.kr, dataset ID 15098529)",
        "operation": operation,
        "official_url": "https://www.data.go.kr/data/15098529/openapi.do",
        "authentication": "data.go.kr service key (TAGO_SERVICE_KEY, stored in .env, not committed)",
        "license": "공공누리 제1유형 (data.go.kr 표준 이용허락, 출처표시 조건)",
        "cityCode": CITY_CODE,
        "cityName": CITY_NAME,
        "retrieval_date": stamp,
        "retrieved_at": retrieved_at,
        "spatial_unit": "route (route_id, official TAGO identifier, exact-match to canonical changwon_transit_route.parquet)",
        "temporal_unit": "노선당 1회 스냅샷: 평일/토요일/일요일 배차간격(분), 첫차/막차 시각",
        "raw_path": str(raw_path.relative_to(ROOT)),
        "raw_sha256": sha256,
        "processed_path": str((processed_dir / f"tago_route_schedule_{stamp}.parquet").relative_to(ROOT)),
        "release_path": "data/release/changwon_transit_route_schedule.parquet",
        "row_count": len(frame),
        "fields": ["route_id", "route_no", "route_type", "start_vehicle_time", "end_vehicle_time", "interval_weekday_min", "interval_sat_min", "interval_sun_min", "service_span_minutes", "estimated_scheduled_trips_weekday", "estimated_scheduled_trips_saturday", "estimated_scheduled_trips_sunday"],
        "used_for": "Feedback 9 — 2026-09 TAGO 배차 스냅샷 기반 공급 빈도 추정",
        "matching_method": "OFFICIAL_ID_EXACT (route_id는 기존 canonical changwon_transit_route.parquet에서 그대로 조회한 값이며 이름 매칭을 사용하지 않음)",
        "limitations": [
            "getRouteInfoIem은 노선당 단일 대표 배차간격(분)만 제공하며 시간대별(예: 출퇴근 vs 한산) 배차 변화는 제공하지 않는다.",
            "estimated_scheduled_trips_* 는 (첫차~막차 운행시간)/(배차간격)의 산술 추정값이며, 실제 관측된 운행 횟수가 아니다. 배차간격이 시간대별로 균일하다고 가정한 근사치다.",
            "2025-08 이용 관측보다 약 13개월 뒤의 스냅샷이므로 2025년 당시 공급량으로 표현할 수 없다.",
            "일부 노선은 interval 필드가 공란(NULL)일 수 있으며 이 경우 추정 운행횟수를 계산하지 않는다(0으로 치환하지 않음).",
            "이 표는 기존 changwon_transit_route.parquet과 별도의 신규 release 파일이며 canonical Parquet은 수정하지 않았다.",
        ],
    }
    (ROOT / "metadata/changwon_transit_service_frequency.yaml").write_text(yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False), encoding="utf-8")

    print(json.dumps(qa, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
