"""Official POI reliability classification, primary context, and
PostGIS load for the experimental POI Context Layer. No POI score, no Anchor
score change, no ranking recompute, no release/Dashboard modification."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.drt_anchor_poi_reliability import (
    CATEGORY_RELIABILITY, OFFICIAL_SOURCE_STATUS, PRIMARY_CATEGORIES,
    anchor_primary_context, dong_pattern, poi_with_reliability, primary_context_summary,
)
from src.database.connection import get_engine

REPORTS = ROOT / "reports"


def _sql_type(dtype) -> str:
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    if pd.api.types.is_integer_dtype(dtype):
        return "bigint"
    if pd.api.types.is_float_dtype(dtype):
        return "double precision"
    return "text"


def _write_table(gdf: gpd.GeoDataFrame, table: str, schema: str, connection) -> None:
    """Manual PostGIS load via SQLAlchemy text() execute, bypassing
    geopandas.to_postgis()/pandas.to_sql() which imports stdlib sqlite3 purely
    for an isinstance check — unavailable in this environment's Python (QGIS
    bundled interpreter missing its sqlite3 DLL), unrelated to PostgreSQL."""
    columns = [c for c in gdf.columns if c != gdf.geometry.name]
    types = {c: _sql_type(gdf[c].dtype) for c in columns}
    connection.execute(text(f'DROP TABLE IF EXISTS {schema}.{table}'))
    column_defs = ", ".join(f'"{c}" {types[c]}' for c in columns)
    connection.execute(text(f'CREATE TABLE {schema}.{table} ({column_defs}, geometry geometry(Geometry,4326))'))
    insert_columns = ", ".join(f'"{c}"' for c in columns) + ", geometry"
    placeholders = ", ".join(f':{c}' for c in columns) + ", ST_GeomFromText(:geom, 4326)"
    insert_sql = text(f'INSERT INTO {schema}.{table} ({insert_columns}) VALUES ({placeholders})')
    rows = []
    for _, row in gdf.iterrows():
        params = {c: (None if pd.isna(row[c]) else (bool(row[c]) if types[c] == "boolean" else row[c])) for c in columns}
        params["geom"] = row[gdf.geometry.name].wkt if row[gdf.geometry.name] is not None else None
        rows.append(params)
    connection.execute(insert_sql, rows)


def load_postgis(poi: gpd.GeoDataFrame, anchors: gpd.GeoDataFrame) -> dict:
    load_dotenv(ROOT / ".env")
    engine = get_engine()
    with engine.begin() as connection:
        _write_table(poi, "drt_priority_poi", "transport", connection)
        connection.execute(text("ALTER TABLE transport.drt_priority_poi ALTER COLUMN poi_id SET NOT NULL"))
        connection.execute(text("ALTER TABLE transport.drt_priority_poi ADD PRIMARY KEY (poi_id)"))
        connection.execute(text("CREATE INDEX drt_priority_poi_geometry_gix ON transport.drt_priority_poi USING GIST (geometry)"))
        connection.execute(text("CREATE INDEX drt_priority_poi_category_idx ON transport.drt_priority_poi (poi_category)"))
        connection.execute(text("CREATE INDEX drt_priority_poi_reliability_idx ON transport.drt_priority_poi (poi_reliability_class)"))

        _write_table(anchors, "drt_anchor_poi_context", "transport", connection)
        connection.execute(text("ALTER TABLE transport.drt_anchor_poi_context ALTER COLUMN stop_id SET NOT NULL"))
        connection.execute(text("ALTER TABLE transport.drt_anchor_poi_context ADD PRIMARY KEY (stop_id)"))
        connection.execute(text("CREATE INDEX drt_anchor_poi_context_geometry_gix ON transport.drt_anchor_poi_context USING GIST (geometry)"))
        connection.execute(text("CREATE INDEX drt_anchor_poi_context_adm_idx ON transport.drt_anchor_poi_context (adm_cd)"))

        poi_count = connection.execute(text("SELECT count(*) FROM transport.drt_priority_poi")).scalar_one()
        anchor_count = connection.execute(text("SELECT count(*) FROM transport.drt_anchor_poi_context")).scalar_one()
    with engine.connect() as connection:
        def exists(table: str, cond: str) -> bool:
            return bool(connection.execute(text(f"SELECT EXISTS(SELECT 1 FROM {cond})")).scalar_one())
        poi_pk = exists("drt_priority_poi", "pg_constraint WHERE conrelid='transport.drt_priority_poi'::regclass AND contype='p'")
        anchor_pk = exists("drt_anchor_poi_context", "pg_constraint WHERE conrelid='transport.drt_anchor_poi_context'::regclass AND contype='p'")
        poi_gist = exists("drt_priority_poi", "pg_indexes WHERE schemaname='transport' AND tablename='drt_priority_poi' AND indexdef ILIKE '%USING gist%'")
        anchor_gist = exists("drt_anchor_poi_context", "pg_indexes WHERE schemaname='transport' AND tablename='drt_anchor_poi_context' AND indexdef ILIKE '%USING gist%'")
        poi_srid = int(connection.execute(text("SELECT DISTINCT ST_SRID(geometry) FROM transport.drt_priority_poi")).scalar_one())
        anchor_srid = int(connection.execute(text("SELECT DISTINCT ST_SRID(geometry) FROM transport.drt_anchor_poi_context")).scalar_one())
        poi_dup = connection.execute(text("SELECT count(*) FROM (SELECT poi_id FROM transport.drt_priority_poi GROUP BY poi_id HAVING count(*) > 1) t")).scalar_one()
        poi_null_geom = connection.execute(text("SELECT count(*) FROM transport.drt_priority_poi WHERE geometry IS NULL")).scalar_one()
    engine.dispose()
    return {"drt_priority_poi": {"rows": int(poi_count), "primary_key": poi_pk, "geometry_gist": poi_gist,
                "srid": poi_srid, "duplicate_id": int(poi_dup), "null_geometry": int(poi_null_geom)},
            "drt_anchor_poi_context": {"rows": int(anchor_count), "primary_key": anchor_pk, "geometry_gist": anchor_gist, "srid": anchor_srid}}


def main() -> None:
    poi = gpd.read_parquet(ROOT / "data/interim/poi/changwon_priority_poi.parquet")
    context = pd.read_parquet(REPORTS / "drt_anchor_poi_context.parquet")
    network = gpd.read_parquet(ROOT / "data/release/changwon_drt_stop_network_accessibility.parquet")

    poi_reliability = poi_with_reliability(poi)
    poi_reliability.to_parquet(REPORTS / "drt_priority_poi_with_reliability.parquet", index=False)

    anchors = anchor_primary_context(context)
    anchors = anchors.merge(network[["stop_id", "geometry"]], on="stop_id", validate="one_to_one")
    anchors = gpd.GeoDataFrame(anchors, geometry="geometry", crs=4326)
    anchors.to_parquet(REPORTS / "drt_anchor_poi_context_reliability.parquet", index=False)

    postgis = load_postgis(poi_reliability, anchors)

    demand = anchors.loc[anchors.is_demand_robust_clean]
    supply = anchors.loc[anchors.is_supply_robust_clean]
    demand_summary, supply_summary = primary_context_summary(demand), primary_context_summary(supply)
    demand_dong, supply_dong = dong_pattern(demand), dong_pattern(supply)

    def examples(frame: pd.DataFrame, condition, columns, n=3):
        subset = frame.loc[condition(frame)]
        return subset.head(n)[columns].to_dict("records")

    example_columns = ["stop_id", "stop_name", "adm_nm", "nearest_primary_poi_walk_m", "primary_category_count_walk_800m", "anchor_context_class"]
    analysis = {"generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "HIRA/MOIS 공식 POI context 신뢰도 구분, PostGIS 적재, Primary POI Context 정리. POI score/Anchor score 변경 없음.",
        "category_reliability": CATEGORY_RELIABILITY, "official_source_status": OFFICIAL_SOURCE_STATUS,
        "primary_categories": list(PRIMARY_CATEGORIES),
        "postgis": postgis,
        "anchor_primary_context": {"demand_supported": demand_summary, "supply_gap": supply_summary},
        "dong_summary": {"demand_supported": demand_dong, "supply_gap": supply_dong},
        "examples": {
            "primary_context_present": examples(anchors, lambda f: f.anchor_context_class.eq("PRIMARY_CONTEXT_PRESENT"), example_columns),
            "primary_context_distant": examples(anchors, lambda f: f.anchor_context_class.eq("PRIMARY_CONTEXT_DISTANT"), example_columns),
            "no_primary_context": examples(anchors, lambda f: f.anchor_context_class.eq("NO_PRIMARY_CONTEXT_WITHIN_1200M"), example_columns)},
        "limitations": [
            "poi_reliability_class는 official completeness 등급이 아니라 현재 공식 POI를 분석 context로 사용할 때의 내부 진단이다.",
            "healthcare/administration의 0 count는 현재 공식 POI 기준 미확인이며 '시설 없음'을 의미하지 않는다.",
            "이 데이터셋은 여전히 experimental이며 data/release, release_manifest에는 반영하지 않았다."],
        "warnings": ["POI 점수, destination score, Anchor score를 생성하지 않았다.", "Anchor ranking을 재계산하지 않았다.", "Dashboard는 수정하지 않았다."],
        "errors": []}
    (REPORTS / "drt_anchor_poi_reliability_analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = ["# DRT Anchor POI Reliability", "",
             "HIRA/MOIS 공식 POI context의 category별 신뢰도 구분과 Primary POI Context 요약이며, POI 점수나 최종 Anchor 후보를 생성하지 않는다.", "",
             "## Category Reliability", ""]
    for category, info in CATEGORY_RELIABILITY.items():
        lines.append(f"- {category}: {info['poi_reliability_class']}")
    lines += ["", "## PostGIS", "", f"- drt_priority_poi rows: {postgis['drt_priority_poi']['rows']}",
              f"- drt_anchor_poi_context rows: {postgis['drt_anchor_poi_context']['rows']}", ""]
    for label, summary in [("Demand-supported", demand_summary), ("Supply-gap", supply_summary)]:
        lines += [f"## Primary Context — {label}", "", f"- anchors: {summary['anchors']}",
                  f"- either primary within800: {summary['either_primary_within800']}",
                  f"- no primary within1200: {summary['no_primary_within1200']}",
                  f"- context class distribution: {summary['context_class_distribution']}", ""]
    lines += ["## Guardrails", "", "- POI 점수, Anchor score 변경, ranking 재계산을 수행하지 않았다.",
              "- data/release, release_manifest, Dashboard는 수정하지 않았다.", ""]
    (REPORTS / "drt_anchor_poi_reliability_analysis.md").write_text("\n".join(lines), encoding="utf-8")

    qa_path = REPORTS / "changwon_drt_poi_qa.json"
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    qa["postgis"] = postgis
    qa["category_reliability"] = {category: info["poi_reliability_class"] for category, info in CATEGORY_RELIABILITY.items()}
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    metadata = {
        "dataset": "changwon_drt_priority_poi", "status": "experimental_context_layer",
        "source": "HIRA API + MOIS local official file", "official_source_attempt": "success / official-only",
        "official_source_status": OFFICIAL_SOURCE_STATUS,
        "official_sources_attempted": {
            "HEALTHCARE": "건강보험심사평가원 병원정보서비스 (data.go.kr 15001698) — approved API",
            "ADMINISTRATION": "행정안전부 읍면동 하부행정기관 현황 (data.go.kr 15059715) — local official file"},
        "category_reliability": {category: info["poi_reliability_class"] for category, info in CATEGORY_RELIABILITY.items()},
        "primary_categories": list(PRIMARY_CATEGORIES),
        "grain_poi": "1 row per POI (poi_id primary key)", "grain_anchor_context": "1 row per unique robust_clean stop_id",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "poi_reliability_class는 official completeness 등급이 아닌 내부 분석 진단이다.",
            "0 count는 현재 공식 POI 기준 미확인이며 '시설 없음'을 의미하지 않는다.",
            "curated release에는 아직 포함하지 않았다.",
            "POI 점수, Anchor score, Service Zone, Dashboard 반영은 이 단계에서 수행하지 않았다."]}
    (ROOT / "metadata/changwon_drt_priority_poi_metadata.yaml").write_text(
        __import__("yaml").safe_dump(metadata, allow_unicode=True, sort_keys=False), encoding="utf-8")

    print(f"Reliability layer complete: poi={postgis['drt_priority_poi']['rows']}, anchors={postgis['drt_anchor_poi_context']['rows']}")


if __name__ == "__main__":
    main()
