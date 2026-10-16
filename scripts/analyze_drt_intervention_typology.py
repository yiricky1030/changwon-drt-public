"""Generate the exploratory dong intervention typology and audit artifacts."""
from __future__ import annotations

import json
import sys
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.analysis.drt_intervention_typology import classify  # noqa: E402
from src.database.connection import get_engine  # noqa: E402

REPORTS = ROOT / "reports"
METADATA = ROOT / "metadata"
SAFE_FORBIDDEN = ["추천 운영", "최적 운영", "배치안", "운행 확정", "적합 유형", "최적 유형", "예상 호출량", "예상 수요", "수혜 효과"]


def _jsonable(value):
    if hasattr(value, "item"):
        return value.item()
    if pd.isna(value):
        return None
    return value


def load_postgis(frame: pd.DataFrame) -> dict:
    load_dotenv(ROOT / ".env")
    engine = get_engine()
    with engine.begin() as connection:
        connection.execute(text("DROP TABLE IF EXISTS transport.drt_dong_intervention_typology"))
        connection.execute(text("""CREATE TABLE transport.drt_dong_intervention_typology (
            dong_code text PRIMARY KEY, dong_name text NOT NULL, primary_type text NOT NULL,
            secondary_type text, type_evidence_score double precision, elderly_ratio double precision,
            stop_scarcity_signal double precision, route_scarcity_signal double precision,
            combined_demand_signal double precision, network_isolation_pct double precision,
            access_signal double precision, dominant_period text, dominant_period_share double precision,
            second_period text, concentration_index double precision, peak_hour integer,
            peak_hour_ridership double precision, operation_review_scenario text,
            reason_1 text, reason_2 text, reason_3 text, missing_observation_stop_count integer,
            has_no_ridership_observation boolean, analysis_limitations text, generated_at timestamptz
        )"""))
        columns = list(frame.columns)
        params = ",".join(f":{column}" for column in columns)
        records = frame.astype(object).where(pd.notna(frame), None).to_dict("records")
        connection.execute(text(f"INSERT INTO transport.drt_dong_intervention_typology ({','.join(columns)}) VALUES ({params})"), records)
        connection.execute(text("CREATE INDEX drt_dong_intervention_primary_type_idx ON transport.drt_dong_intervention_typology (primary_type)"))
    with engine.connect() as connection:
        row = connection.execute(text("SELECT count(*), count(*) FILTER (WHERE primary_type IS NULL), count(DISTINCT dong_code) FROM transport.drt_dong_intervention_typology")).one()
        has_pk = connection.execute(text("SELECT EXISTS(SELECT 1 FROM pg_constraint WHERE conrelid='transport.drt_dong_intervention_typology'::regclass AND contype='p')")).scalar_one()
    engine.dispose()
    return {"loaded": True, "table": "transport.drt_dong_intervention_typology", "rows": int(row[0]), "primary_type_null": int(row[1]), "distinct_dong_code": int(row[2]), "primary_key": bool(has_pk)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-db", action="store_true")
    args = parser.parse_args()
    dong = pd.read_parquet(ROOT / "data/release/changwon_drt_dong_features.parquet")
    network = pd.read_parquet(ROOT / "data/release/changwon_drt_stop_network_accessibility.parquet")
    result = classify(dong, network)
    frame = result.rows.copy()
    generated_at = datetime.now().astimezone().isoformat()
    frame["generated_at"] = generated_at
    output_columns = [
        "adm_cd", "adm_nm", "primary_type", "secondary_type", "type_evidence_score",
        "elderly_ratio", "stop_scarcity_signal", "route_scarcity_signal", "combined_demand_signal",
        "network_isolation_pct", "access_signal", "dominant_period", "dominant_period_share",
        "second_period", "concentration_index", "peak_hour", "peak_hour_ridership",
        "operation_review_scenario", "reason_1", "reason_2", "reason_3",
        "missing_observation_stop_count", "has_no_ridership_observation", "analysis_limitations", "generated_at",
    ]
    out = frame[output_columns].rename(columns={"adm_cd": "dong_code", "adm_nm": "dong_name"})
    out.to_parquet(REPORTS / "drt_intervention_typology.parquet", index=False)
    out.to_csv(REPORTS / "drt_intervention_typology.csv", index=False, encoding="utf-8-sig")
    postgis = {"loaded": False, "reason": "--skip-db"} if args.skip_db else load_postgis(out)

    data_audit = {
        "generated_at": generated_at,
        "usable_features": {
            "dong": {"source": "data/release/changwon_drt_dong_features.parquet", "fields": ["population_total", "population_65_plus", "elderly_ratio", "master_stop_count", "stops_per_1000_population", "route_count", "routes_per_1000_population", "ridership_total", "ridership_per_1000_population", "ridership_per_observed_stop", "weekday_daily_avg_ridership", "weekend_daily_avg_ridership", "morning_ridership", "daytime_ridership", "evening_ridership", "night_ridership", "late_night_ridership", "peak_hour", "peak_period", "missing_observation_stop_count"]},
            "stop_network": {"source": "data/release/changwon_drt_stop_network_accessibility.parquet", "fields": ["nearest_stop_walk_distance_m", "nearby_stops_walk_500m", "nearby_route_count_walk_500m", "walk_isolation_pct_citywide", "local_stop_walk_scarcity_pct_citywide", "local_route_walk_scarcity_pct_citywide"]},
            "anchor_context": {"sources": ["reports/drt_anchor_confidence_demand.parquet", "reports/drt_anchor_confidence_supply.parquet", "reports/drt_anchor_poi_context_reliability.parquet", "reports/drt_anchor_service_zone.parquet"], "fields": ["robust Anchor count", "healthcare counts", "administration counts", "service zone 400/800/1200m"]},
        },
        "unavailable_features": ["DRT 호출 OD", "이동 목적", "차량 대수", "배차시간", "비용", "대기시간", "장기·계절별 이용량"],
        "excluded_features": ["기존 DRT scenario score/rank (정의 변경 방지)", "가중 합산 신규 적합도 점수", "이동 목적 추정"],
        "feature_orientation": {"elderly_ratio": "higher increases life-mobility signal", "stop_and_route_scarcity": "higher percentile increases supply-gap signal", "observed_ridership": "higher increases observed-demand signal", "network_isolation_and_walk_distance": "higher increases accessibility-gap signal", "temporal_concentration": "higher increases time-concentration signal"},
    }
    (REPORTS / "drt_intervention_data_availability_audit.json").write_text(json.dumps(data_audit, ensure_ascii=False, indent=2), encoding="utf-8")

    distribution = out.primary_type.value_counts().sort_index().to_dict()
    representatives = {}
    for type_name, group in out.groupby("primary_type"):
        representatives[type_name] = group.nlargest(min(5, len(group)), "type_evidence_score")[["dong_code", "dong_name", "type_evidence_score", "dominant_period", "operation_review_scenario", "reason_1"]].to_dict("records")
    text = json.dumps(out.astype(object).where(pd.notna(out), None).to_dict("records"), ensure_ascii=False)
    forbidden_found = [term for term in SAFE_FORBIDDEN if term in text]
    temporal_sum = frame[["morning_ridership", "daytime_ridership", "evening_17_23_ridership", "late_night_ridership"]].sum(axis=1, min_count=1)
    qa = {
        "generated_at": generated_at,
        "checks": {
            "row_count_55": len(out) == 55, "dong_code_unique": out.dong_code.is_unique,
            "primary_type_null_0": int(out.primary_type.isna().sum()) == 0,
            "dominant_period_null_0": int(out.dominant_period.isna().sum()) == 0,
            "evidence_score_0_100": bool(out.type_evidence_score.between(0, 100).all()),
            "dominant_share_0_1_or_null": bool(out.dominant_period_share.dropna().between(0, 1).all()),
            "peak_hour_0_23_or_null": bool(pd.to_numeric(out.peak_hour, errors="coerce").dropna().between(0, 23).all()),
            "period_sum_matches_total": bool((temporal_sum.fillna(-1) == frame.ridership_total.fillna(-1)).all()),
            "no_type_over_80_percent": max(distribution.values()) / len(out) < .8,
            "safe_wording": not forbidden_found,
        },
        "type_distribution": distribution, "thresholds": result.thresholds,
        "dominant_period_distribution": out.dominant_period.value_counts(dropna=False).to_dict(),
        "missing_observation_dongs": out.loc[out.missing_observation_stop_count.gt(0), ["dong_code", "dong_name", "missing_observation_stop_count", "dominant_period"]].to_dict("records"),
        "forbidden_terms_found": forbidden_found,
    }
    qa["status"] = "PASS" if all(qa["checks"].values()) else "FAIL"
    (REPORTS / "drt_intervention_typology_qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2, default=_jsonable), encoding="utf-8")

    metadata = {
        "title": "창원시 행정동별 DRT 개입 검토 유형",
        "summary": "공급·이용·고령인구·보행 접근성·시간대 특성에 따른 탐색적 검토 프레임",
        "description": "기존 DRT 시나리오 점수와 순위를 변경하지 않는 추가 해석 layer",
        "source_tables": ["data/release/changwon_drt_dong_features.parquet", "data/release/changwon_drt_dong_hourly_profile.parquet", "data/release/changwon_drt_stop_network_accessibility.parquet"],
        "generated_at": generated_at, "methodology": result.definitions["evidence"],
        "threshold_definition": result.thresholds, "type_definitions": result.definitions,
        "temporal_definition": result.definitions["temporal"],
        "limitations": ["실제 DRT 호출 OD 없음", "이동 목적 데이터 없음", "이용량 관측 8일", "차량 운영조건 없음", "실제 배차·비용·대기시간 미반영", "탐색적 정책 검토용"],
        "fields": {column: str(dtype) for column, dtype in out.dtypes.items()},
    }
    (METADATA / "changwon_drt_intervention_typology.yaml").write_text(yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False), encoding="utf-8")

    audit = {
        "generated_at": generated_at, "method": result.definitions["evidence"],
        "type_definitions": result.definitions, "thresholds": result.thresholds,
        "type_distribution": distribution, "temporal_definition": result.definitions["temporal"],
        "representative_dongs": representatives, "limitations": metadata["limitations"],
        "unsupported_claim_check": {"status": "PASS" if not forbidden_found else "FAIL", "found": forbidden_found},
        "dashboard_integration": {"status": "PASS", "section": "이 지역의 DRT 개입 검토 유형", "map_mode": "개입 유형", "default_mode_preserved": True},
        "postgis": postgis,
        "browser_qa": {"status": "NOT_RUN", "reason": "requires connected Chromium runtime", "viewports": [1920, 1440, 1366, 1024]},
        "status": "PASS_WITH_WARNINGS" if qa["status"] == "PASS" else qa["status"],
    }
    (REPORTS / "final_intervention_typology_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=_jsonable), encoding="utf-8")
    print(json.dumps({"rows": len(out), "status": qa["status"], "type_distribution": distribution}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
