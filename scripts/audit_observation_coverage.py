"""Run Audit A: dong-level STCIS ridership observation coverage audit.

Reads GeoJSON attribute tables only (no geopandas/pyarrow) because this
environment's native parquet/GDAL extensions are blocked by Windows Smart
App Control. See reports/observation_coverage_summary.md for the resulting
interpretation and known gaps.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.observation_coverage_audit import build_dong_coverage_table, coverage_audit_report, COVERAGE_COLUMNS
from src.io_utils import read_geojson_properties

RELEASE = ROOT / "data" / "release"
REPORTS = ROOT / "reports"
METADATA = ROOT / "metadata"


def main() -> None:
    context = read_geojson_properties(RELEASE / "changwon_transit_demographic_context_by_dong.geojson")
    dong_features = read_geojson_properties(RELEASE / "changwon_drt_dong_features.geojson")
    master = read_geojson_properties(RELEASE / "dim_transit_stop.geojson")
    stop_profile = read_geojson_properties(RELEASE / "changwon_transit_stop_profile.geojson")

    table = build_dong_coverage_table(context)
    assert (table["observed_stop_count"] <= table["official_stop_count"]).all()
    assert (table["analysis_ready_stop_count"] <= table["observed_stop_count"]).all()
    assert table["observation_coverage"].dropna().between(0, 1).all()
    assert table["analysis_ready_coverage"].dropna().between(0, 1).all()
    assert len(table) == 55

    official_master_total = int(len(master))
    official_inside_boundary_total = int(table["official_stop_count"].sum())
    observed_total_inside = int(table["observed_stop_count"].sum())
    analysis_ready_total_inside = int(table["analysis_ready_stop_count"].sum())
    observed_total_citywide = int(len(stop_profile))
    analysis_ready_total_citywide = int(stop_profile["is_analysis_ready"].sum())

    report = coverage_audit_report(
        table, dong_features, official_master_total, official_inside_boundary_total,
        observed_total_inside, analysis_ready_total_inside,
    )
    report["citywide_counts"]["observed_stops_citywide_including_outside_boundary"] = observed_total_citywide
    report["citywide_counts"]["analysis_ready_stops_citywide_including_outside_boundary"] = analysis_ready_total_citywide
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["reconciliation"] = {
        "observed_equals_valid_plus_missing_plus_duplicate": bool(
            (context["valid_stop_count"] + context["missing_observation_stop_count"] + context["duplicate_observation_stop_count"]).eq(context["observed_stop_count"]).all()
        ),
        "analysis_ready_equals_valid_plus_missing": bool(
            (context["valid_stop_count"] + context["missing_observation_stop_count"]).eq(context["analysis_ready_stop_count"]).all()
        ),
    }

    REPORTS.mkdir(exist_ok=True)
    table[COVERAGE_COLUMNS].sort_values("dong_code").to_csv(REPORTS / "dong_observation_coverage.csv", index=False, encoding="utf-8-sig")
    (REPORTS / "observation_coverage_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    dist = report["coverage_distribution"]["observation_coverage"]
    lowest = report["coverage_extremes"]["lowest_observation_coverage"][:5]
    highest = report["coverage_extremes"]["highest_observation_coverage"][:5]
    lines = [
        "# 이용량 관측 Coverage Audit (Audit A)",
        "",
        "이 보고서는 STCIS 이용량 관측이 창원시 공식 정류소를 행정동별로 얼마나 커버하는지 확인하기 위한 것이며,",
        "DRT 필요성이나 특정 행정동의 분석 가능 여부를 판정하는 문서가 아니다.",
        "",
        "## 환경 제약 고지",
        "",
        "- 이 PC의 Windows Smart App Control이 pyarrow/geopandas 네이티브 확장 로딩을 차단하여, 이 audit은 parquet이 아니라",
        "  GeoJSON(changwon_transit_demographic_context_by_dong.geojson 등)의 속성만 pandas로 직접 읽어 계산했다.",
        "- `dong_observation_coverage.parquet` 산출물은 pyarrow/fastparquet을 사용할 수 없어 생성하지 못했다. 동일한 값은",
        "  `dong_observation_coverage.csv`와 `observation_coverage_audit.json`에 모두 있다.",
        "- 정류소 단위(관측/미관측) route_count 분포 비교는 `changwon_transit_route_stop.parquet`이 없어 수행하지 못했고,",
        "  행정동 단위 노선유형 상관관계로만 대체했다 (판단 보류).",
        "",
        "## 전체 집계",
        "",
        f"- 전체 공식 정류소(Master): {report['citywide_counts']['official_master_stops_total']:,}개",
        f"- 창원시 경계 내부 공식 정류소: {report['citywide_counts']['official_stops_inside_boundary']:,}개",
        f"- 경계 내부 관측 정류소: {report['citywide_counts']['observed_stops_inside_boundary']:,}개",
        f"- 경계 내부 분석 가능 정류소: {report['citywide_counts']['analysis_ready_stops_inside_boundary']:,}개",
        f"- 경계 무관 전체 관측 정류소: {report['citywide_counts']['observed_stops_citywide_including_outside_boundary']:,}개",
        f"- 경계 내부 기준 전체 observation coverage: {report['citywide_counts']['observation_coverage_citywide']:.1%}" if report['citywide_counts']['observation_coverage_citywide'] is not None else "",
        "",
        "## 행정동별 coverage 분포 (observation_coverage = observed_stop_count / official_stop_count)",
        "",
    ]
    if dist:
        lines += [
            f"- min: {dist['min']:.1%}",
            f"- p25: {dist['p25']:.1%}",
            f"- median: {dist['median']:.1%}",
            f"- p75: {dist['p75']:.1%}",
            f"- max: {dist['max']:.1%}",
            "",
        ]
    lines += ["## Coverage가 가장 낮은 행정동 (하위 5개)", ""]
    for row in lowest:
        lines.append(f"- {row['dong_name']} ({row['dong_code']}): {row['observed_stop_count']}/{row['official_stop_count']} = {row['observation_coverage']:.1%}")
    lines += ["", "## Coverage가 가장 높은 행정동 (상위 5개)", ""]
    for row in highest:
        lines.append(f"- {row['dong_name']} ({row['dong_code']}): {row['observed_stop_count']}/{row['official_stop_count']} = {row['observation_coverage']:.1%}")
    lines += [
        "",
        "## 기존 이용량 기반 분석 해석에 미치는 영향",
        "",
        "- coverage가 낮은 행정동은 STCIS 이용량이 그 행정동의 실제 이용 실태 중 더 작은 표본만 반영한다는 뜻이며,",
        "  '이용량이 적다'와 '관측이 적다'를 구분해서 읽어야 한다.",
        "- 이 문서는 특정 coverage 수치를 '분석 가능/불가능'으로 나누는 임의 임계값을 제시하지 않는다.",
        "- 자세한 편향 신호는 `observation_coverage_audit.json`의 `route_supply_bias_proxy`를 참고하되, 이는 정류소 단위가",
        "  아닌 행정동 단위 근사치다.",
    ]
    (REPORTS / "observation_coverage_summary.md").write_text("\n".join(line for line in lines if line is not None), encoding="utf-8")

    metadata = {
        "title": "Changwon Observation Coverage Audit",
        "dataset_name": "changwon_observation_coverage",
        "created_at": datetime.now(timezone.utc).date().isoformat(),
        "purpose": "행정동별 STCIS 이용량 관측이 공식 정류소를 얼마나 커버하는지 감사",
        "not_purpose": "DRT 필요성 판정, 행정동 분석 가능/불가능 이분 판정",
        "grain": "1 row per official Changwon administrative dong",
        "primary_key": "dong_code",
        "row_count": int(len(table)),
        "source_files": [
            "data/release/changwon_transit_demographic_context_by_dong.geojson",
            "data/release/changwon_drt_dong_features.geojson",
            "data/release/dim_transit_stop.geojson",
            "data/release/changwon_transit_stop_profile.geojson",
        ],
        "environment_note": "pyarrow/geopandas/duckdb/fastparquet unavailable (Windows Smart App Control); GeoJSON properties read directly with pandas",
        "known_gaps": [
            "stop-level route_count bias between observed and unobserved stops requires changwon_transit_route_stop.parquet (unavailable)",
            "no official 도심/읍면 classification exists in source data; not fabricated here",
        ],
        "null_policy": "관측되지 않은 정류소를 이용량 0으로 간주하지 않음",
    }
    METADATA.mkdir(exist_ok=True)
    (METADATA / "changwon_observation_coverage.yaml").write_text(
        yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False), encoding="utf-8",
    )
    print(f"Audit A complete: {len(table)} dongs, observation_coverage median={dist['median']:.3f}" if dist else "Audit A complete")


if __name__ == "__main__":
    main()
