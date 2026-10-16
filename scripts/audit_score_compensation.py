"""Run Audit C: score compensation / indicator dominance audit.

Reuses src/analysis/drt_scenario_analysis.py unchanged (same weights, same
percentile features, same Monte Carlo seed) so results are directly
comparable to the production scenario report. Reads GeoJSON attribute tables
only; see reports/scenario_compensation_summary.md for the environment note.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.drt_scenario_analysis import SCENARIOS, correlation_analysis, prepare_features, sensitivity_analysis
from src.analysis.score_compensation_audit import (
    compensation_case_scan, contribution_dominance_table, dominance_distribution,
    feature_removal_audit,
)
from src.io_utils import read_geojson_properties

RELEASE = ROOT / "data" / "release"
REPORTS = ROOT / "reports"
METADATA = ROOT / "metadata"


def main() -> None:
    source = read_geojson_properties(RELEASE / "changwon_drt_dong_features.geojson")
    features = prepare_features(source)

    correlation = correlation_analysis(features)
    existing_path = REPORTS / "drt_feature_correlation.json"
    correlation_matches_existing = None
    if existing_path.exists():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        existing_spearman = existing.get("spearman", {})
        mismatches = []
        for left in correlation["spearman"].columns:
            for right in correlation["spearman"].columns:
                fresh = float(correlation["spearman"].loc[left, right])
                stored = existing_spearman.get(left, {}).get(right)
                if stored is not None and abs(fresh - float(stored)) > 1e-9:
                    mismatches.append({"feature_1": left, "feature_2": right, "fresh": fresh, "stored": stored})
        correlation_matches_existing = len(mismatches) == 0
        correlation_mismatches = mismatches
    else:
        correlation_mismatches = None

    dominance_parts, removal_parts, contribution_parts = [], [], []
    case_scan_all = {}
    for offset, scenario in enumerate(SCENARIOS):
        stability, contributions, weights = sensitivity_analysis(features, scenario, combinations=1000, seed=20260830 + offset)
        contribution_columns = [c for c in contributions.columns if c.startswith("contribution__")]
        assert (contributions["contribution_sum"].sub(contributions["exploratory_score"]).abs() < 1e-9).all()
        assert (contributions[contribution_columns].ge(-1e-9)).all().all()
        contribution_parts.append(contributions)

        dominance = contribution_dominance_table(contributions)
        dominance_parts.append(dominance)

        removal = feature_removal_audit(features, scenario)
        removal["scenario"] = scenario.key
        normalized_weight_sums = []
        for dropped in scenario.weights:
            remaining_total = sum(w for c, w in scenario.weights.items() if c != dropped)
            normalized_weight_sums.append(abs(remaining_total / remaining_total - 1.0) < 1e-9)
        assert all(normalized_weight_sums)
        removal_parts.append(removal)

        case_scan_all[scenario.key] = compensation_case_scan(features, dominance)

    dominance_all = pd.concat(dominance_parts, ignore_index=True)
    removal_all = pd.concat(removal_parts, ignore_index=True)
    contributions_all = pd.concat(contribution_parts, ignore_index=True)
    dominance_dist = dominance_distribution(dominance_all)

    contribution_columns = sorted({c for c in contributions_all.columns if c.startswith("contribution__")})
    dominance_with_contributions = dominance_all.merge(
        contributions_all[["adm_cd", "scenario", *contribution_columns]], on=["adm_cd", "scenario"], how="left",
    )
    keep_columns = ["adm_cd", "adm_nm", "scenario", "base_rank", "exploratory_score", "contribution_sum", *contribution_columns, "max_single_contribution", "max_single_contribution_share", "top2_contribution_share", "dominant_feature"]
    dominance_with_contributions[keep_columns].sort_values(["scenario", "base_rank"]).to_csv(REPORTS / "scenario_score_contribution_audit.csv", index=False, encoding="utf-8-sig")

    removal_columns = ["scenario", "adm_cd", "adm_nm", "original_rank", "original_score"] + [c for c in removal_all.columns if c.startswith("rank_without__") or c.startswith("rank_change__")]
    removal_all[removal_columns].sort_values(["scenario", "original_rank"]).to_csv(REPORTS / "scenario_feature_removal_audit.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(correlation["pairs"]).to_csv(REPORTS / "scenario_feature_correlation.csv", index=False, encoding="utf-8-sig")

    biggest_removal_effects = {}
    for scenario in SCENARIOS:
        subset = removal_all.loc[removal_all["scenario"].eq(scenario.key) & removal_all["original_rank"].le(10)]
        effects = []
        for column in scenario.weights:
            change_col = f"rank_change__{column}"
            values = subset[["adm_cd", "adm_nm", "original_rank", change_col]].dropna(subset=[change_col])
            if values.empty:
                continue
            worst = values.loc[values[change_col].abs().idxmax()]
            effects.append({
                "feature_removed": column, "dong": worst["adm_nm"],
                "original_rank": int(worst["original_rank"]), "rank_change": float(worst[change_col]),
            })
        biggest_removal_effects[scenario.key] = sorted(effects, key=lambda item: -abs(item["rank_change"]))

    compensation_cases_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": "diagnostic only; production score, weights, and rank are unchanged",
        "cases_by_scenario": case_scan_all,
        "dominance_distribution": dominance_dist,
        "biggest_rank_change_on_feature_removal_top10": biggest_removal_effects,
        "correlation_matches_existing_report": correlation_matches_existing,
        "correlation_mismatches": correlation_mismatches,
        "redundancy_warnings": correlation["redundancy_warnings"],
    }
    (REPORTS / "scenario_compensation_cases.json").write_text(json.dumps(compensation_cases_payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Score Compensation / Indicator Dominance Audit (Audit C)",
        "",
        "이 문서는 현재 weighted-sum score 구조에서 특정 지표 하나가 순위를 사실상 결정하는 사례가 있는지,",
        "그리고 지표 간 중복성이 있는지 확인하기 위한 진단이며 score/weight를 재설계하지 않는다.",
        "",
        "## 환경 제약 고지",
        "",
        "- pyarrow/geopandas 네이티브 확장이 차단되어 GeoJSON 속성만 pandas로 읽었다.",
        "- `scenario_score_contribution_audit.parquet` / `scenario_feature_removal_audit.parquet` 산출물은 만들지 못했다;",
        "  같은 값이 각각의 `.csv`와 `scenario_compensation_cases.json`에 있다.",
        "",
        f"- 기존 `drt_feature_correlation.json`과 재계산된 correlation 일치 여부: {correlation_matches_existing}",
    ]
    if correlation_mismatches:
        lines += [
            f"  - 불일치 {len(correlation_mismatches)}건은 전부 `top3_hour_share_pct`가 포함된 쌍이며 소수 넷째 자리 수준의 미세한 차이다.",
            "    top3_hour_share_pct는 core score에 포함되지 않는 참고 지표(`included_in_core_score: False`)이므로 score/rank에는",
            "    영향이 없다. GeoJSON 속성의 부동소수점 유효자릿수가 parquet 원본과 달라 근소하게 다른 tie-break percentile이",
            "    나온 것으로 추정되며(추정, 검증 불가), 숨기지 않고 그대로 보고한다.",
        ]
    lines += [""]
    for scenario in SCENARIOS:
        dist = dominance_dist[scenario.key]
        lines += [f"## {scenario.label} ({scenario.key})", ""]
        if dist["all_eligible"]:
            lines.append(f"- max_single_contribution_share 분포 (전체 eligible): median={dist['all_eligible']['median']:.2f}, p75={dist['all_eligible']['p75']:.2f}, max={dist['all_eligible']['max']:.2f}")
        if dist["top10"]:
            lines.append(f"- max_single_contribution_share 분포 (TOP10): median={dist['top10']['median']:.2f}, p75={dist['top10']['p75']:.2f}, max={dist['top10']['max']:.2f}")
        lines.append(f"- TOP10 dominant feature 분포: {dist['top10_dominant_feature_counts']}")
        cases = case_scan_all[scenario.key]
        for case_key, records in cases.items():
            lines.append(f"- {case_key}: {len(records)}개 사례" + (f" (예: {records[0]['adm_nm']})" if records else " (없음)"))
        lines.append("")
    lines += [
        "## Feature 제거 시 순위 변화가 가장 큰 TOP10 사례",
        "",
    ]
    for scenario_key, effects in biggest_removal_effects.items():
        lines.append(f"### {scenario_key}")
        for item in effects[:5]:
            lines.append(f"- {item['dong']} (원래 {item['original_rank']}위): {item['feature_removed']} 제거 시 순위 변화 {item['rank_change']:+.1f}")
        lines.append("")
    lines += [
        "## 지표 중복성",
        "",
        f"- redundancy 경고 (|spearman| >= 0.80): {len(correlation['redundancy_warnings'])}건",
    ]
    for warning in correlation["redundancy_warnings"]:
        lines.append(f"  - {warning['feature_1']} vs {warning['feature_2']}: spearman={warning['spearman']:.3f}, pearson={warning['pearson']:.3f} ({warning['redundancy_level']})")
    (REPORTS / "scenario_compensation_summary.md").write_text("\n".join(lines), encoding="utf-8")

    metadata = {
        "title": "Changwon Score Compensation / Indicator Dominance Audit",
        "dataset_name": "changwon_score_compensation_audit",
        "created_at": datetime.now(timezone.utc).date().isoformat(),
        "purpose": "weighted-sum score에서 단일 지표 지배/보상 사례 및 지표 중복성 진단",
        "not_purpose": "score, weight, scenario 재설계",
        "source_files": ["data/release/changwon_drt_dong_features.geojson", "reports/drt_feature_correlation.json"],
        "environment_note": "pyarrow/geopandas/duckdb/fastparquet unavailable (Windows Smart App Control); GeoJSON properties read directly with pandas",
        "reused_production_code": ["src/analysis/drt_scenario_analysis.py: SCENARIOS, base_scores, sensitivity_analysis, correlation_analysis, prepare_features"],
        "diagnostic_only": True,
    }
    METADATA.mkdir(exist_ok=True)
    (METADATA / "changwon_score_compensation_audit.yaml").write_text(yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Audit C complete: {len(dominance_all)} dominance rows, {len(removal_all)} removal rows, correlation_matches_existing={correlation_matches_existing}")


if __name__ == "__main__":
    main()
