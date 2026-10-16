"""Generate exploratory DRT correlation, sensitivity, and stability reports."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.drt_scenario_analysis import (
    CANDIDATE_FEATURES, QA_FEATURES, SCENARIOS, base_scores, correlation_analysis,
    distribution_analysis, group_profiles, prepare_features, scenario_differentiation,
    sensitivity_analysis,
)

REPORTS = ROOT / "reports"


def serializable_matrix(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {row: {column: float(value) for column, value in values.items()} for row, values in frame.to_dict(orient="index").items()}


def main() -> None:
    source = gpd.read_parquet(ROOT / "data" / "release" / "changwon_drt_dong_features.parquet")
    features = prepare_features(source)
    correlation = correlation_analysis(features)
    distribution = distribution_analysis(features)

    long_correlation = pd.DataFrame(correlation["pairs"])
    long_correlation.to_csv(REPORTS / "drt_feature_correlation.csv", index=False, encoding="utf-8-sig")
    correlation_json = {
        "candidate_features": CANDIDATE_FEATURES,
        "pearson": serializable_matrix(correlation["pearson"]),
        "spearman": serializable_matrix(correlation["spearman"]),
        "redundancy_thresholds": {"high": 0.80, "very_high": 0.90},
        "redundancy_warnings": correlation["redundancy_warnings"],
    }
    (REPORTS / "drt_feature_correlation.json").write_text(json.dumps(correlation_json, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS / "drt_feature_distribution.json").write_text(json.dumps(distribution, ensure_ascii=False, indent=2), encoding="utf-8")

    stability_parts, contribution_parts, base_parts = [], [], []
    scenario_report = {}
    for offset, scenario in enumerate(SCENARIOS):
        stability, contributions, weights = sensitivity_analysis(features, scenario, combinations=1000, seed=20260830 + offset)
        base = base_scores(features, scenario)
        stability_parts.append(stability)
        contribution_parts.append(contributions)
        base_parts.append(base)
        valid = base.dropna(subset=["exploratory_score"])
        scenario_report[scenario.key] = {
            "label": scenario.label, "exploratory_not_policy_recommendation": True,
            "base_weights": scenario.weights, "eligible_dong_count": int(len(valid)),
            "ineligible_dongs": base.loc[base["exploratory_score"].isna(), "adm_nm"].tolist(),
            "base_score_min": float(valid["exploratory_score"].min()),
            "base_score_max": float(valid["exploratory_score"].max()),
            "base_rank_first_five": valid.nsmallest(5, "base_rank")[["adm_nm", "base_rank", "exploratory_score"]].to_dict(orient="records"),
            "high_top5_frequency": stability.loc[stability["top5_frequency"].gt(0)].nlargest(10, "top5_frequency")[["adm_nm", "top5_frequency", "top10_frequency", "mean_rank", "rank_std", "stability_class"]].to_dict(orient="records"),
            "weight_generation": {"method": "fixed-seed uniform perturbation +/-0.10 then non-negative clipping and row normalization", "seed": 20260830 + offset, "combinations": len(weights)},
            "weight_ranges": {column: {"min": float(weights[column].min()), "max": float(weights[column].max())} for column in weights},
            "rank_stability_class_counts": stability["stability_class"].value_counts().to_dict(),
            "sensitivity_summary": {
                "mean_rank_std": float(stability["rank_std"].mean()),
                "max_rank_std": float(stability["rank_std"].max()),
                "max_rank_std_dong": str(stability.loc[stability["rank_std"].idxmax(), "adm_nm"]),
                "mean_rank_span": float((stability["worst_rank"] - stability["best_rank"]).mean()),
                "max_rank_span": int((stability["worst_rank"] - stability["best_rank"]).max()),
            },
            "profile_comparison": group_profiles(features, base, scenario),
        }

    stability_all = pd.concat(stability_parts, ignore_index=True)
    contributions_all = pd.concat(contribution_parts, ignore_index=True)
    base_all = pd.concat(base_parts, ignore_index=True)
    stability_all.to_parquet(REPORTS / "drt_scenario_sensitivity.parquet", index=False)
    contributions_all.to_parquet(REPORTS / "drt_scenario_contributions.parquet", index=False)
    base_all.to_parquet(REPORTS / "drt_scenario_base_scores.parquet", index=False)

    differentiation = scenario_differentiation(base_all)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "exploratory scenario sensitivity; not a policy recommendation",
        "candidate_features": CANDIDATE_FEATURES,
        "qa_features_excluded": sorted(QA_FEATURES),
        "feature_relationships": correlation_json,
        "scenarios": scenario_report,
        "scenario_rank_spearman": differentiation,
        "scenario_differentiation_warnings": [key for key, value in differentiation.items() if abs(value) >= 0.85],
        "temporal": {
            "top3_vs_demand_spearman": float(correlation["spearman"].loc["top3_hour_share_pct", "ridership_per_1000_population_pct"]),
            "weekend_weekday_ratio": distribution["weekend_weekday_ratio"],
            "included_in_core_score": False,
        },
        "null_policy": "Required feature NULL makes the scenario score NULL; weights are not renormalized.",
        "warnings": ["Scores and ranks are exploratory analytical outputs, not DRT recommendations.", "Eight-day ridership cannot establish long-term or seasonal operating patterns."],
        "errors": [],
    }
    (REPORTS / "drt_scenario_analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# DRT Exploratory Scenario Analysis", "", "이 보고서는 정책 추천이 아니라 Feature 중복과 가중치 민감도를 검토하는 분석 산출물이다.", "", "## Scenario summary", ""]
    for scenario in SCENARIOS:
        item = scenario_report[scenario.key]
        lines += [f"### {scenario.label}", "", f"- Eligible dongs: {item['eligible_dong_count']}", f"- Base score range: {item['base_score_min']:.4f}–{item['base_score_max']:.4f}", f"- Sensitivity combinations: {item['weight_generation']['combinations']}", ""]
    lines += ["## Guardrails", "", "- QA 및 coverage Feature는 모든 시나리오 점수에서 제외했다.", "- 필수 Feature가 NULL이면 score를 NULL로 유지했다.", "- 상위 빈도와 rank stability는 최종 후보지 추천이 아니다.", ""]
    (REPORTS / "drt_scenario_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Scenario analysis complete: {len(stability_all)} stability rows, {len(contributions_all)} contribution rows")


if __name__ == "__main__":
    main()
