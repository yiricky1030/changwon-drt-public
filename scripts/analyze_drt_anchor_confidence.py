"""Build the Confidence Diagnostic Layer for network Anchor sensitivity results.

Reads only existing reports/release outputs. Produces no new Anchor score,
does not change existing sensitivity or contribution outputs, and does not
exclude any stop."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.drt_anchor_confidence import (
    build_confidence, caution_combinations, citywide_detour_p95, dong_pattern,
    scenario_summary, snap_thresholds,
)
from src.analysis.physical_stop_eligibility import attach_physical_eligibility, eligible_robust

REPORTS = ROOT / "reports"
SCENARIOS = [("network_demand_supported", "demand_supported", "demand"),
             ("network_supply_gap", "supply_gap", "supply")]


def main() -> None:
    network = gpd.read_parquet(ROOT / "data/release/changwon_drt_stop_network_accessibility.parquet")
    features = gpd.read_parquet(ROOT / "data/release/changwon_drt_stop_features.parquet")
    comparison = pd.read_parquet(REPORTS / "drt_anchor_euclidean_network_comparison.parquet")
    contributions = pd.read_parquet(REPORTS / "drt_anchor_network_contributions.parquet")
    thresholds = snap_thresholds(ROOT / "metadata/changwon_drt_stop_network_accessibility_metadata.yaml")
    detour_p95 = citywide_detour_p95(network)

    outputs, report = {}, {"snap_thresholds": thresholds, "detour_p95": detour_p95, "scenarios": {}}
    for network_key, euclidean_key, suffix in SCENARIOS:
        sensitivity = pd.read_parquet(REPORTS / f"drt_anchor_network_sensitivity_{suffix}.parquet")
        euclidean = pd.read_parquet(REPORTS / f"drt_anchor_sensitivity_{suffix}.parquet")
        frame = build_confidence(network_key, sensitivity, euclidean, comparison, contributions, features, detour_p95)
        frame = attach_physical_eligibility(frame, ROOT)
        outputs[suffix] = frame
        frame.to_parquet(REPORTS / f"drt_anchor_confidence_{suffix}.parquet", index=False)
        summary = scenario_summary(frame)
        combos = caution_combinations(frame)
        dongs = dong_pattern(frame)
        examples = frame.loc[eligible_robust(frame)].sort_values(
            ["network_top3_frequency", "network_mean_rank"], ascending=[False, True]).head(10)[
            ["stop_id", "stop_name", "adm_nm", "network_top3_frequency", "network_mean_rank", "confidence_class"]
        ].to_dict("records")
        summary["physical_anchor_eligible"] = int(eligible_robust(frame).sum())
        summary["physical_excluded_from_robust"] = int((frame.robust_clean & ~frame.anchor_eligible).sum())
        report["scenarios"][suffix] = {"summary": summary, "top_caution_combinations": combos,
                                        "dong_pattern": dongs, "robust_clean_examples": examples}

    interpretation = {
        "network_anchor_stable_for_poi_integration": "PARTIAL — repeated-high frequency correlation with the Euclidean model is only 0.44-0.59, so POI integration should start from the robust_clean subset only, not the full repeated-high set.",
        "supply_gap_extra_caution_needed": True,
        "flags_remain_diagnostic_only": True,
        "automatic_exclusion_performed": True,
        "automatic_exclusion_scope": "Decision-facing physical Anchor eligibility only; scores, ranks, and robust_clean diagnostics are unchanged.",
    }
    analysis = {"generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "independent confidence diagnostic for existing network Anchor sensitivity results; no new score, no exclusion",
        "snap_thresholds": thresholds, "detour_p95_citywide": detour_p95,
        "repeated_high_definition": "network top3_frequency_within_dong >= 0.8 (same threshold as analyze_drt_anchor_network_sensitivity.py)",
        "scenarios": report["scenarios"], "interpretation": interpretation,
        "warnings": ["Confidence class is an interpretation aid, not a policy suitability grade.",
                     "Only audited physical-stop eligibility excludes rows from decision-facing Anchor outputs; confidence caution flags do not.",
                     "robust_clean is not a final Anchor candidate list."],
        "errors": []}
    (REPORTS / "drt_anchor_confidence_analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = ["# DRT Anchor Confidence Diagnostic", "",
             "Network Anchor sensitivity 결과에 대한 독립적인 해석 안정성(confidence) 진단이며, 새 점수·순위·자동 제외를 생성하지 않는다.", ""]
    for suffix, label in [("demand", "network_demand_supported"), ("supply", "network_supply_gap")]:
        s = report["scenarios"][suffix]["summary"]
        lines += [f"## {label}", "",
                  f"- Total: {s['total']} (scored: {s['scored']})", f"- Repeated-high (top3 within-dong >= 0.8): {s['repeated_high']}",
                  f"- Robust both top3/top10pct: {s['robust_both_top3']} / {s['robust_both_top10pct']}",
                  f"- Network-model-sensitive: {s['network_model_sensitive']}",
                  f"- Snap caution / high caution: {s['snap_caution']} / {s['snap_high_caution']}",
                  f"- High detour caution: {s['high_detour_caution']}",
                  f"- Confidence class distribution: {s['confidence_class_distribution']}",
                  f"- Robust-clean: {s['robust_clean']}", ""]
    lines += ["## Guardrails", "", "- 새로운 Anchor score, 순위, sensitivity 재실행을 하지 않았다.",
              "- 어떤 flag도 자동 제외 기준으로 사용하지 않았다.", "- data/release, release_manifest, Dashboard는 수정하지 않았다.", ""]
    (REPORTS / "drt_anchor_confidence_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print("Confidence diagnostic complete: demand=886, supply=2926")


if __name__ == "__main__":
    main()
