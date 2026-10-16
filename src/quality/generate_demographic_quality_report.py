"""Generate demographic release QA from actual frames."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd


def build_demographic_quality_report(data: gpd.GeoDataFrame, population: pd.DataFrame) -> dict[str, Any]:
    """Report code join, population validity, preservation and geometry."""
    population_codes = set(population.adm_cd.astype(str))
    boundary_codes = set(data.dong_code.astype(str))
    names = data[["dong_code", "dong_name"]].merge(
        population[["adm_cd", "total_adm_nm"]], left_on="dong_code", right_on="adm_cd", how="left"
    )
    quality = {
        "negative_total": int((data.population_total < 0).sum()),
        "negative_elderly": int((data.population_65_plus < 0).sum()),
        "elderly_gt_total": int((data.population_65_plus > data.population_total).sum()),
        "null_population": int(data[["population_total", "population_65_plus"]].isna().any(axis=1).sum()),
    }
    join = {
        "matched": len(boundary_codes & population_codes),
        "unmatched_boundary": len(boundary_codes - population_codes),
        "unmatched_population": len(population_codes - boundary_codes),
        "duplicate_adm_cd": int(population.adm_cd.duplicated(keep=False).sum()),
        "name_mismatch": int((names.total_adm_nm.notna() & names.dong_name.ne(names.total_adm_nm)).sum()),
    }
    invariants = {
        "dong_rows": len(data), "master_stop_sum": int(data.master_stop_count.sum()),
        "observed_stop_sum": int(data.observed_stop_count.sum()),
        "analysis_ready_stop_sum": int(data.analysis_ready_stop_count.sum()),
    }
    failed = any(quality.values()) or join["matched"] != 55 or any(join[k] for k in ("unmatched_boundary", "unmatched_population", "duplicate_adm_cd")) or invariants != {"dong_rows": 55, "master_stop_sum": 2926, "observed_stop_sum": 888, "analysis_ready_stop_sum": 886}
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "population_source": "통계청 SGIS 인구주택총조사 OpenAPI",
        "population_reference_year": 2024,
        "population_totals": {"population_total": int(data.population_total.sum()), "population_65_plus": int(data.population_65_plus.sum())},
        "boundary_dong_count": len(data), "population_dong_count": len(population),
        "join": join, "population_quality": quality, "context_invariants": invariants,
        "geometry": {"crs": f"EPSG:{data.crs.to_epsg()}", "invalid": int((~data.geometry.is_valid).sum()), "null": int(data.geometry.isna().sum())},
        "warnings": ["Boundary (2025-06-30), population (2024), and transit observations (2025-08-18 to 2025-08-25) have different reference periods.", "Demographic metrics are context values, not DRT suitability or vulnerability scores."],
        "status": "FAIL" if failed else "PASS_WITH_WARNINGS",
    }


def write_demographic_quality_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
