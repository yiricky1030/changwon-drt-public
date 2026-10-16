"""Administrative-dong context quality report."""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd


def build_context_quality_report(
    national_boundary: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    stop_join: gpd.GeoDataFrame,
    context: gpd.GeoDataFrame,
    profile: gpd.GeoDataFrame,
    source: str,
    source_url: str,
    lineage_summary: dict[str, Any] | None = None,
    authoritative_code_field: str = "ADM_CD",
    authoritative_name_field: str = "ADM_NM",
) -> dict[str, Any]:
    """Summarize boundary, join, context, population and geometry QA."""
    status_counts = stop_join.match_status.value_counts().to_dict()
    warnings = []
    outside = int(status_counts.get("OUTSIDE_BOUNDARY", 0))
    mismatch = int(status_counts.get("MASTER_NAME_MISMATCH", 0))
    missing = int(status_counts.get("MISSING_MASTER_VALUE", 0))
    ambiguous = int(status_counts.get("BOUNDARY_AMBIGUOUS", 0))
    if outside:
        warnings.append(f"{outside} official master stops are outside the Changwon administrative-dong boundary.")
    if mismatch:
        warnings.append(f"{mismatch} spatially assigned stops have a master dong name that differs from the official administrative dong name.")
    if missing:
        warnings.append(f"{missing} master stops have no master_dong_value; spatial assignment is retained with a missing-value QA status.")
    if ambiguous:
        warnings.append(f"{ambiguous} stops intersect multiple polygons and were not assigned.")
    boundary_intersects = int((stop_join.boundary_intersects_candidate_count > 0).sum())
    if boundary_intersects:
        warnings.append(f"{boundary_intersects} within-unmatched stops intersect a boundary edge and remain unassigned for review.")
    assigned_ids = set(stop_join.loc[stop_join.dong_code.notna(), "stop_id"])
    excluded_profile = profile.loc[~profile.stop_id.isin(assigned_ids), ["stop_id", "stop_name", "is_analysis_ready"]]
    if not excluded_profile.empty:
        warnings.append(
            f"{len(excluded_profile)} observed stop is outside the Changwon boundary and is excluded from context totals: "
            + ", ".join(excluded_profile.stop_id.astype(str).tolist())
            + "."
        )
    invalid = int((boundary.geometry.notna() & ~boundary.geometry.is_valid).sum())
    failed = bool(boundary.empty or boundary.dong_code.duplicated().any() or invalid or ambiguous)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "boundary": {
            "source": source,
            "source_url": source_url,
            "national_total_polygons": len(national_boundary),
            "total_polygons": len(boundary),
            "reference_dates": sorted(boundary.reference_date.dropna().astype(str).unique().tolist()),
            "null_geometry": int(boundary.geometry.isna().sum()),
            "invalid_geometry": invalid,
            "duplicate_dong_code": int(boundary.dong_code.duplicated(keep=False).sum()),
            "duplicate_dong_name": int(boundary.dong_name.duplicated(keep=False).sum()),
        },
        "stop_spatial_join": {
            "master_stops": int(stop_join.stop_id.nunique()),
            "spatially_assigned": int(stop_join.dong_code.notna().sum()),
            "matched": int(status_counts.get("MATCH", 0)),
            "master_name_mismatch": mismatch,
            "outside_boundary": outside,
            "boundary_ambiguous": ambiguous,
            "missing_master_value": missing,
            "boundary_intersects_candidates": boundary_intersects,
            "name_comparison": {str(key): int(value) for key, value in stop_join.name_comparison.value_counts().items()},
        },
        "context": {
            "total_dong_rows": len(context),
            "null_master_stop_count": int(context.master_stop_count.isna().sum()),
            "dong_with_zero_stops": int(context.master_stop_count.eq(0).sum()),
            "dong_with_no_observed_stops": int(context.observed_stop_count.eq(0).sum()),
            "dong_with_non_analysis_ready_stops": int(context.duplicate_observation_stop_count.gt(0).sum()),
            "sum_master_stop_count": int(context.master_stop_count.sum()),
            "sum_observed_stop_count": int(context.observed_stop_count.sum()),
            "sum_analysis_ready_stop_count": int(context.analysis_ready_stop_count.sum()),
            "global_observed_stop_count": len(profile),
            "global_analysis_ready_stop_count": int(profile.is_analysis_ready.sum()),
            "excluded_observed_stops": excluded_profile.to_dict("records"),
        },
        "population": {
            "population_source": None,
            "reference_date": None,
            "matched_dong_count": 0,
            "unmatched_dong_count": len(context),
            "status": "NOT_INCLUDED",
        },
        "geometry": {
            "crs": f"EPSG:{context.crs.to_epsg()}",
            "invalid": int((context.geometry.notna() & ~context.geometry.is_valid).sum()),
            "null": int(context.geometry.isna().sum()),
        },
        "warnings": warnings,
        "status": "FAIL" if failed else ("PASS_WITH_WARNINGS" if warnings else "PASS"),
    }
    if lineage_summary:
        report.update(lineage_summary)
    report["authoritative_geography"] = {
        "source": source,
        "code_field": authoritative_code_field,
        "name_field": authoritative_name_field,
        "boundary_reference_date": sorted(boundary.reference_date.dropna().astype(str).unique().tolist()),
        "assignment_method": "point within official administrative-dong polygon",
        "master_dong_value_used_as_join_key": False,
    }
    return report


def write_context_quality_report(report: dict[str, Any], path: Path) -> None:
    """Write context QA as UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
