"""Machine-readable release manifest generation."""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd


def build_release_manifest(
    profile: gpd.GeoDataFrame,
    version: str,
    master_stop_count: int,
) -> dict[str, Any]:
    """Describe profile readiness, coverage, CRS and source period."""
    status_counts = {str(key): int(value) for key, value in profile["quality_status"].value_counts().items()}
    observed_stop_count = int(profile["stop_id"].nunique())
    ready = int(profile["is_analysis_ready"].sum())
    return {
        "dataset": "changwon_transit_stop_profile",
        "version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "row_count": len(profile),
        "analysis_ready_rows": ready,
        "non_analysis_ready_rows": len(profile) - ready,
        "quality_status_counts": status_counts,
        "crs": f"EPSG:{profile.crs.to_epsg()}",
        "source_period": {
            "start": str(profile["observation_start"].min()),
            "end": str(profile["observation_end"].max()),
        },
        "coverage": {
            "master_stop_count": master_stop_count,
            "observed_stop_count": observed_stop_count,
            "observed_stop_ratio": observed_stop_count / master_stop_count,
        },
    }


def write_release_manifest(manifest: dict[str, Any], path: Path) -> None:
    """Write a UTF-8 JSON release manifest."""
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def add_context_manifest_entry(
    manifest: dict[str, Any],
    context: gpd.GeoDataFrame,
) -> dict[str, Any]:
    """Add the context layer while preserving the existing profile manifest contract."""
    result = dict(manifest)
    result["datasets"] = dict(result.get("datasets", {}))
    result["datasets"]["changwon_transit_context_by_dong"] = {
        "row_count": len(context),
        "crs": f"EPSG:{context.crs.to_epsg()}",
        "geometry_types": sorted(context.geom_type.dropna().unique().tolist()),
        "reference_dates": sorted(context.reference_date.dropna().astype(str).unique().tolist()),
        "sum_master_stop_count": int(context.master_stop_count.sum()),
        "sum_observed_stop_count": int(context.observed_stop_count.sum()),
        "sum_analysis_ready_stop_count": int(context.analysis_ready_stop_count.sum()),
    }
    return result


def add_demographic_manifest_entry(manifest: dict[str, Any], context: gpd.GeoDataFrame) -> dict[str, Any]:
    """Add demographic context lineage and invariant totals."""
    result = dict(manifest)
    result["datasets"] = dict(result.get("datasets", {}))
    result["datasets"]["changwon_transit_demographic_context_by_dong"] = {
        "row_count": len(context), "crs": f"EPSG:{context.crs.to_epsg()}",
        "geometry_types": sorted(context.geom_type.unique().tolist()),
        "population_reference_year": 2024,
        "sum_master_stop_count": int(context.master_stop_count.sum()),
        "sum_observed_stop_count": int(context.observed_stop_count.sum()),
        "sum_analysis_ready_stop_count": int(context.analysis_ready_stop_count.sum()),
    }
    return result
