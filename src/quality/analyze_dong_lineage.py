"""Lineage analysis for outside stops and source locality-name mismatches."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from src.transform.build_dong_context import normalize_admin_name


def classify_outside_regions(
    stop_join: gpd.GeoDataFrame,
    national_boundary: gpd.GeoDataFrame,
    region_prefixes: dict[str, str],
) -> gpd.GeoDataFrame:
    """Locate every Changwon-boundary-outside stop in the national official boundary."""
    outside = stop_join.loc[stop_join.match_status.eq("OUTSIDE_BOUNDARY")].copy()
    polygons = national_boundary.to_crs(outside.crs)[["dong_code", "dong_name", "geometry"]]
    candidates = gpd.sjoin(
        outside.drop(columns=["dong_code", "dong_name", "index_right"], errors="ignore"),
        polygons.rename(columns={"dong_code": "actual_adm_cd", "dong_name": "actual_adm_nm"}),
        how="left", predicate="within",
    )
    counts = candidates.groupby("stop_id")["actual_adm_cd"].count()
    actual = candidates.drop_duplicates("stop_id", keep="first").copy()
    ambiguous_ids = set(counts[counts > 1].index)
    actual.loc[actual.stop_id.isin(ambiguous_ids), ["actual_adm_cd", "actual_adm_nm"]] = pd.NA
    actual["actual_boundary_candidate_count"] = actual.stop_id.map(counts).fillna(0).astype("int64")
    actual["actual_region"] = actual.actual_adm_cd.astype("string").str[:5].map(region_prefixes)
    actual["actual_region"] = actual["actual_region"].fillna(
        actual.actual_adm_cd.notna().map({True: "기타", False: "미해결"})
    )
    return actual


def _numbered_admin_base(name: object) -> str | None:
    """Collapse only an explicit numbered administrative-dong suffix, e.g. 구암1동."""
    normalized = normalize_admin_name(name)
    if normalized is None:
        return None
    return re.sub(r"\d+동$", "동", normalized)


def classify_master_dong_comparison(stop_join: gpd.GeoDataFrame) -> pd.DataFrame:
    """Classify assigned name comparisons conservatively from explicit evidence."""
    assigned = stop_join.loc[stop_join.dong_code.notna()].copy()
    classification = pd.Series("UNRESOLVED_MISMATCH", index=assigned.index, dtype="string")
    classification.loc[assigned.name_comparison.isin(["EXACT", "NORMALIZED"])] = "SAME"
    classification.loc[assigned.name_comparison.eq("MISSING")] = "MISSING_MASTER"
    mismatch = assigned.name_comparison.eq("MISMATCH")
    numbered_split = assigned.master_dong_value.map(normalize_admin_name).eq(
        assigned.dong_name.map(_numbered_admin_base)
    )
    classification.loc[mismatch & numbered_split] = "ADMINISTRATIVE_LEVEL_DIFFERENCE"
    assigned["mismatch_classification"] = classification
    return assigned


def build_lineage_outputs(
    stop_join: gpd.GeoDataFrame,
    national_boundary: gpd.GeoDataFrame,
    region_prefixes: dict[str, str],
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return detailed outside/mismatch records, pair counts and QA summary."""
    outside = classify_outside_regions(stop_join, national_boundary, region_prefixes)
    comparison = classify_master_dong_comparison(stop_join)
    mismatch = comparison.loc[comparison.name_comparison.eq("MISMATCH")].copy()
    pairs = (
        mismatch.groupby(["master_dong_value", "dong_name", "mismatch_classification"], dropna=False)
        .size().rename("count").reset_index()
        .rename(columns={"dong_name": "official_adm_nm"})
        .sort_values(["count", "master_dong_value", "official_adm_nm"], ascending=[False, True, True])
    )
    classification_counts = comparison.mismatch_classification.value_counts()
    ordered_classes = [
        "SAME", "MISSING_MASTER", "ADMINISTRATIVE_LEVEL_DIFFERENCE",
        "HISTORICAL_OR_ALTERNATE_NAME", "UNRESOLVED_MISMATCH",
    ]
    region_counts = outside.actual_region.value_counts()
    ordered_regions = ["창원시 내부", "김해시", "부산광역시", "함안군", "기타", "미해결"]
    summary = {
        "outside_boundary": {
            "total": len(outside),
            "by_actual_region": {key: int(region_counts.get(key, 0)) for key in ordered_regions},
            "adm_cd_prefix_region_mapping": region_prefixes,
            "unresolved": int(outside.actual_region.eq("미해결").sum()),
            "ambiguous_national_boundary": int(outside.actual_boundary_candidate_count.gt(1).sum()),
        },
        "master_dong_comparison": {
            "same": int(comparison.mismatch_classification.eq("SAME").sum()),
            "missing_master": int(comparison.mismatch_classification.eq("MISSING_MASTER").sum()),
            "mismatch": int(comparison.name_comparison.eq("MISMATCH").sum()),
            "mismatch_classification_counts": {key: int(classification_counts.get(key, 0)) for key in ordered_classes},
            "top_mismatch_pairs": pairs.head(20).to_dict("records"),
        },
    }
    return outside, mismatch, pairs, summary


def write_lineage_csvs(
    outside: gpd.GeoDataFrame,
    mismatch: pd.DataFrame,
    pairs: pd.DataFrame,
    report_dir: Path,
) -> None:
    """Write auditable CSV extracts without modifying source or assignments."""
    report_dir.mkdir(parents=True, exist_ok=True)
    outside_csv = pd.DataFrame(outside.drop(columns="geometry"))
    outside_csv["geometry"] = outside.geometry.to_wkt().values
    outside_columns = [
        "stop_id", "stop_name", "direction", "longitude", "latitude",
        "master_dong_value", "geometry", "actual_adm_cd", "actual_adm_nm",
        "actual_region", "actual_boundary_candidate_count",
    ]
    outside_csv[outside_columns].sort_values("stop_id").to_csv(
        report_dir / "outside_boundary_stops.csv", index=False, encoding="utf-8-sig"
    )
    mismatch_columns = [
        "stop_id", "stop_name", "master_dong_value", "dong_code", "dong_name",
        "longitude", "latitude", "mismatch_classification",
    ]
    mismatch[mismatch_columns].rename(
        columns={"dong_code": "official_adm_cd", "dong_name": "official_adm_nm"}
    ).sort_values(["master_dong_value", "official_adm_nm", "stop_id"]).to_csv(
        report_dir / "master_dong_mismatch.csv", index=False, encoding="utf-8-sig"
    )
    pairs.to_csv(report_dir / "master_dong_mismatch_pairs.csv", index=False, encoding="utf-8-sig")
