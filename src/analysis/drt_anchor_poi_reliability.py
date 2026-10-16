"""Official POI reliability and primary POI context classification for
robust_clean network Anchor stops. This is an internal analysis diagnostic on
how usable the current POI context is, not a policy suitability
completeness grade, and not a policy suitability rating."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

PRIMARY_CATEGORIES = ("HEALTHCARE", "ADMINISTRATION")

CATEGORY_RELIABILITY = {
    "HEALTHCARE": {"poi_reliability_class": "CONTEXT_USABLE",
        "poi_reliability_reason": "HIRA 공식 병원정보 API 수집 성공 — 공식 source 기반 context로 재평가 가능"},
    "ADMINISTRATION": {"poi_reliability_class": "CONTEXT_USABLE",
        "poi_reliability_reason": "MOIS 공식 읍면동 하부행정기관 55건 수집 성공 — 공식 source 기반 context로 재평가 가능"},
}

OFFICIAL_SOURCE_STATUS = "OFFICIAL_SOURCE_AVAILABLE"


def poi_with_reliability(poi: pd.DataFrame) -> pd.DataFrame:
    frame = poi.copy()
    frame["poi_reliability_class"] = frame.poi_category.map(lambda c: CATEGORY_RELIABILITY[c]["poi_reliability_class"])
    frame["poi_reliability_reason"] = frame.poi_category.map(lambda c: CATEGORY_RELIABILITY[c]["poi_reliability_reason"])
    frame["official_source_available"] = ~frame.source_name.eq("OpenStreetMap") if "source_name" in frame else False
    frame["official_source_status"] = "OFFICIAL_SOURCE_AVAILABLE"
    return frame


def anchor_primary_context(context: pd.DataFrame) -> pd.DataFrame:
    frame = context.copy()
    healthcare_near, admin_near = frame.nearest_healthcare_walk_m, frame.nearest_administration_walk_m
    frame["nearest_primary_poi_walk_m"] = pd.concat([healthcare_near, admin_near], axis=1).min(axis=1, skipna=True)
    for radius in (500, 800, 1200):
        frame[f"primary_poi_within_walk_{radius}m"] = (
            frame[f"healthcare_within_walk_{radius}m"].fillna(0) + frame[f"administration_within_walk_{radius}m"].fillna(0))
    frame["has_healthcare_within_800m"] = frame.healthcare_within_walk_800m.fillna(0).gt(0)
    frame["has_administration_within_800m"] = frame.administration_within_walk_800m.fillna(0).gt(0)
    frame["primary_category_count_walk_800m"] = frame.has_healthcare_within_800m.astype(int) + frame.has_administration_within_800m.astype(int)

    unreachable = ~frame.network_reachable_flag
    present = frame.primary_category_count_walk_800m.ge(1)
    distant = ~present & frame.nearest_primary_poi_walk_m.notna() & frame.nearest_primary_poi_walk_m.le(1200)
    no_context = ~present & (frame.nearest_primary_poi_walk_m.isna() | frame.nearest_primary_poi_walk_m.gt(1200))
    frame["anchor_context_class"] = np.select(
        [unreachable, present, distant, no_context],
        ["NETWORK_UNREACHABLE_TO_PRIMARY_POI", "PRIMARY_CONTEXT_PRESENT", "PRIMARY_CONTEXT_DISTANT", "NO_PRIMARY_CONTEXT_WITHIN_1200M"],
        default="NO_PRIMARY_CONTEXT_WITHIN_1200M")
    return frame


def primary_context_summary(frame: pd.DataFrame) -> dict[str, Any]:
    nearest = frame.nearest_primary_poi_walk_m.dropna()
    return {"anchors": int(len(frame)),
            "healthcare_within800": int(frame.has_healthcare_within_800m.sum()),
            "administration_within800": int(frame.has_administration_within_800m.sum()),
            "either_primary_within800": int(frame.primary_category_count_walk_800m.ge(1).sum()),
            "either_primary_within1200": int((frame.primary_poi_within_walk_1200m.gt(0)).sum()),
            "no_primary_within1200": int(frame.anchor_context_class.eq("NO_PRIMARY_CONTEXT_WITHIN_1200M").sum()),
            "nearest_primary_min": float(nearest.min()) if len(nearest) else None,
            "nearest_primary_median": float(nearest.median()) if len(nearest) else None,
            "nearest_primary_p95": float(nearest.quantile(.95)) if len(nearest) else None,
            "nearest_primary_max": float(nearest.max()) if len(nearest) else None,
            "context_class_distribution": frame.anchor_context_class.value_counts().to_dict()}


def dong_pattern(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for adm_cd, group in frame.groupby("adm_cd"):
        rows.append({"adm_cd": adm_cd, "adm_nm": group.adm_nm.iloc[0], "robust_clean_anchor_count": int(len(group)),
                     "primary_context_present": int(group.anchor_context_class.eq("PRIMARY_CONTEXT_PRESENT").sum()),
                     "primary_context_distant": int(group.anchor_context_class.eq("PRIMARY_CONTEXT_DISTANT").sum()),
                     "no_primary_within1200": int(group.anchor_context_class.eq("NO_PRIMARY_CONTEXT_WITHIN_1200M").sum())})
    return rows
