"""Aggregate official TAGO route schedule data to 행정동 (adm_cd) level.

Join path (official ID / spatial join only, no name-based matching):
  changwon_transit_route_stop.parquet (route_id, stop_id, is_matched)
  -> changwon_drt_stop_features.parquet (stop_id -> adm_cd, canonical spatial join)
  -> changwon_transit_route_schedule.parquet (route_id -> schedule fields)

Produces a NEW, separate release table. Does not modify the canonical
changwon_drt_dong_features.parquet or its score/weights.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    bridge = pd.read_parquet(ROOT / "data/release/changwon_transit_route_stop.parquet")
    stop_features = pd.read_parquet(ROOT / "data/release/changwon_drt_stop_features.parquet", columns=["stop_id", "adm_cd", "adm_nm"])
    schedule = pd.read_parquet(ROOT / "data/release/changwon_transit_route_schedule.parquet")
    dong = pd.read_parquet(ROOT / "data/release/changwon_drt_dong_features.parquet", columns=["adm_cd", "adm_nm"])

    matched = bridge[bridge.is_matched.fillna(False)][["route_id", "stop_id"]].drop_duplicates()
    matched["stop_id"] = matched.stop_id.astype(str)
    stop_features = stop_features.copy()
    stop_features["stop_id"] = stop_features.stop_id.astype(str)

    route_stop_dong = matched.merge(stop_features, on="stop_id", how="inner")[["route_id", "stop_id", "adm_cd"]]
    route_dong = route_stop_dong[["route_id", "adm_cd"]].drop_duplicates()
    unmatched_route_stop_pairs = len(matched) - len(route_stop_dong)

    route_dong_schedule = route_dong.merge(schedule, on="route_id", how="left")

    has_interval = route_dong_schedule.interval_weekday_min.notna()
    grouped = route_dong_schedule.groupby("adm_cd")
    agg = grouped.agg(
        routes_serving_dong=("route_id", "nunique"),
    ).reset_index()

    with_data = route_dong_schedule[has_interval].groupby("adm_cd").agg(
        routes_with_schedule_data=("route_id", "nunique"),
        mean_weekday_interval_min=("interval_weekday_min", "mean"),
        min_weekday_interval_min=("interval_weekday_min", "min"),
        max_weekday_interval_min=("interval_weekday_min", "max"),
        estimated_route_trips_serving_dong_weekday=("estimated_scheduled_trips_weekday", "sum"),
        estimated_route_trips_serving_dong_saturday=("estimated_scheduled_trips_saturday", "sum"),
        estimated_route_trips_serving_dong_sunday=("estimated_scheduled_trips_sunday", "sum"),
    ).reset_index()

    result = dong.merge(agg, on="adm_cd", how="left").merge(with_data, on="adm_cd", how="left")
    result["routes_serving_dong"] = result.routes_serving_dong.astype("Int64")
    result["routes_with_schedule_data"] = result.routes_with_schedule_data.astype("Int64")
    result["routes_missing_schedule_data"] = (result.routes_serving_dong.fillna(0) - result.routes_with_schedule_data.fillna(0)).astype("Int64")
    result["has_any_route"] = result.routes_serving_dong.fillna(0) > 0
    result["has_schedule_data"] = result.routes_with_schedule_data.fillna(0) > 0

    release_path = ROOT / "data/release/changwon_transit_service_intensity_by_dong.parquet"
    result.to_parquet(release_path, index=False)

    qa = {
        "dong_count": len(result),
        "dong_with_any_route": int(result.has_any_route.sum()),
        "dong_with_zero_route": int((~result.has_any_route).sum()),
        "dong_with_schedule_data": int(result.has_schedule_data.sum()),
        "dong_with_route_but_no_schedule_data": int((result.has_any_route & ~result.has_schedule_data).sum()),
        "route_stop_pairs_considered": int(len(matched)),
        "route_stop_pairs_unmatched_to_dong": int(unmatched_route_stop_pairs),
        "distinct_route_dong_pairs": int(len(route_dong)),
        "route_stop_dong_rows_before_deduplication": int(len(route_stop_dong)),
        "route_stop_dong_duplicate_excess_removed": int(len(route_stop_dong) - len(route_dong)),
        "aggregation_grain": "route_id × adm_cd; each route schedule estimate is summed once per dong after dropping duplicate stops",
        "not_stop_events": True,
        "canonical_dong_features_parquet_modified": False,
        "join_method": "OFFICIAL_ID_EXACT (route_id, stop_id); adm_cd from canonical spatial join in changwon_drt_stop_features.parquet, not name matching",
    }
    (ROOT / "reports/transit_service_intensity_by_dong_quality_report.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
