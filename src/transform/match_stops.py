"""Name-constrained, meter-based stop matching."""
import numpy as np
import geopandas as gpd
import pandas as pd
from .normalize_stop import normalize_stop_name


def match_stops(ridership: pd.DataFrame, master: gpd.GeoDataFrame, tolerance_m: float = 10.0, projected_crs: str = "EPSG:5179") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match unique source stops and preserve ambiguous candidates."""
    work = ridership.copy()
    work["stop_name_normalized"] = work["stop_name_raw"].map(normalize_stop_name)
    keys = work[["stop_name_raw", "stop_name_normalized", "longitude", "latitude"]].drop_duplicates().reset_index(drop=True)
    points = gpd.GeoDataFrame(keys, geometry=gpd.points_from_xy(keys.longitude, keys.latitude), crs="EPSG:4326").to_crs(projected_crs)
    master_work = master.copy()
    if "longitude" not in master_work.columns:
        master_work["longitude"] = master_work.geometry.x
    if "latitude" not in master_work.columns:
        master_work["latitude"] = master_work.geometry.y
    projected_master = master_work.to_crs(projected_crs)
    rows = []
    for _, row in points.iterrows():
        candidates = projected_master[projected_master["stop_name"] == row.stop_name_normalized].copy()
        candidates["distance_m"] = candidates.geometry.distance(row.geometry)
        exact = candidates[
            np.isclose(candidates["longitude"], row.longitude, rtol=0.0, atol=1e-12)
            & np.isclose(candidates["latitude"], row.latitude, rtol=0.0, atol=1e-12)
        ]
        eligible = exact if not exact.empty else candidates[candidates["distance_m"] <= tolerance_m]
        status = "UNMATCHED" if eligible.empty else ("AMBIGUOUS" if len(eligible) > 1 else ("EXACT" if not exact.empty else "NEAREST"))
        chosen = eligible.iloc[0] if len(eligible) == 1 else None
        rows.append({"ridership_stop_name":row.stop_name_raw, "ridership_longitude":row.longitude, "ridership_latitude":row.latitude, "matched_stop_id":None if chosen is None else chosen.stop_id, "matched_stop_name":None if chosen is None else chosen.stop_name, "distance_m":None if chosen is None else float(chosen.distance_m), "match_status":status, "candidate_count":len(eligible)})
    qa = pd.DataFrame(rows)
    merged = work.merge(qa, left_on=["stop_name_raw", "longitude", "latitude"], right_on=["ridership_stop_name", "ridership_longitude", "ridership_latitude"], how="left", validate="many_to_one")
    merged["stop_id"] = merged["matched_stop_id"]
    return merged, qa
