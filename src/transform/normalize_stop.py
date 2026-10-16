"""Conservative stop-name and geometry normalization."""
import re
import unicodedata
import geopandas as gpd
import pandas as pd


def normalize_stop_name(value: object) -> str | None:
    """Normalize Unicode and whitespace only."""
    if pd.isna(value):
        return None
    text = unicodedata.normalize("NFC", str(value)).strip()
    return re.sub(r"\s+", " ", text) or None


def normalize_stop_master(frame: pd.DataFrame) -> gpd.GeoDataFrame:
    """Create canonical names and EPSG:4326 point geometry."""
    result = frame.copy()
    result["stop_name"] = result["stop_name_raw"].map(normalize_stop_name)
    valid = result["longitude"].notna() & result["latitude"].notna()
    geometry = gpd.GeoSeries([None] * len(result), index=result.index, crs="EPSG:4326")
    geometry.loc[valid] = gpd.points_from_xy(result.loc[valid, "longitude"], result.loc[valid, "latitude"])
    result = gpd.GeoDataFrame(result, geometry=geometry, crs="EPSG:4326")
    result["source"] = "경상남도 창원시 버스정류소 위치정보 (2024-12-31)"
    result["loaded_at"] = pd.Timestamp.now(tz="UTC")
    return result

