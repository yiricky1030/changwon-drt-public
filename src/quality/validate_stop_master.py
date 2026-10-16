"""Stop-master quality checks."""
import geopandas as gpd


def validate_stop_master(frame: gpd.GeoDataFrame) -> dict[str, object]:
    """Return lossless master-data quality metrics."""
    valid_range = frame.longitude.between(124, 132) & frame.latitude.between(33, 39)
    coordinate_groups = frame.groupby(["longitude", "latitude"], dropna=False).size()
    name_coordinates = frame.groupby("stop_name", dropna=False)[["longitude", "latitude"]].apply(lambda x: len(x.drop_duplicates()))
    return {"row_count":len(frame), "unique_stop_id":int(frame.stop_id.nunique(dropna=True)), "null_stop_id":int(frame.stop_id.isna().sum()), "duplicate_stop_id":int(frame.stop_id.duplicated(keep=False).sum()), "null_stop_name":int(frame.stop_name.isna().sum()), "null_longitude":int(frame.longitude.isna().sum()), "null_latitude":int(frame.latitude.isna().sum()), "invalid_coordinate_range":int((~valid_range & frame.longitude.notna() & frame.latitude.notna()).sum()), "null_geometry":int(frame.geometry.isna().sum()), "invalid_geometry":int((frame.geometry.notna() & ~frame.geometry.is_valid).sum()), "duplicate_coordinate_rows":int(coordinate_groups[coordinate_groups > 1].sum()), "duplicate_stop_names":int(frame.stop_name.duplicated(keep=False).sum()), "same_name_different_coordinates":int((name_coordinates > 1).sum()), "same_coordinate_different_stop_id":int(frame.groupby(["longitude", "latitude"], dropna=False).stop_id.nunique().gt(1).sum())}
