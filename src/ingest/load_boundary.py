"""Official census administrative-dong boundary ingestion."""
from pathlib import Path

import geopandas as gpd


def load_dong_boundary(
    path: Path,
    layer: str,
    code_field: str,
    name_field: str,
    reference_date_field: str,
) -> gpd.GeoDataFrame:
    """Load the verified SHP layer from its immutable ZIP archive."""
    frame = gpd.read_file(f"zip://{path}", layer=layer)
    required = {code_field, name_field, reference_date_field, "geometry"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Boundary missing verified columns: {missing}")
    if frame.crs is None:
        raise ValueError("Boundary CRS is missing")
    return frame.rename(
        columns={
            code_field: "dong_code",
            name_field: "dong_name",
            reference_date_field: "reference_date_raw",
        }
    )


def extract_changwon_boundary(
    boundary: gpd.GeoDataFrame,
    sigungu_prefixes: list[str],
) -> gpd.GeoDataFrame:
    """Filter Changwon by verified official five-digit code prefixes."""
    prefixes = boundary["dong_code"].astype("string").str[:5]
    result = boundary[prefixes.isin(sigungu_prefixes)].copy()
    result["reference_date"] = result["reference_date_raw"].astype("string")
    return result.to_crs("EPSG:4326")
