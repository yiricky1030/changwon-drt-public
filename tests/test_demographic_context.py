import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from src.export.release_manifest import add_demographic_manifest_entry
from src.transform.build_demographic_context import build_demographic_context


def sample_context():
    return gpd.GeoDataFrame({
        "dong_code": ["38111110"], "dong_name": ["동읍"], "reference_date": ["20250630"],
        "master_stop_count": [10], "observed_stop_count": [4], "analysis_ready_stop_count": [3],
        "total_ridership": [100], "source": ["boundary"], "dataset_version": ["test"],
        "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
    }, crs="EPSG:4326")


def test_population_join_uses_official_code():
    population = pd.DataFrame({"adm_cd": ["38111110"], "population_total": [1000], "population_65_plus": [200]})
    result = build_demographic_context(sample_context(), population)
    assert result.loc[0, "population_total"] == 1000


def test_population_zero_returns_null_metrics():
    population = pd.DataFrame({"adm_cd": ["38111110"], "population_total": [0], "population_65_plus": [0]})
    result = build_demographic_context(sample_context(), population)
    assert pd.isna(result.loc[0, "elderly_ratio"])
    assert pd.isna(result.loc[0, "stops_per_1000_population"])


def test_demographic_enrichment_preserves_context_counts():
    population = pd.DataFrame({"adm_cd": ["38111110"], "population_total": [1000], "population_65_plus": [200]})
    result = build_demographic_context(sample_context(), population)
    assert result.master_stop_count.sum() == 10
    assert result.observed_stop_count.sum() == 4


def test_elderly_not_greater_than_total():
    population = pd.DataFrame({"adm_cd": ["38111110"], "population_total": [1000], "population_65_plus": [200]})
    result = build_demographic_context(sample_context(), population)
    assert not (result.population_65_plus > result.population_total).any()


def test_demographic_manifest_entry():
    population = pd.DataFrame({"adm_cd": ["38111110"], "population_total": [1000], "population_65_plus": [200]})
    result = build_demographic_context(sample_context(), population)
    manifest = add_demographic_manifest_entry({}, result)
    assert manifest["datasets"]["changwon_transit_demographic_context_by_dong"]["row_count"] == 1
