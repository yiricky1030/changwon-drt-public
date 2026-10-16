"""Build the code-joined SGIS demographic context layer."""
from __future__ import annotations

import geopandas as gpd
import pandas as pd


def safe_ratio(numerator: pd.Series, denominator: pd.Series, scale: float = 1.0) -> pd.Series:
    """Divide only by positive observed population; never impute zero."""
    return numerator / denominator.where(denominator.gt(0)) * scale


def build_demographic_context(context: gpd.GeoDataFrame, population: pd.DataFrame) -> gpd.GeoDataFrame:
    """Enrich by authoritative eight-digit code and preserve context columns."""
    pop = population.rename(columns={"adm_cd": "dong_code"}).copy()
    result = context.merge(
        pop[["dong_code", "population_total", "population_65_plus"]],
        on="dong_code", how="left", validate="one_to_one",
    )
    result["elderly_ratio"] = safe_ratio(result.population_65_plus, result.population_total)
    result["stops_per_1000_population"] = safe_ratio(result.master_stop_count, result.population_total, 1000)
    result["observed_stops_per_1000_population"] = safe_ratio(result.observed_stop_count, result.population_total, 1000)
    result["analysis_ready_stops_per_1000_population"] = safe_ratio(result.analysis_ready_stop_count, result.population_total, 1000)
    result["ridership_per_1000_population"] = safe_ratio(result.total_ridership, result.population_total, 1000)
    result["population_reference_year"] = 2024
    result["population_reference_date"] = pd.NaT
    result["boundary_reference_date"] = pd.to_datetime(result.pop("reference_date")).dt.date
    result["transit_reference_start"] = pd.Timestamp("2025-08-18").date()
    result["transit_reference_end"] = pd.Timestamp("2025-08-25").date()
    result["source"] = result.source.astype(str) + "; 통계청 SGIS 인구주택총조사 2024"
    return gpd.GeoDataFrame(result, geometry="geometry", crs=context.crs)
