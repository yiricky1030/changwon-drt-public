"""One-row-per-stop analytical profile."""
import geopandas as gpd
import pandas as pd


def build_stop_profile(master: gpd.GeoDataFrame, fact: pd.DataFrame, bands: dict[str, list[int]]) -> gpd.GeoDataFrame:
    """Aggregate observed values without interpreting NULL as zero."""
    observed = fact.dropna(subset=["stop_id", "service_date"]).copy()
    observed["is_weekend"] = pd.to_datetime(observed.service_date).dt.dayofweek >= 5
    duplicate_mask = observed.duplicated(["stop_id", "service_date", "hour"], keep=False)
    daily = observed.groupby(["stop_id", "service_date"], as_index=False).ridership.sum(min_count=1)
    base = observed.groupby("stop_id").agg(observation_start=("service_date", "min"), observation_end=("service_date", "max"), observation_days=("service_date", "nunique"), total_ridership=("ridership", lambda x:x.sum(min_count=1)), missing_observation_count=("ridership", lambda x:int(x.isna().sum())))
    base["avg_daily_ridership"] = daily.groupby("stop_id").ridership.mean()
    base["weekday_ridership"] = observed[~observed.is_weekend].groupby("stop_id").ridership.sum(min_count=1)
    base["weekend_ridership"] = observed[observed.is_weekend].groupby("stop_id").ridership.sum(min_count=1)
    for name, hours in bands.items():
        base[f"{name}_ridership"] = observed[observed.hour.isin(hours)].groupby("stop_id").ridership.sum(min_count=1)
    hourly = observed.groupby(["stop_id", "hour"], as_index=False).ridership.sum(min_count=1).dropna(subset=["ridership"])
    peaks = hourly.sort_values(["stop_id", "ridership", "hour"], ascending=[True, False, True]).drop_duplicates("stop_id").set_index("stop_id")
    base["peak_hour"] = peaks.hour
    base["peak_hour_ridership"] = peaks.ridership
    duplicate_counts = observed.loc[duplicate_mask].groupby("stop_id").size()
    base["duplicate_observation_count"] = duplicate_counts.reindex(base.index, fill_value=0).astype("int64")
    base["has_duplicate_observations"] = base["duplicate_observation_count"] > 0
    base["quality_status"] = "VALID"
    missing_only = (base["missing_observation_count"] > 0) & ~base["has_duplicate_observations"]
    base.loc[missing_only, "quality_status"] = "MISSING_OBSERVATIONS"
    base.loc[base["has_duplicate_observations"], "quality_status"] = "DUPLICATE_OBSERVATIONS"
    base["is_analysis_ready"] = ~base["has_duplicate_observations"]
    analytical_columns = [
        "total_ridership", "avg_daily_ridership", "weekday_ridership",
        "weekend_ridership", "morning_ridership", "daytime_ridership",
        "evening_ridership", "late_night_ridership", "peak_hour",
        "peak_hour_ridership",
    ]
    base.loc[base["has_duplicate_observations"], analytical_columns] = pd.NA
    columns = ["stop_id", "stop_name", "direction", "longitude", "latitude", "geometry"]
    profile = master[columns].merge(base.reset_index(), on="stop_id", how="inner", validate="one_to_one")
    profile["source"] = "official stop master + matched ridership"
    profile["generated_at"] = pd.Timestamp.now(tz="UTC")
    return gpd.GeoDataFrame(profile, geometry="geometry", crs="EPSG:4326")
