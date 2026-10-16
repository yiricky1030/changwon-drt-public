"""Ridership quality checks."""
import pandas as pd


def validate_ridership(raw: pd.DataFrame, long: pd.DataFrame, matching_qa: pd.DataFrame) -> dict[str, object]:
    """Summarize temporal, value, duplicate and matching quality."""
    dates = long.service_date.dropna()
    eligible = long.dropna(subset=["stop_id", "service_date"])
    duplicates = eligible.duplicated(["stop_id", "service_date", "hour"], keep=False)
    duplicate_extra = eligible.duplicated(["stop_id", "service_date", "hour"], keep="first")
    duplicate_rows = eligible.loc[duplicates]
    duplicate_groups = duplicate_rows.groupby(["stop_id", "service_date", "hour"], dropna=False).ngroups
    affected = (
        duplicate_rows[["stop_id", "stop_name_normalized"]]
        .drop_duplicates()
        .rename(columns={"stop_name_normalized": "stop_name"})
        .sort_values(["stop_name", "stop_id"])
        .to_dict("records")
    )
    hourly_null = long.groupby("hour").ridership.apply(lambda x: int(x.isna().sum())).to_dict()
    raw_dates = pd.to_datetime(raw.service_date_raw.astype("string").str.extract(r"(\d{4}-\d{2}-\d{2})", expand=False), errors="coerce").dt.date
    daily = []
    for date in sorted(dates.unique()):
        day = long[long.service_date == date]
        daily.append({"service_date":str(date), "raw_stop_rows":int((raw_dates == date).sum()), "matched_stop_count":int(day.loc[day.match_status.isin(["EXACT", "NEAREST"]), "stop_id"].nunique()), "unmatched_count":int(day.loc[day.match_status == "UNMATCHED", ["stop_name_raw", "longitude", "latitude"]].drop_duplicates().shape[0]), "total_ridership":int(day.ridership.sum(skipna=True)), "null_observation_count":int(day.ridership.isna().sum())})
    return {"raw_row_count":len(raw), "long_row_count":len(long), "invalid_service_date":int(long.service_date.isna().sum()), "invalid_hour":int((~long.hour.between(0, 23)).sum()), "date_min":str(dates.min()) if not dates.empty else None, "date_max":str(dates.max()) if not dates.empty else None, "unique_dates":int(dates.nunique()), "null_ridership":int(long.ridership.isna().sum()), "negative_ridership":int((long.ridership < 0).sum()), "duplicate_stop_date_hour":int(duplicates.sum()), "duplicate_extra_row_count":int(duplicate_extra.sum()), "duplicate_group_count":int(duplicate_groups), "affected_stops":affected, "hourly_null_count":{str(k):v for k, v in hourly_null.items()}, "hour_03_null_count":int(hourly_null.get(3, 0)), "daily_summary":daily, "identical_stop_name_multiple_stop_id":int(long.dropna(subset=["stop_id"]).groupby("stop_name_normalized").stop_id.nunique().gt(1).sum()), "ambiguous_source_stops":int((matching_qa.match_status == "AMBIGUOUS").sum()), "unmatched_source_stops":int((matching_qa.match_status == "UNMATCHED").sum())}
