"""Wide-to-long transformation preserving missing observations."""
import pandas as pd

HOURS = list(range(4, 24)) + list(range(0, 4))


def resolve_hour_columns(columns: list[object]) -> dict[object, int]:
    """Map padded or unpadded hour headers to integers."""
    mapping = {c: int(str(c).strip()) for c in columns if str(c).strip().isdigit() and int(str(c).strip()) in HOURS}
    missing = sorted(set(HOURS) - set(mapping.values()))
    if missing:
        raise ValueError(f"Missing hour columns: {missing}")
    return mapping


def parse_service_date(series: pd.Series) -> pd.Series:
    """Parse YYYY-MM-DD when a Korean weekday suffix is present."""
    value = series.astype("string").str.extract(r"(\d{4}-\d{2}-\d{2})", expand=False)
    return pd.to_datetime(value, errors="coerce").dt.date


def wide_to_long(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Melt 24 hours and record non-numeric coercions without filling NULL."""
    hour_map = resolve_hour_columns(list(frame.columns))
    work = frame.copy().rename(columns={key: f"hour_{value:02d}" for key, value in hour_map.items()})
    hour_columns = [f"hour_{hour:02d}" for hour in HOURS]
    before = int(work[hour_columns].notna().sum().sum())
    for column in hour_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    after = int(work[hour_columns].notna().sum().sum())
    long = work.melt(id_vars=[c for c in work.columns if c not in hour_columns], value_vars=hour_columns, var_name="hour", value_name="ridership")
    long["hour"] = long["hour"].str.removeprefix("hour_").astype("int64")
    long["service_date"] = parse_service_date(long["service_date_raw"])
    long["ridership"] = long["ridership"].round().astype("Int64")
    return long, {"non_numeric_observation_count": before - after}


def annotate_duplicate_observations(frame: pd.DataFrame) -> pd.DataFrame:
    """Flag duplicate stop/date/hour observations without removing any row."""
    result = frame.copy()
    keys = ["stop_id", "service_date", "hour"]
    eligible = result["stop_id"].notna() & result["service_date"].notna()
    result["duplicate_group_size"] = pd.Series(1, index=result.index, dtype="Int64")
    result.loc[eligible, "duplicate_group_size"] = (
        result.loc[eligible].groupby(keys, dropna=False)["stop_id"].transform("size").astype("Int64")
    )
    result["is_duplicate_observation"] = result["duplicate_group_size"].gt(1)
    return result
