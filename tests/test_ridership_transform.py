import pandas as pd
from src.transform.reshape_ridership import HOURS, annotate_duplicate_observations, wide_to_long


def fixture_frame():
    data = {"stop_name_raw":["A"], "service_date_raw":["2025-08-18(월)"], "longitude":[128.6], "latitude":[35.2]}
    data.update({str(hour):[None if hour == 3 else hour] for hour in HOURS})
    return pd.DataFrame(data)


def test_wide_to_long_preserves_null():
    long, _ = wide_to_long(fixture_frame())
    assert pd.isna(long.loc[long.hour == 3, "ridership"]).all()


def test_hour_conversion():
    long, _ = wide_to_long(fixture_frame())
    assert set(long.hour) == set(range(24))
    assert str(long.service_date.iloc[0]) == "2025-08-18"


def test_duplicate_annotation_preserves_all_rows():
    frame = pd.DataFrame({"stop_id":["1", "1", "2"], "service_date":[pd.Timestamp("2025-08-18").date()] * 3, "hour":[5, 5, 5]})
    result = annotate_duplicate_observations(frame)
    assert len(result) == 3
    assert result.is_duplicate_observation.tolist() == [True, True, False]
    assert result.duplicate_group_size.tolist() == [2, 2, 1]
