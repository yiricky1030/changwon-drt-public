import pandas as pd
from src.quality.validate_ridership import validate_ridership


def test_negative_ridership_detection():
    raw = pd.DataFrame({"service_date_raw":["2025-08-18"]})
    long = pd.DataFrame({"stop_id":["1"], "service_date":[pd.Timestamp("2025-08-18").date()], "hour":[4], "ridership":pd.Series([-1],dtype="Int64"), "match_status":["EXACT"], "stop_name_raw":["A"], "stop_name_normalized":["A"], "longitude":[128.6], "latitude":[35.2]})
    qa = pd.DataFrame({"match_status":["EXACT"]})
    assert validate_ridership(raw, long, qa)["negative_ridership"] == 1

