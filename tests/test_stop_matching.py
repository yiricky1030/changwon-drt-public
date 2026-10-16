import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from src.transform.match_stops import match_stops
from src.transform.normalize_stop import normalize_stop_master, normalize_stop_name


def test_normalize_stop_name():
    assert normalize_stop_name("  용전.남산마을입구  ") == "용전.남산마을입구"
    assert normalize_stop_name("웅동농협앞\t 정문") == "웅동농협앞 정문"


def test_duplicate_stop_names_are_not_dropped():
    raw = pd.DataFrame({"stop_id":["1","2"], "stop_name_raw":["웅동농협앞","웅동농협앞"], "longitude":[128.8,128.9], "latitude":[35.1,35.2]})
    assert len(normalize_stop_master(raw)) == 2


def test_ambiguous_matching_does_not_auto_select():
    master = gpd.GeoDataFrame({"stop_id":["1","2"], "stop_name":["같은정류장","같은정류장"]}, geometry=[Point(128.6,35.2),Point(128.6,35.2)], crs="EPSG:4326")
    rider = pd.DataFrame({"stop_name_raw":["같은정류장"], "longitude":[128.6], "latitude":[35.2]})
    matched, qa = match_stops(rider, master)
    assert qa.loc[0,"match_status"] == "AMBIGUOUS"
    assert pd.isna(matched.loc[0,"stop_id"])


def test_geometry_creation():
    raw = pd.DataFrame({"stop_id":["1"], "stop_name_raw":["정류장"], "longitude":[128.6], "latitude":[35.2]})
    result = normalize_stop_master(raw)
    assert result.crs.to_epsg() == 4326 and result.geometry.notna().all()

