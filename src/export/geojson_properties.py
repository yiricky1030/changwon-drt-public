"""Strict GeoJSON properties: preserve numbers, arrays, dates and JSON null."""
import json
from datetime import date, datetime
import numpy as np
import pandas as pd
from shapely.geometry import mapping


def json_value(value):
    if isinstance(value, dict): return {str(k):json_value(v) for k,v in value.items()}
    if isinstance(value, (list,tuple,np.ndarray)): return [json_value(v) for v in value]
    if value is None or value is pd.NA or value is pd.NaT: return None
    if isinstance(value, (float,np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value, (bool,np.bool_)): return bool(value)
    if isinstance(value, (int,np.integer)): return int(value)
    if isinstance(value, (datetime,date,pd.Timestamp)): return value.isoformat()
    if pd.isna(value): return None
    return value


def write_properties_geojson(frame, path):
    geometry=frame.geometry.name
    payload={'type':'FeatureCollection','features':[
        {'type':'Feature','properties':{k:json_value(v) for k,v in row.items() if k!=geometry},
         'geometry':mapping(row[geometry]) if row[geometry] is not None else None}
        for _,row in frame.iterrows()]}
    path.write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')),encoding='utf-8')
