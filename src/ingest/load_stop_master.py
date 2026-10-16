"""Official stop-master ingestion."""
from pathlib import Path
import pandas as pd
from .csv_loader import read_csv_safe, require_columns

COLUMN_MAP = {"정류소ID":"stop_id", "서비스ID":"service_id", "정류소명":"stop_name_raw", "방면":"direction", "X좌표":"longitude", "Y좌표":"latitude", "동코드":"master_dong_value", "수정일":"source_modified_at", "생성일":"source_created_at"}


def load_stop_master(path: Path) -> tuple[pd.DataFrame, str]:
    """Load all official rows without deduplication."""
    frame, encoding = read_csv_safe(path, dtype={"정류소ID":"string", "서비스ID":"string", "동코드":"string"})
    require_columns(frame, {"정류소ID", "정류소명", "X좌표", "Y좌표"}, "stop master")
    frame = frame.rename(columns=COLUMN_MAP)
    for column in ("longitude", "latitude"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("source_created_at", "source_modified_at"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.date
    return frame, encoding
