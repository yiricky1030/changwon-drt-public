"""Ridership ingestion."""
from pathlib import Path
import pandas as pd
from .csv_loader import read_csv_safe, require_columns


def load_ridership(path: Path) -> tuple[pd.DataFrame, str]:
    """Load ridership data and map actual Korean headers."""
    frame, encoding = read_csv_safe(path)
    require_columns(frame, {"정류장명", "일자", "X좌표", "Y좌표"}, "ridership")
    frame = frame.rename(columns={"정류장명":"stop_name_raw", "일자":"service_date_raw", "X좌표":"longitude", "Y좌표":"latitude"})
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    return frame, encoding
