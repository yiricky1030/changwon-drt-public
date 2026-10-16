"""Non-destructive CSV loading with Korean encoding fallbacks."""
from pathlib import Path
import pandas as pd

ENCODINGS = ("utf-8-sig", "utf-8", "cp949", "euc-kr")


def read_csv_safe(path: Path, **kwargs: object) -> tuple[pd.DataFrame, str]:
    """Read a CSV using strict ordered encoding fallbacks."""
    errors = []
    for encoding in ENCODINGS:
        try:
            frame = pd.read_csv(path, encoding=encoding, **kwargs)
            frame.columns = [str(c).strip().lstrip("\ufeff") for c in frame.columns]
            return frame, encoding
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise UnicodeError(f"Cannot decode {path}: {'; '.join(errors)}")


def require_columns(frame: pd.DataFrame, columns: set[str], source: str) -> None:
    """Fail clearly if required semantic columns are absent."""
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}")

