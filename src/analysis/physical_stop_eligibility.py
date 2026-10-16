"""Audited physical-stop eligibility for decision-facing Anchor products."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def attach_physical_eligibility(frame: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Attach explicit ID-based eligibility without changing scores or ranks."""
    audit_path = root / "reports/virtual_stop_integrity_audit.csv"
    if not audit_path.is_file():
        raise FileNotFoundError(f"Run scripts/audit_virtual_stops.py first: {audit_path}")
    audit = pd.read_csv(audit_path, dtype={"stop_id": str})
    lookup = audit.set_index("stop_id")
    result = frame.copy()
    stop_ids = result.stop_id.astype(str)
    result["physical_stop_status"] = stop_ids.map(lookup.physical_stop_status).fillna("PHYSICAL")
    result["anchor_eligible"] = stop_ids.map(lookup.anchor_eligible).fillna(True).astype(bool)
    result["anchor_exclusion_reason"] = stop_ids.map(lookup.anchor_exclusion_reason)
    return result


def eligible_robust(frame: pd.DataFrame) -> pd.Series:
    eligible = frame.anchor_eligible if "anchor_eligible" in frame else True
    return frame.robust_clean & eligible
