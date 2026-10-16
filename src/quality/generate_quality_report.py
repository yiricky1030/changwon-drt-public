"""Quality report creation."""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_quality_report(stop_metrics: dict[str, Any], ridership_metrics: dict[str, Any], matching_qa: Any, config: dict[str, Any], encodings: dict[str, str], reshape_metrics: dict[str, int], release_metrics: dict[str, int] | None = None) -> dict[str, Any]:
    """Build report and classify only severe usability problems as FAIL."""
    matching_counts = {key:int((matching_qa.match_status == key.upper()).sum()) for key in ("exact", "nearest", "ambiguous", "unmatched")}
    source_unique = len(matching_qa)
    matching = {"source_unique_stops": source_unique, **matching_counts, "match_rate": (matching_counts["exact"] + matching_counts["nearest"]) / max(source_unique, 1)}
    warnings = []
    if stop_metrics["duplicate_stop_id"]: warnings.append("Official stop_id duplicates exist.")
    if ridership_metrics["hour_03_null_count"]: warnings.append(f"{ridership_metrics['null_ridership']} ridership observations are NULL; all occur at hour 03.")
    if ridership_metrics["duplicate_stop_date_hour"]: warnings.append(f"Duplicate source observations exist for {len(ridership_metrics['affected_stops'])} stop names across the observation period; records are preserved with a surrogate record_id.")
    warnings.append("Stop master reference date and ridership reference date differ.")
    if matching["ambiguous"]: warnings.append("Ambiguous stop matches exist and were not auto-selected.")
    if matching["unmatched"]: warnings.append("Unmatched source stops exist.")
    if reshape_metrics["non_numeric_observation_count"]: warnings.append("Non-numeric ridership values were coerced to NULL.")
    invalid_ratio = (stop_metrics["null_geometry"] + stop_metrics["invalid_geometry"]) / max(stop_metrics["row_count"], 1)
    unmatched_ratio = (matching["unmatched"] + matching["ambiguous"]) / max(source_unique, 1)
    failed = bool(stop_metrics["duplicate_stop_id"] or invalid_ratio > config["quality"]["max_invalid_geometry_ratio"] or unmatched_ratio > config["quality"]["max_unmatched_ratio"])
    duplicates = {"duplicate_row_count_keep_false":ridership_metrics["duplicate_stop_date_hour"], "duplicate_extra_row_count":ridership_metrics["duplicate_extra_row_count"], "duplicate_group_count":ridership_metrics["duplicate_group_count"], "affected_stops":ridership_metrics["affected_stops"]}
    temporal = {"date_min":ridership_metrics["date_min"], "date_max":ridership_metrics["date_max"], "unique_dates":ridership_metrics["unique_dates"], "rows_by_date":ridership_metrics["daily_summary"]}
    nulls = {"total_null_ridership":ridership_metrics["null_ridership"], "nulls_by_hour":ridership_metrics["hourly_null_count"]}
    return {"generated_at":datetime.now(timezone.utc).isoformat(), "source_files":{"stop_master":Path(config["paths"]["stop_master"]).name, "ridership":Path(config["paths"]["ridership"]).name, "encodings":encodings}, "stop_master":stop_metrics, "ridership":ridership_metrics, "duplicates":duplicates, "matching":matching, "temporal":temporal, "nulls":nulls, "release":release_metrics or {}, "expected_temporal_coverage":{"start":str(config["expected"]["observation_start"]), "end":str(config["expected"]["observation_end"])}, "reshape":reshape_metrics, "warnings":warnings, "status":"FAIL" if failed else ("PASS_WITH_WARNINGS" if warnings else "PASS")}


def write_quality_report(report: dict[str, Any], path: Path) -> None:
    """Write human-readable UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
