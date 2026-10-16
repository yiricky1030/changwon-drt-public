"""Configuration loading and project-relative path resolution."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load YAML configuration and resolve all paths from the project root."""
    project_root = Path(__file__).resolve().parents[1]
    config_path = path or project_root / "config" / "settings.yaml"
    load_dotenv(project_root / ".env")
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["project_root"] = project_root
    for key in ("stop_master", "ridership", "processed_dir", "release_dir", "report", "boundary_zip", "context_report", "database_report", "demographic_report", "sgis_population_processed"):
        config["paths"][key] = (project_root / config["paths"][key]).resolve()
    return config
