"""Configuration loading utilities for scripts and tests."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a plain dictionary."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
