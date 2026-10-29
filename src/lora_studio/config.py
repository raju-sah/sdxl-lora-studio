"""Centralized configuration loader for ``configs/sdxl_lora.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "sdxl_lora.yaml"


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load YAML configuration from ``config_path`` (defaults to ``configs/sdxl_lora.yaml``)."""
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}
