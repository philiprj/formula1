"""Configuration loader for f1deg."""

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONF_DIR = PROJECT_ROOT / "conf"
DATA_DIR = PROJECT_ROOT / "data"


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a single YAML file."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(
    model_name: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load merged configuration.

    Merges base.yaml + features.yaml + optional model config + runtime overrides.
    """
    config = load_yaml(CONF_DIR / "base.yaml")
    config["features"] = load_yaml(CONF_DIR / "features.yaml")

    if model_name:
        model_path = CONF_DIR / "models" / f"{model_name}.yaml"
        if model_path.exists():
            config["model"] = load_yaml(model_path)
        else:
            raise FileNotFoundError(f"Model config not found: {model_path}")

    if overrides:
        _deep_merge(config, overrides)

    # Resolve data directory
    config["data_dir"] = str(PROJECT_ROOT / config.get("data_dir", "data"))

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, modifying base in place."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base
