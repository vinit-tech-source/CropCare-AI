"""
CropCare AI — Configuration loader.

Loads settings from ``configs/config.yaml`` and exposes them as a
module-level dictionary and convenience accessors. All paths are
resolved relative to the project root.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolve project root (two levels up from src/config.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "config.yaml"

# ---------------------------------------------------------------------------
# Load once at import time
# ---------------------------------------------------------------------------
_config: dict[str, Any] | None = None


def _load_config() -> dict[str, Any]:
    """Read and parse the YAML configuration file."""
    global _config  # noqa: PLW0603
    if _config is not None:
        return _config

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {CONFIG_PATH}. "
            "Ensure configs/config.yaml exists in the project root."
        )

    with open(CONFIG_PATH, encoding="utf-8") as fh:
        _config = yaml.safe_load(fh)

    logger.debug("Configuration loaded from %s", CONFIG_PATH)
    return _config


def get_config() -> dict[str, Any]:
    """Return the full configuration dictionary."""
    return _load_config()


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------


def get_random_seed() -> int:
    """Return the global random seed."""
    return int(get_config()["random_seed"])


def get_dataset_config() -> dict[str, Any]:
    """Return the ``dataset`` section of the config."""
    return get_config()["dataset"]


def get_preprocessing_config() -> dict[str, Any]:
    """Return the ``preprocessing`` section of the config."""
    return get_config()["preprocessing"]


def get_segmentation_config() -> dict[str, Any]:
    """Return the ``segmentation`` section of the config."""
    return get_config()["segmentation"]


def get_features_config() -> dict[str, Any]:
    """Return the ``features`` section of the config."""
    return get_config()["features"]


def get_training_config() -> dict[str, Any]:
    """Return the ``training`` section of the config."""
    return get_config()["training"]


def get_prediction_config() -> dict[str, Any]:
    """Return the ``prediction`` section of the config."""
    return get_config()["prediction"]


def get_paths_config() -> dict[str, Any]:
    """Return the ``paths`` section of the config."""
    return get_config()["paths"]


def resolve_path(relative: str) -> Path:
    """Resolve a config-relative path against the project root."""
    return PROJECT_ROOT / relative
