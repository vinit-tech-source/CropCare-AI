"""
CropCare AI — Shared utilities.

Logging setup, random seed management, path helpers, and common I/O.
"""

from __future__ import annotations

import contextlib
import json
import logging
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.config import PROJECT_ROOT, get_random_seed

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a consistent format.

    Parameters
    ----------
    level : int
        Logging level (default ``logging.INFO``).
    """
    if hasattr(sys.stdout, "reconfigure"):
        with contextlib.suppress(Exception):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        datefmt=_LOG_DATE_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_global_seed(seed: int | None = None) -> int:
    """Set random seeds for reproducibility.

    Parameters
    ----------
    seed : int | None
        Seed value. If *None*, read from ``config.yaml``.

    Returns
    -------
    int
        The seed that was set.
    """
    if seed is None:
        seed = get_random_seed()

    random.seed(seed)
    np.random.seed(seed)

    logging.getLogger(__name__).debug("Global random seed set to %d", seed)
    return seed


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it does not exist.

    Returns
    -------
    Path
        The same *path* for chaining.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def project_path(*parts: str) -> Path:
    """Build an absolute path relative to the project root.

    Example
    -------
    >>> project_path("data", "raw")
    PosixPath('/…/CropCare AI/data/raw')
    """
    return PROJECT_ROOT.joinpath(*parts)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def save_json(data: Any, path: Path, indent: int = 2) -> None:
    """Serialise *data* to a JSON file, creating parent dirs as needed."""
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, ensure_ascii=False, default=str)
    logging.getLogger(__name__).info("Saved JSON → %s", path)


def load_json(path: Path) -> Any:
    """Load and return a JSON file."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------


def timestamp_str() -> str:
    """Return a filesystem-safe UTC timestamp string, e.g. ``20241021_163005``."""
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
