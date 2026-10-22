"""
CropCare AI — Shared test fixtures.
"""

from __future__ import annotations

import os
from pathlib import Path

# Silence loky CPU detection warning on Windows
if "LOKY_MAX_CPU_COUNT" not in os.environ:
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

import numpy as np
import pytest


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for test outputs."""
    return tmp_path


@pytest.fixture()
def sample_image_rgb() -> np.ndarray:
    """Return a small synthetic 64×64 RGB image for unit tests.

    This is NOT real data — it is used only for testing code paths.
    """
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)


@pytest.fixture()
def sample_gray_image() -> np.ndarray:
    """Return a small synthetic 64×64 grayscale image."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, size=(64, 64), dtype=np.uint8)


@pytest.fixture()
def sample_feature_vector() -> np.ndarray:
    """Return a synthetic feature vector (~105 dimensions, exactly 106)."""
    rng = np.random.RandomState(42)
    return rng.rand(106)


@pytest.fixture()
def mock_plantvillage_dir(tmp_path: Path) -> Path:
    """Create a mock PlantVillage directory structure for subset testing.

    Creates tiny 4×4 JPEG-like files (not valid images, but enough to
    test the file-copy logic).
    """
    train_dir = tmp_path / "New Plant Diseases Dataset(Augmented)" / "train"
    valid_dir = tmp_path / "New Plant Diseases Dataset(Augmented)" / "valid"

    classes = [
        "Apple___Apple_scab",
        "Apple___healthy",
        "Corn_(maize)___Common_rust_",
        "Corn_(maize)___healthy",
        "Potato___Early_blight",
        "Potato___healthy",
        "Tomato___Early_blight",
        "Tomato___healthy",
        "Grape___healthy",  # Should be skipped (not a target crop)
    ]

    for split_dir in (train_dir, valid_dir):
        for cls in classes:
            cls_dir = split_dir / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            # Create tiny dummy files
            for i in range(10):
                (cls_dir / f"img_{i:04d}.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 100)

    return tmp_path
