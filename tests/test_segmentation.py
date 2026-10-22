"""CropCare AI — Unit tests for Leaf & Disease Segmentation Module."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.segmentation import (
    extract_leaf_mask,
    generate_elbow_silhouette_plot,
    segment_leaf_and_disease,
)


@pytest.fixture()
def synthetic_diseased_leaf() -> np.ndarray:
    """Create a synthetic 128x128 image with a green leaf and a brown lesion on a gray background."""
    img = np.full((128, 128, 3), 128, dtype=np.uint8)  # Neutral gray background

    # Draw green leaf in center
    cv2.circle(img, (64, 64), 45, (34, 139, 34), -1)

    # Draw brown disease lesion inside the leaf
    cv2.circle(img, (64, 64), 15, (139, 69, 19), -1)

    return img


class TestExtractLeafMask:
    """Tests for extract_leaf_mask function."""

    def test_leaf_mask_detects_green_leaf(self, synthetic_diseased_leaf: np.ndarray) -> None:
        """Verify leaf mask isolates the green leaf circle."""
        mask = extract_leaf_mask(synthetic_diseased_leaf)
        assert mask.shape == (128, 128)
        assert mask.dtype == np.uint8
        # Center should be foreground (255)
        assert mask[64, 64] == 255
        # Corner background should be 0
        assert mask[5, 5] == 0

    def test_leaf_mask_fallback_on_uniform_image(self) -> None:
        """Verify fallback triggers on image without chromatic contrast."""
        uniform = np.full((64, 64, 3), 200, dtype=np.uint8)
        mask = extract_leaf_mask(uniform)
        assert mask.shape == (64, 64)
        assert mask.dtype == np.uint8


class TestSegmentLeafAndDisease:
    """Tests for segment_leaf_and_disease function."""

    def test_segmentation_outputs_and_types(self, synthetic_diseased_leaf: np.ndarray) -> None:
        """Verify return tuple structure, shapes, and value ranges."""
        leaf_mask, diseased_mask, overlay, severity_pct = segment_leaf_and_disease(
            synthetic_diseased_leaf, k=3, random_state=42
        )

        assert leaf_mask.shape == (128, 128)
        assert diseased_mask.shape == (128, 128)
        assert overlay.shape == (128, 128, 3)
        assert isinstance(severity_pct, float)
        assert 0.0 <= severity_pct <= 100.0

        # Lesions should be identified inside the leaf
        assert np.any(diseased_mask > 0)
        # Diseased mask should be a subset of the leaf mask
        overlap = (diseased_mask > 0) & (leaf_mask == 0)
        assert np.sum(overlap) == 0

    def test_segmentation_on_empty_image(self) -> None:
        """Verify graceful handling of blank/black image."""
        black_img = np.zeros((64, 64, 3), dtype=np.uint8)
        leaf_mask, diseased_mask, overlay, severity_pct = segment_leaf_and_disease(black_img)

        assert leaf_mask.shape == (64, 64)
        assert diseased_mask.shape == (64, 64)
        assert severity_pct == 0.0

    def test_overlay_modifies_lesions(self, synthetic_diseased_leaf: np.ndarray) -> None:
        """Verify overlay image highlights lesion regions."""
        _, diseased_mask, overlay, _ = segment_leaf_and_disease(
            synthetic_diseased_leaf, k=3, random_state=42
        )
        if np.any(diseased_mask > 0):
            # Overlay should differ from original image in lesion areas
            diff = np.abs(overlay.astype(int) - synthetic_diseased_leaf.astype(int))
            assert np.sum(diff) > 0


class TestElbowSilhouettePlot:
    """Tests for generate_elbow_silhouette_plot function."""

    def test_generate_plot(self, synthetic_diseased_leaf: np.ndarray, tmp_path: Path) -> None:
        """Verify elbow and silhouette plot generates valid figure and scores."""
        out_file = tmp_path / "elbow_test.png"
        res = generate_elbow_silhouette_plot(
            images_rgb=[synthetic_diseased_leaf],
            max_k=4,
            sample_pixels=500,
            output_path=out_file,
        )

        assert out_file.exists()
        assert out_file.stat().st_size > 0
        assert res["k_values"] == [2, 3, 4]
        assert len(res["inertias"]) == 3
        assert len(res["silhouettes"]) == 3
        assert res["selected_k"] == 3
