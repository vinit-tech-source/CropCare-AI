"""CropCare AI — Unit tests for Feature Extraction Module."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.features import (
    extract_all_features,
    extract_color_features,
    extract_features_for_image_path,
    extract_glcm_features,
    extract_hu_moments,
    extract_image_features,
    extract_lbp_features,
    get_feature_names,
)


@pytest.fixture()
def leaf_and_masks() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a synthetic RGB leaf image and its corresponding leaf and lesion masks."""
    img = np.full((128, 128, 3), 128, dtype=np.uint8)
    leaf_mask = np.zeros((128, 128), dtype=np.uint8)
    diseased_mask = np.zeros((128, 128), dtype=np.uint8)

    # Green leaf
    cv2.circle(img, (64, 64), 40, (34, 139, 34), -1)
    cv2.circle(leaf_mask, (64, 64), 40, 255, -1)

    # Brown lesion
    cv2.circle(img, (64, 64), 12, (139, 69, 19), -1)
    cv2.circle(diseased_mask, (64, 64), 12, 255, -1)

    return img, leaf_mask, diseased_mask


class TestFeatureNames:
    """Tests for get_feature_names function."""

    def test_feature_count_is_106(self) -> None:
        """Verify feature count matches the ~105D specification (exactly 106)."""
        names = get_feature_names()
        assert len(names) == 106

    def test_feature_names_are_unique(self) -> None:
        """Verify no duplicate feature names exist."""
        names = get_feature_names()
        assert len(names) == len(set(names))

    def test_feature_names_contain_expected_groups(self) -> None:
        """Verify color, GLCM, LBP, Hu, and severity feature groups are represented."""
        names = get_feature_names()
        assert any(n.startswith("hsv_leaf_") for n in names)
        assert any(n.startswith("hsv_lesion_") for n in names)
        assert any(n.startswith("glcm_") for n in names)
        assert any(n.startswith("lbp_bin_") for n in names)
        assert any(n.startswith("hu_moment_") for n in names)
        assert "disease_severity_pct" in names


class TestColorFeatures:
    """Tests for extract_color_features function."""

    def test_extract_color_features_shape(
        self, leaf_and_masks: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Verify color feature vector dimensionality and names."""
        img, leaf_mask, diseased_mask = leaf_and_masks
        feats, names = extract_color_features(img, leaf_mask, diseased_mask, hsv_bins=8)

        # 24 (leaf hist) + 6 (leaf moments) + 24 (lesion hist) + 6 (lesion moments) = 60
        assert feats.shape == (60,)
        assert len(names) == 60
        assert np.all(np.isfinite(feats))

    def test_color_features_without_lesions(
        self, leaf_and_masks: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Verify healthy leaf with empty lesion mask produces zero lesion features."""
        img, leaf_mask, _ = leaf_and_masks
        empty_lesion_mask = np.zeros_like(leaf_mask)
        feats, _ = extract_color_features(img, leaf_mask, empty_lesion_mask, hsv_bins=8)

        # Lesion features occupy the second half of the 60 features (indices 30 to 59)
        assert np.all(feats[30:] == 0.0)


class TestTextureFeatures:
    """Tests for GLCM and LBP texture extraction."""

    def test_extract_glcm_features(
        self, leaf_and_masks: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Verify GLCM returns 20 features (5 properties x 2 distances x 2 stats)."""
        img, leaf_mask, _ = leaf_and_masks
        feats, names = extract_glcm_features(img, leaf_mask)

        assert feats.shape == (20,)
        assert len(names) == 20
        assert np.all(np.isfinite(feats))

    def test_extract_lbp_features(
        self, leaf_and_masks: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Verify LBP uniform histogram has 18 bins and sums to ~1.0."""
        img, leaf_mask, _ = leaf_and_masks
        feats, names = extract_lbp_features(img, leaf_mask, radius=2, n_points=16)

        assert feats.shape == (18,)
        assert len(names) == 18
        assert np.isclose(np.sum(feats), 1.0, atol=1e-3)
        assert np.all(np.isfinite(feats))


class TestShapeFeatures:
    """Tests for Hu moments extraction."""

    def test_extract_hu_moments_shape(
        self, leaf_and_masks: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Verify Hu moments return 7 log-transformed finite floats."""
        _, leaf_mask, _ = leaf_and_masks
        feats, names = extract_hu_moments(leaf_mask)

        assert feats.shape == (7,)
        assert len(names) == 7
        assert np.all(np.isfinite(feats))

    def test_hu_moments_empty_mask(self) -> None:
        """Verify empty mask produces 7 zeros without error."""
        empty = np.zeros((64, 64), dtype=np.uint8)
        feats, _ = extract_hu_moments(empty)
        assert feats.shape == (7,)
        assert np.all(feats == 0.0)


class TestFullFeaturePipeline:
    """Tests for extract_image_features and extract_features_for_image_path."""

    def test_extract_image_features_vector(
        self, leaf_and_masks: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Verify combined vector has length 106 and matches expected feature names."""
        img, leaf_mask, diseased_mask = leaf_and_masks
        vector = extract_image_features(img, leaf_mask, diseased_mask, severity_pct=15.5)

        assert vector.shape == (106,)
        assert np.all(np.isfinite(vector))
        # The last element is severity_pct
        assert np.isclose(vector[-1], 15.5)

    def test_extract_features_for_image_path(self, tmp_path: Path) -> None:
        """Test file-based feature extraction returns 106D array."""
        img_path = tmp_path / "test_leaf.png"
        img = np.full((128, 128, 3), 128, dtype=np.uint8)
        cv2.circle(img, (64, 64), 40, (34, 139, 34), -1)
        cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        vector = extract_features_for_image_path(img_path)
        assert vector is not None
        assert vector.shape == (106,)

    def test_extract_features_missing_path(self, tmp_path: Path) -> None:
        """Test missing file path returns None."""
        assert extract_features_for_image_path(tmp_path / "not_there.jpg") is None

    def test_feature_extraction_reproducibility(
        self, leaf_and_masks: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Verify multiple extractions from the same input yield identical vectors."""
        img, leaf_mask, diseased_mask = leaf_and_masks
        v1 = extract_image_features(img, leaf_mask, diseased_mask, severity_pct=10.0)
        v2 = extract_image_features(img, leaf_mask, diseased_mask, severity_pct=10.0)
        assert np.array_equal(v1, v2)

    def test_extract_all_features(self, tmp_path: Path) -> None:
        """Test extract_all_features with mock manifest and output CSV."""
        img_file = tmp_path / "mock_leaf.jpg"
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        cv2.circle(img, (32, 32), 20, (34, 139, 34), -1)
        cv2.imwrite(str(img_file), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # Relative path from project root
        manifest_file = tmp_path / "test_manifest.json"
        out_csv = tmp_path / "features.csv"

        manifest_data = {
            "records": [
                {
                    "relative_path": str(img_file),
                    "filename": "mock_leaf.jpg",
                    "crop": "Tomato",
                    "class_name": "Tomato___healthy",
                    "disease": "healthy",
                    "is_healthy": True,
                    "split": "train",
                }
            ]
        }
        manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

        df = extract_all_features(
            manifest_path=manifest_file,
            output_csv=out_csv,
            n_jobs=1,
        )

        assert out_csv.exists()
        assert len(df) == 1
        assert "class_name" in df.columns
        assert "disease_severity_pct" in df.columns
        assert "hsv_leaf_h_bin_0" in df.columns
