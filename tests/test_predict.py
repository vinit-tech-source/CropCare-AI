"""
CropCare AI — Unit tests for the prediction pipeline and CLI commands.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.predict import (
    DiseasePrediction,
    PredictionResult,
    _find_available_disease_model,
    list_classes,
    predict_crop_disease_and_fertilizer,
    predict_disease,
)


@pytest.fixture()
def sample_leaf_image_file(tmp_path: Path) -> Path:
    """Create a temporary valid RGB leaf image file for testing."""
    # Create green leaf with a small brown spot
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[:, :] = (30, 140, 40)  # Green leaf
    img[100:130, 100:130] = (20, 60, 100)  # Brownish lesion
    target = tmp_path / "test_leaf.jpg"
    cv2.imwrite(str(target), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return target


class TestDiseasePrediction:
    """Tests for disease prediction pipeline."""

    def test_predict_disease_nonexistent_image(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            predict_disease(tmp_path / "not_there.jpg")

    def test_predict_disease_corrupt_image(self, tmp_path: Path) -> None:
        corrupt = tmp_path / "corrupt.jpg"
        corrupt.write_bytes(b"not an image file")
        with pytest.raises(ValueError, match="Failed to read"):
            predict_disease(corrupt)

    def test_predict_disease_fallback_or_trained(self, sample_leaf_image_file: Path) -> None:
        pred = predict_disease(sample_leaf_image_file)
        assert isinstance(pred, DiseasePrediction)
        assert pred.predicted_class != ""
        assert pred.display_name != ""
        assert 0.0 <= pred.confidence <= 1.0
        assert pred.severity_pct >= 0.0
        assert pred.feature_vector is not None
        assert len(pred.feature_vector) == 106


class TestEndToEndPrediction:
    """Tests for complete crop disease diagnosis and fertilizer advisory."""

    def test_predict_crop_disease_and_fertilizer(
        self, sample_leaf_image_file: Path, tmp_path: Path
    ) -> None:
        res = predict_crop_disease_and_fertilizer(
            image_path=sample_leaf_image_file,
            crop="Tomato",
            soil="Loamy",
            N=45.0,
            P=35.0,
            K=25.0,
            temp=28.0,
            humidity=65.0,
            moisture=40.0,
            save_overlay=True,
        )

        assert isinstance(res, PredictionResult)
        assert res.image_path == str(sample_leaf_image_file)
        assert res.disease is not None
        assert res.recommendation is not None
        assert res.recommendation.crop == "Tomato"
        assert res.recommendation.soil_type == "Loamy"

        # Check serialization
        d = res.to_dict()
        assert isinstance(d, dict)
        assert "image_path" in d
        assert "disease" in d
        assert "recommendation" in d


class TestListClasses:
    """Tests for list_classes CLI helper."""

    def test_list_classes_runs(self) -> None:
        # Should execute cleanly without exceptions
        list_classes()


class TestModelFinder:
    """Tests for disease model file scanner."""

    def test_find_available_disease_model(self) -> None:
        # Returns either None or (model_name, Path)
        res = _find_available_disease_model()
        if res is not None:
            name, path = res
            assert isinstance(name, str)
            assert isinstance(path, Path)
