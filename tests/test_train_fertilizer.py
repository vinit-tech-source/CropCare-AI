"""
CropCare AI — Unit tests for fertilizer training module.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.train_fertilizer import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    build_fertilizer_pipeline,
    load_fertilizer_data,
    load_fertilizer_pipeline,
    train_fertilizer_model,
)


@pytest.fixture()
def sample_fertilizer_csv(tmp_path: Path) -> Path:
    """Create a small synthetic fertilizer CSV for isolated tests."""
    data = {
        "Temparature": [26, 29, 34, 32, 28, 26, 25, 33, 30, 29],
        "Humidity ": [52, 52, 65, 62, 54, 52, 50, 64, 60, 58],
        "Moisture": [38, 45, 62, 34, 46, 35, 64, 50, 42, 33],
        "Soil Type": ["Sandy", "Loamy", "Black", "Red", "Clayey"] * 2,
        "Crop Type": ["Maize", "Sugarcane", "Cotton", "Tobacco", "Paddy"] * 2,
        "Nitrogen": [37, 12, 7, 22, 35, 12, 9, 41, 21, 9],
        "Potassium": [0, 0, 9, 0, 0, 10, 0, 0, 0, 7],
        "Phosphorous": [0, 36, 30, 20, 0, 13, 10, 0, 18, 30],
        "Fertilizer Name": ["Urea", "DAP", "14-35-14", "28-28", "Urea"] * 2,
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "synthetic_fertilizer.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


class TestLoadFertilizerData:
    """Tests for load_fertilizer_data function."""

    def test_load_valid_csv(self, sample_fertilizer_csv: Path) -> None:
        X, y = load_fertilizer_data(sample_fertilizer_csv)
        assert len(X) == 10
        assert len(y) == 10
        for col in NUMERICAL_FEATURES + CATEGORICAL_FEATURES:
            assert col in X.columns
        assert TARGET_COLUMN not in X.columns

    def test_load_nonexistent_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_fertilizer_data(tmp_path / "non_existent.csv")

    def test_load_missing_columns(self, tmp_path: Path) -> None:
        bad_df = pd.DataFrame({"Temperature": [25], "Crop Type": ["Maize"]})
        bad_path = tmp_path / "bad.csv"
        bad_df.to_csv(bad_path, index=False)
        with pytest.raises(ValueError, match="Missing required columns"):
            load_fertilizer_data(bad_path)


class TestBuildFertilizerPipeline:
    """Tests for build_fertilizer_pipeline function."""

    def test_pipeline_steps(self) -> None:
        pipe = build_fertilizer_pipeline("random_forest")
        assert isinstance(pipe, Pipeline)
        assert "preprocessor" in pipe.named_steps
        assert "classifier" in pipe.named_steps

    @pytest.mark.parametrize("clf_type", ["random_forest", "logistic_regression", "svm", "knn"])
    def test_all_supported_classifiers(self, clf_type: str) -> None:
        pipe = build_fertilizer_pipeline(clf_type)
        assert pipe is not None

    def test_unsupported_classifier_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported classifier_type"):
            build_fertilizer_pipeline("xgboost")


class TestTrainFertilizerModel:
    """Tests for model training and persistence."""

    def test_train_and_save_model(self, sample_fertilizer_csv: Path, tmp_path: Path) -> None:
        out_model = tmp_path / "test_fert.joblib"
        results = train_fertilizer_model(
            csv_path=sample_fertilizer_csv,
            model_save_path=out_model,
            classifier_type="random_forest",
            cv_folds=2,
        )

        assert out_model.exists()
        assert results["train_accuracy"] >= 0.0
        assert len(results["classes"]) > 0

        # Load back and verify prediction
        loaded_pipe, classes = load_fertilizer_pipeline(out_model)
        assert isinstance(loaded_pipe, Pipeline)
        assert len(classes) == len(results["classes"])

        # Test single prediction
        row = pd.DataFrame(
            [
                {
                    "Temparature": 26,
                    "Humidity": 52,
                    "Moisture": 38,
                    "Soil Type": "Sandy",
                    "Crop Type": "Maize",
                    "Nitrogen": 37,
                    "Potassium": 0,
                    "Phosphorous": 0,
                }
            ]
        )
        pred = loaded_pipe.predict(row)
        assert pred[0] in classes

    def test_load_nonexistent_model_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_fertilizer_pipeline(tmp_path / "does_not_exist.joblib")
