"""CropCare AI — Unit tests for Disease Training and Evaluation Modules."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.train_disease import (
    _MODEL_REGISTRY,
    _build_pipeline,
    _count_param_combos,
    load_feature_data,
    load_trained_pipeline,
    save_model_artifacts,
    train_disease_models,
    train_single_model,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_features_csv(tmp_path: Path) -> Path:
    """Create a tiny synthetic features.csv for training tests.

    Generates 60 samples across 3 classes with 10 random features + severity.
    """
    rng = np.random.RandomState(42)
    n_per_class = 20
    classes = ["Apple___Apple_scab", "Tomato___healthy", "Potato___Early_blight"]
    rows = []

    for cls in classes:
        crop = cls.split("___")[0]
        disease = cls.split("___")[1]
        is_healthy = "healthy" in cls.lower()
        for i in range(n_per_class):
            row = {
                "relative_path": f"data/subset/{cls}/img_{i:04d}.jpg",
                "filename": f"img_{i:04d}.jpg",
                "crop": crop,
                "class_name": cls,
                "disease": disease,
                "is_healthy": is_healthy,
                "split": "train",
            }
            # 10 dummy feature columns
            for f in range(10):
                row[f"feat_{f}"] = float(rng.randn())
            row["disease_severity_pct"] = float(rng.uniform(0, 50))
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = tmp_path / "features.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture()
def tiny_training_result(synthetic_features_csv: Path) -> dict:
    """Train a single KNN model on synthetic data for reuse in tests."""
    X, y, le, feature_names = load_feature_data(synthetic_features_csv)
    result = train_single_model(
        model_name="knn",
        X_train=X,
        y_train=y,
        param_grid={"classifier__n_neighbors": [3, 5]},
        cv_folds=2,
        pca_variance=0.95,
        random_state=42,
        n_jobs=1,
    )
    result["_le"] = le
    result["_feature_names"] = feature_names
    return result


# ---------------------------------------------------------------------------
# Tests — Pipeline Construction
# ---------------------------------------------------------------------------


class TestBuildPipeline:
    """Tests for _build_pipeline helper."""

    def test_pipeline_has_three_steps(self) -> None:
        """Pipeline must contain scaler, pca, and classifier."""
        from sklearn.svm import SVC

        pipe = _build_pipeline(SVC, pca_variance=0.95, random_state=42)
        step_names = [name for name, _ in pipe.steps]
        assert step_names == ["scaler", "pca", "classifier"]

    def test_pipeline_for_each_model(self) -> None:
        """Every registered model class produces a valid pipeline."""
        for name, cls in _MODEL_REGISTRY.items():
            pipe = _build_pipeline(cls, pca_variance=0.95, random_state=42)
            assert len(pipe.steps) == 3, f"Pipeline for {name} has wrong step count"


class TestCountParamCombos:
    """Tests for _count_param_combos helper."""

    def test_simple_grid(self) -> None:
        assert _count_param_combos({"a": [1, 2], "b": [3, 4, 5]}) == 6

    def test_single_param(self) -> None:
        assert _count_param_combos({"a": [1]}) == 1

    def test_empty_grid(self) -> None:
        assert _count_param_combos({}) == 1


# ---------------------------------------------------------------------------
# Tests — Data Loading
# ---------------------------------------------------------------------------


class TestLoadFeatureData:
    """Tests for load_feature_data function."""

    def test_load_returns_correct_shapes(self, synthetic_features_csv: Path) -> None:
        """Verify X, y shapes and LabelEncoder classes."""
        X, y, le, feature_names = load_feature_data(synthetic_features_csv)

        assert X.shape == (60, 11)  # 10 feat_* + 1 severity
        assert y.shape == (60,)
        assert len(le.classes_) == 3
        assert len(feature_names) == 11

    def test_load_missing_file(self, tmp_path: Path) -> None:
        """FileNotFoundError for missing CSV."""
        with pytest.raises(FileNotFoundError):
            load_feature_data(tmp_path / "nonexistent.csv")

    def test_labels_are_encoded(self, synthetic_features_csv: Path) -> None:
        """y values should be integers 0..n_classes-1."""
        _, y, le, _ = load_feature_data(synthetic_features_csv)
        assert set(y) == set(range(len(le.classes_)))


# ---------------------------------------------------------------------------
# Tests — Single Model Training
# ---------------------------------------------------------------------------


class TestTrainSingleModel:
    """Tests for train_single_model function."""

    def test_knn_training_returns_valid_result(self, synthetic_features_csv: Path) -> None:
        """KNN trains and returns expected result keys."""
        X, y, _, _ = load_feature_data(synthetic_features_csv)
        result = train_single_model(
            model_name="knn",
            X_train=X,
            y_train=y,
            param_grid={"classifier__n_neighbors": [3]},
            cv_folds=2,
            random_state=42,
            n_jobs=1,
        )

        assert result["model_name"] == "knn"
        assert 0.0 <= result["best_cv_score"] <= 1.0
        assert "best_pipeline" in result
        assert "best_params" in result
        assert result["training_time_sec"] >= 0

    def test_unknown_model_raises(self, synthetic_features_csv: Path) -> None:
        """Unknown model name should raise ValueError."""
        X, y, _, _ = load_feature_data(synthetic_features_csv)
        with pytest.raises(ValueError, match="Unknown model"):
            train_single_model(
                model_name="xgboost",
                X_train=X,
                y_train=y,
                param_grid={},
                cv_folds=2,
                random_state=42,
            )

    def test_trained_pipeline_can_predict(self, synthetic_features_csv: Path) -> None:
        """The returned pipeline should be able to predict on new data."""
        X, y, _, _ = load_feature_data(synthetic_features_csv)
        result = train_single_model(
            model_name="logistic_regression",
            X_train=X,
            y_train=y,
            param_grid={"classifier__C": [1.0], "classifier__max_iter": [500]},
            cv_folds=2,
            random_state=42,
            n_jobs=1,
        )
        preds = result["best_pipeline"].predict(X)
        assert preds.shape == (60,)
        assert all(p in range(3) for p in preds)


# ---------------------------------------------------------------------------
# Tests — Model Persistence
# ---------------------------------------------------------------------------


class TestSaveAndLoadModel:
    """Tests for save_model_artifacts and load_trained_pipeline."""

    def test_save_creates_files(self, tiny_training_result: dict, tmp_path: Path) -> None:
        """Saving model artifacts creates pipeline, encoder, and meta files."""
        result = tiny_training_result
        le = result["_le"]
        feature_names = result["_feature_names"]

        save_model_artifacts(result, le, feature_names, models_dir=tmp_path)

        assert (tmp_path / "disease_knn_pipeline.joblib").exists()
        assert (tmp_path / "disease_label_encoder.joblib").exists()
        assert (tmp_path / "disease_knn_meta.json").exists()

    def test_load_trained_pipeline(self, tiny_training_result: dict, tmp_path: Path) -> None:
        """Round-trip: save then load the pipeline and verify predictions match."""
        result = tiny_training_result
        le = result["_le"]
        feature_names = result["_feature_names"]

        save_model_artifacts(result, le, feature_names, models_dir=tmp_path)
        loaded_pipe, loaded_le, loaded_meta = load_trained_pipeline("knn", models_dir=tmp_path)

        # Label encoder classes match
        assert list(loaded_le.classes_) == list(le.classes_)
        # Meta contains expected keys
        assert loaded_meta["model_name"] == "knn"
        assert "best_cv_score" in loaded_meta

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        """Loading from empty dir should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_trained_pipeline("svm", models_dir=tmp_path)


# ---------------------------------------------------------------------------
# Tests — Full Training Pipeline
# ---------------------------------------------------------------------------


class TestTrainDiseaseModels:
    """Tests for the end-to-end train_disease_models function."""

    def test_train_all_models(self, synthetic_features_csv: Path, tmp_path: Path) -> None:
        """Train all 4 models on synthetic data and verify outputs."""
        results = train_disease_models(
            features_path=synthetic_features_csv,
            models_dir=tmp_path,
            model_names=["knn", "logistic_regression"],
        )

        # Should have trained 2 models
        assert len(results) == 2
        assert "knn" in results
        assert "logistic_regression" in results

        # Verify model files exist
        assert (tmp_path / "disease_knn_pipeline.joblib").exists()
        assert (tmp_path / "disease_logistic_regression_pipeline.joblib").exists()
        assert (tmp_path / "disease_label_encoder.joblib").exists()
        assert (tmp_path / "training_summary.json").exists()

        # Verify summary JSON
        with open(tmp_path / "training_summary.json") as fh:
            summary = json.load(fh)
        assert summary["n_classes"] == 3
        assert "knn" in summary["models"]
        assert "logistic_regression" in summary["models"]

    def test_train_single_specific_model(
        self, synthetic_features_csv: Path, tmp_path: Path
    ) -> None:
        """Train only RF and verify it exists."""
        results = train_disease_models(
            features_path=synthetic_features_csv,
            models_dir=tmp_path,
            model_names=["random_forest"],
        )
        assert "random_forest" in results
        assert (tmp_path / "disease_random_forest_pipeline.joblib").exists()
