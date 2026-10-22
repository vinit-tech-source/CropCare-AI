"""CropCare AI — Unit tests for Model Evaluation Module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    compare_models,
    evaluate_single_model,
)
from src.train_disease import (
    load_feature_data,
    save_model_artifacts,
    train_single_model,
)


@pytest.fixture()
def synthetic_features_csv(tmp_path: Path) -> Path:
    """Create a tiny synthetic features.csv for evaluation tests."""
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
            for f in range(10):
                row[f"feat_{f}"] = float(rng.randn())
            row["disease_severity_pct"] = float(rng.uniform(0, 50))
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = tmp_path / "features.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture()
def trained_models_dir(synthetic_features_csv: Path, tmp_path: Path) -> Path:
    """Train KNN and LR models on synthetic data and return the models dir."""
    X, y, le, feature_names = load_feature_data(synthetic_features_csv)
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    for name, grid in [
        ("knn", {"classifier__n_neighbors": [3]}),
        ("logistic_regression", {"classifier__C": [1.0], "classifier__max_iter": [500]}),
    ]:
        result = train_single_model(
            model_name=name,
            X_train=X,
            y_train=y,
            param_grid=grid,
            cv_folds=2,
            random_state=42,
            n_jobs=1,
        )
        save_model_artifacts(result, le, feature_names, models_dir=models_dir)

    return models_dir


class TestEvaluateSingleModel:
    """Tests for evaluate_single_model function."""

    def test_evaluate_returns_metrics(
        self,
        synthetic_features_csv: Path,
        trained_models_dir: Path,
        tmp_path: Path,
    ) -> None:
        """Evaluation returns a dict with expected metric keys."""
        X, y, le, _ = load_feature_data(synthetic_features_csv)
        figures_dir = tmp_path / "figures"
        reports_dir = tmp_path / "reports"

        metrics = evaluate_single_model(
            model_name="knn",
            X=X,
            y=y,
            le=le,
            models_dir=trained_models_dir,
            figures_dir=figures_dir,
            reports_dir=reports_dir,
        )

        assert "cv_accuracy_mean" in metrics
        assert "cv_f1_macro_mean" in metrics
        assert 0.0 <= metrics["cv_accuracy_mean"] <= 1.0
        assert metrics["model_name"] == "knn"

    def test_evaluate_creates_confusion_matrix(
        self,
        synthetic_features_csv: Path,
        trained_models_dir: Path,
        tmp_path: Path,
    ) -> None:
        """Confusion matrix figure should be saved."""
        X, y, le, _ = load_feature_data(synthetic_features_csv)
        figures_dir = tmp_path / "figures"
        reports_dir = tmp_path / "reports"

        evaluate_single_model(
            model_name="knn",
            X=X,
            y=y,
            le=le,
            models_dir=trained_models_dir,
            figures_dir=figures_dir,
            reports_dir=reports_dir,
        )

        assert (figures_dir / "confusion_matrix_knn.png").exists()


class TestCompareModels:
    """Tests for compare_models function."""

    def test_compare_returns_dataframe(
        self,
        synthetic_features_csv: Path,
        trained_models_dir: Path,
        tmp_path: Path,
    ) -> None:
        """compare_models returns a non-empty DataFrame with model names."""
        # Patch config paths to use tmp dirs
        df = compare_models(
            features_path=synthetic_features_csv,
            models_dir=trained_models_dir,
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "model_name" in df.columns
        assert "cv_accuracy_mean" in df.columns

    def test_compare_empty_dir(self, synthetic_features_csv: Path, tmp_path: Path) -> None:
        """No trained models returns empty DataFrame."""
        empty_dir = tmp_path / "empty_models"
        empty_dir.mkdir()
        df = compare_models(
            features_path=synthetic_features_csv,
            models_dir=empty_dir,
        )
        assert df.empty
