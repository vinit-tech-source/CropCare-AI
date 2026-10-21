"""Tests for src.config — configuration loading."""

from __future__ import annotations

from src.config import (
    CONFIG_PATH,
    PROJECT_ROOT,
    get_config,
    get_dataset_config,
    get_features_config,
    get_prediction_config,
    get_preprocessing_config,
    get_random_seed,
    get_segmentation_config,
    get_training_config,
    resolve_path,
)


class TestConfig:
    """Tests for the configuration loader."""

    def test_config_file_exists(self) -> None:
        """Config file must exist at the expected path."""
        assert CONFIG_PATH.exists(), f"Config not found: {CONFIG_PATH}"

    def test_get_config_returns_dict(self) -> None:
        """get_config() should return a dictionary."""
        cfg = get_config()
        assert isinstance(cfg, dict)

    def test_random_seed(self) -> None:
        """Random seed should be 42 as specified."""
        assert get_random_seed() == 42

    def test_dataset_config_has_crops(self) -> None:
        """Dataset config must list the target crops."""
        ds = get_dataset_config()
        assert "crops" in ds
        assert set(ds["crops"]) == {"Apple", "Corn", "Potato", "Tomato"}

    def test_preprocessing_config(self) -> None:
        """Preprocessing config should define image_size and blur_kernel."""
        pp = get_preprocessing_config()
        assert pp["image_size"] == 256
        assert pp["blur_kernel"] == 5

    def test_segmentation_config(self) -> None:
        """Segmentation config should define n_clusters."""
        seg = get_segmentation_config()
        assert seg["n_clusters"] == 3

    def test_features_config(self) -> None:
        """Features config should define hsv_bins and GLCM properties."""
        feat = get_features_config()
        assert feat["hsv_bins"] == 8
        assert "contrast" in feat["glcm_properties"]

    def test_training_config(self) -> None:
        """Training config should define cv_folds and param grids."""
        tr = get_training_config()
        assert tr["cv_folds_tuning"] == 5
        assert "svm" in tr["param_grids"]

    def test_prediction_config(self) -> None:
        """Prediction config should define confidence_threshold."""
        pred = get_prediction_config()
        assert pred["confidence_threshold"] == 0.50

    def test_resolve_path(self) -> None:
        """resolve_path should return an absolute path under project root."""
        p = resolve_path("data/raw")
        assert p.is_absolute()
        assert str(PROJECT_ROOT) in str(p)
