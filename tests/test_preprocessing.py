"""CropCare AI — Unit tests for Image Preprocessing Module."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.preprocessing import (
    apply_gaussian_blur,
    load_and_preprocess_image,
    load_image,
    resize_image,
    rgb_to_gray,
    rgb_to_hsv,
    rgb_to_lab,
)


class TestLoadImage:
    """Tests for load_image function."""

    def test_load_valid_image(self, tmp_path: Path) -> None:
        """Test loading a valid saved image returns RGB array."""
        img_file = tmp_path / "valid.jpg"
        rgb_data = np.full((50, 50, 3), [100, 150, 200], dtype=np.uint8)
        # cv2.imwrite expects BGR
        cv2.imwrite(str(img_file), cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR))

        loaded = load_image(img_file)
        assert loaded is not None
        assert loaded.shape == (50, 50, 3)
        assert loaded.dtype == np.uint8
        # Check channel ordering preserved as RGB
        assert np.allclose(loaded[0, 0], [100, 150, 200], atol=2)

    def test_load_nonexistent_image(self, tmp_path: Path) -> None:
        """Test loading a missing image returns None without crashing."""
        missing = tmp_path / "does_not_exist.jpg"
        assert load_image(missing) is None

    def test_load_corrupt_image(self, tmp_path: Path) -> None:
        """Test loading a corrupt file returns None."""
        corrupt = tmp_path / "corrupt.jpg"
        corrupt.write_bytes(b"NOT_A_VALID_JPEG_HEADER")
        assert load_image(corrupt) is None


class TestResizeImage:
    """Tests for resize_image function."""

    def test_resize_integer_target(self, sample_image_rgb: np.ndarray) -> None:
        """Test resizing to a square target dimension."""
        resized = resize_image(sample_image_rgb, target_size=128)
        assert resized.shape == (128, 128, 3)

    def test_resize_tuple_target(self, sample_image_rgb: np.ndarray) -> None:
        """Test resizing to (width, height) tuple."""
        resized = resize_image(sample_image_rgb, target_size=(64, 32))
        assert resized.shape == (32, 64, 3)


class TestGaussianBlur:
    """Tests for apply_gaussian_blur function."""

    def test_blur_shape_preservation(self, sample_image_rgb: np.ndarray) -> None:
        """Test blur maintains array dimensions."""
        blurred = apply_gaussian_blur(sample_image_rgb, kernel_size=5)
        assert blurred.shape == sample_image_rgb.shape
        assert blurred.dtype == np.uint8

    def test_blur_even_kernel_handling(self, sample_image_rgb: np.ndarray) -> None:
        """Test that even kernel sizes are safely promoted to odd."""
        blurred = apply_gaussian_blur(sample_image_rgb, kernel_size=4)
        assert blurred.shape == sample_image_rgb.shape


class TestColorConversions:
    """Tests for color conversion utility functions."""

    def test_rgb_to_hsv(self, sample_image_rgb: np.ndarray) -> None:
        """Test RGB to HSV conversion."""
        hsv = rgb_to_hsv(sample_image_rgb)
        assert hsv.shape == sample_image_rgb.shape
        # Hue channel in OpenCV ranges 0..179
        assert np.max(hsv[:, :, 0]) <= 180

    def test_rgb_to_lab(self, sample_image_rgb: np.ndarray) -> None:
        """Test RGB to Lab conversion."""
        lab = rgb_to_lab(sample_image_rgb)
        assert lab.shape == sample_image_rgb.shape

    def test_rgb_to_gray(self, sample_image_rgb: np.ndarray) -> None:
        """Test RGB to grayscale conversion."""
        gray = rgb_to_gray(sample_image_rgb)
        assert gray.ndim == 2
        assert gray.shape == sample_image_rgb.shape[:2]


class TestLoadAndPreprocessImage:
    """Tests for load_and_preprocess_image pipeline."""

    def test_end_to_end_preprocessing(self, tmp_path: Path) -> None:
        """Test full pipeline load -> RGB -> resize -> blur."""
        img_file = tmp_path / "leaf.png"
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb[:, :] = [34, 139, 34]  # Forest green
        cv2.imwrite(str(img_file), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        result = load_and_preprocess_image(img_file, target_size=128, blur_kernel=3)
        assert result is not None
        assert result.shape == (128, 128, 3)

    def test_preprocessing_missing_file(self, tmp_path: Path) -> None:
        """Test missing file returns None."""
        assert load_and_preprocess_image(tmp_path / "nope.jpg") is None
