"""
CropCare AI — Image Preprocessing Module.

Handles robust loading, color conversion, resizing, and Gaussian filtering
for leaf images. Corrupt or unreadable image files are logged and skipped gracefully.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from src.config import get_preprocessing_config

logger = logging.getLogger(__name__)


def load_image(image_path: Path | str) -> np.ndarray | None:
    """Load an image from disk and convert from BGR to RGB.

    Parameters
    ----------
    image_path : Path | str
        Path to image file.

    Returns
    -------
    np.ndarray | None
        RGB image as uint8 array of shape (H, W, 3), or *None* if corrupt/unreadable.
    """
    path_obj = Path(image_path)
    if not path_obj.exists():
        logger.warning("Image file does not exist: %s", path_obj)
        return None

    try:
        # cv2.imread loads in BGR
        bgr = cv2.imread(str(path_obj), cv2.IMREAD_COLOR)
        if bgr is None or bgr.size == 0:
            logger.warning("Corrupt or empty image file: %s", path_obj)
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception as exc:
        logger.warning("Failed to decode image '%s': %s", path_obj, exc)
        return None


def resize_image(image: np.ndarray, target_size: int | tuple[int, int] = 256) -> np.ndarray:
    """Resize image to square target dimension using INTER_AREA for downsampling.

    Parameters
    ----------
    image : np.ndarray
        Input RGB image array.
    target_size : int | tuple[int, int]
        Desired output resolution (width, height) or single integer.

    Returns
    -------
    np.ndarray
        Resized image of shape (target_size, target_size, 3).
    """
    size = (target_size, target_size) if isinstance(target_size, int) else target_size
    h, w = image.shape[:2]
    interpolation = cv2.INTER_AREA if (h > size[1] or w > size[0]) else cv2.INTER_LINEAR
    return cv2.resize(image, size, interpolation=interpolation)


def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5, sigma: float = 0.0) -> np.ndarray:
    """Apply Gaussian smoothing to suppress sensor noise while preserving lesion edges.

    Parameters
    ----------
    image : np.ndarray
        Input RGB image array.
    kernel_size : int
        Gaussian kernel dimension (must be positive odd integer).
    sigma : float
        Gaussian standard deviation. If 0.0, computed from kernel size.

    Returns
    -------
    np.ndarray
        Blurred image.
    """
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(image, (k, k), sigmaX=sigma)


def load_and_preprocess_image(
    image_path: Path | str,
    target_size: int | tuple[int, int] | None = None,
    blur_kernel: int | None = None,
) -> np.ndarray | None:
    """End-to-end preprocessing pipeline: load -> RGB -> resize -> Gaussian blur.

    Parameters
    ----------
    image_path : Path | str
        Path to input leaf image.
    target_size : int | tuple[int, int] | None
        Target image dimension. If *None*, read from config.yaml.
    blur_kernel : int | None
        Blur kernel size. If *None*, read from config.yaml.

    Returns
    -------
    np.ndarray | None
        Preprocessed RGB image of shape (target_size, target_size, 3), or *None* if corrupt.
    """
    cfg = get_preprocessing_config()
    target_size = target_size if target_size is not None else cfg.get("image_size", 256)
    blur_kernel = blur_kernel if blur_kernel is not None else cfg.get("blur_kernel", 5)

    rgb = load_image(image_path)
    if rgb is None:
        return None

    resized = resize_image(rgb, target_size)
    smoothed = apply_gaussian_blur(resized, kernel_size=blur_kernel)
    return smoothed


def rgb_to_hsv(image: np.ndarray) -> np.ndarray:
    """Convert RGB image to HSV color space."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)


def rgb_to_lab(image: np.ndarray) -> np.ndarray:
    """Convert RGB image to CIE Lab color space."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2Lab)


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    """Convert RGB image to single-channel 8-bit grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
