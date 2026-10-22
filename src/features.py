"""CropCare AI — Feature Extraction Module.

Extracts a ~105-dimensional (106D) feature vector combining:
1. Color features: HSV histograms (8 bins per channel) and color moments (mean, std)
   for both the whole leaf and isolated disease lesions (60 features).
2. Texture features: Gray-Level Co-occurrence Matrix (GLCM) properties (contrast,
   homogeneity, energy, correlation, dissimilarity) across distances and angles (20 features),
   and Local Binary Patterns (LBP) uniform histogram (18 features).
3. Shape features: Hu invariant moments with logarithmic scale transform (7 features).
4. Severity feature: Lesion area percentage on the leaf (1 feature).

Total dimensionality: 106 features.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from tqdm import tqdm

from src.config import (
    get_dataset_config,
    get_features_config,
    resolve_path,
)
from src.preprocessing import load_and_preprocess_image, rgb_to_gray, rgb_to_hsv
from src.segmentation import segment_leaf_and_disease

logger = logging.getLogger(__name__)

# Fallback core count for loky/joblib on Windows
if "LOKY_MAX_CPU_COUNT" not in os.environ:
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)


def get_feature_names() -> list[str]:
    """Return an ordered list of all 106 feature names.

    Returns
    -------
    list[str]
        106 feature names in the exact order of the feature vector.
    """
    cfg = get_features_config()
    hsv_bins = int(cfg.get("hsv_bins", 8))
    distances = [int(d) for d in cfg.get("glcm_distances", [1, 3])]
    properties = list(
        cfg.get(
            "glcm_properties",
            ["contrast", "homogeneity", "energy", "correlation", "dissimilarity"],
        )
    )
    n_points = int(cfg.get("lbp_n_points", 16))

    names: list[str] = []

    # 1. Color — Whole Leaf HSV histograms (3 * hsv_bins = 24)
    for ch in ("h", "s", "v"):
        for b in range(hsv_bins):
            names.append(f"hsv_leaf_{ch}_bin_{b}")

    # Color — Whole Leaf HSV moments (mean, std = 6)
    for ch in ("h", "s", "v"):
        names.append(f"hsv_leaf_{ch}_mean")
        names.append(f"hsv_leaf_{ch}_std")

    # Color — Diseased region HSV histograms (24)
    for ch in ("h", "s", "v"):
        for b in range(hsv_bins):
            names.append(f"hsv_lesion_{ch}_bin_{b}")

    # Color — Diseased region HSV moments (6)
    for ch in ("h", "s", "v"):
        names.append(f"hsv_lesion_{ch}_mean")
        names.append(f"hsv_lesion_{ch}_std")

    # 2. Texture — GLCM mean & std across angles for each distance & property (2 * 5 * 2 = 20)
    for d in distances:
        for prop in properties:
            names.append(f"glcm_{prop}_d{d}_mean")
    for d in distances:
        for prop in properties:
            names.append(f"glcm_{prop}_d{d}_std")

    # Texture — LBP uniform histogram (n_points + 2 = 18)
    lbp_bins = n_points + 2
    for b in range(lbp_bins):
        names.append(f"lbp_bin_{b}")

    # 3. Shape — Hu Moments (7)
    for i in range(1, 8):
        names.append(f"hu_moment_{i}")

    # 4. Severity (1)
    names.append("disease_severity_pct")

    return names


def extract_color_features(
    image_rgb: np.ndarray,
    leaf_mask: np.ndarray,
    diseased_mask: np.ndarray | None = None,
    hsv_bins: int = 8,
) -> tuple[np.ndarray, list[str]]:
    """Extract HSV color histograms and color moments for leaf and lesion areas.

    Parameters
    ----------
    image_rgb : np.ndarray
        Input RGB image array of shape (H, W, 3).
    leaf_mask : np.ndarray
        Binary leaf mask (uint8, 0 or 255) of shape (H, W).
    diseased_mask : np.ndarray | None
        Binary disease lesion mask (uint8, 0 or 255) of shape (H, W).
    hsv_bins : int
        Number of bins per HSV channel (default 8).

    Returns
    -------
    tuple[np.ndarray, list[str]]
        - 60-element 1D float array of color features.
        - Corresponding feature names.
    """
    hsv = rgb_to_hsv(image_rgb)
    leaf_pts = hsv[leaf_mask > 0]
    has_leaf = len(leaf_pts) > 0

    features: list[float] = []
    names: list[str] = []

    # 1. Whole leaf HSV histograms (H: 0-180, S: 0-256, V: 0-256 in OpenCV)
    ranges = [(0, 180), (0, 256), (0, 256)]
    for i, ch in enumerate(("h", "s", "v")):
        if has_leaf:
            hist, _ = np.histogram(leaf_pts[:, i], bins=hsv_bins, range=ranges[i])
            hist_norm = hist.astype(np.float64) / (hist.sum() + 1e-7)
        else:
            hist_norm = np.zeros(hsv_bins, dtype=np.float64)

        for b, val in enumerate(hist_norm):
            features.append(float(val))
            names.append(f"hsv_leaf_{ch}_bin_{b}")

    # Whole leaf moments (mean, std)
    for i, ch in enumerate(("h", "s", "v")):
        if has_leaf:
            ch_vals = leaf_pts[:, i].astype(np.float64)
            mean_val = float(np.mean(ch_vals))
            std_val = float(np.std(ch_vals))
        else:
            mean_val, std_val = 0.0, 0.0

        features.append(mean_val)
        names.append(f"hsv_leaf_{ch}_mean")
        features.append(std_val)
        names.append(f"hsv_leaf_{ch}_std")

    # 2. Diseased lesion HSV histograms and moments
    has_lesion = diseased_mask is not None and np.any(diseased_mask > 0)
    lesion_pts = hsv[diseased_mask > 0] if has_lesion else np.empty((0, 3), dtype=np.uint8)

    for i, ch in enumerate(("h", "s", "v")):
        if has_lesion and len(lesion_pts) > 0:
            hist, _ = np.histogram(lesion_pts[:, i], bins=hsv_bins, range=ranges[i])
            hist_norm = hist.astype(np.float64) / (hist.sum() + 1e-7)
        else:
            hist_norm = np.zeros(hsv_bins, dtype=np.float64)

        for b, val in enumerate(hist_norm):
            features.append(float(val))
            names.append(f"hsv_lesion_{ch}_bin_{b}")

    for i, ch in enumerate(("h", "s", "v")):
        if has_lesion and len(lesion_pts) > 0:
            ch_vals = lesion_pts[:, i].astype(np.float64)
            mean_val = float(np.mean(ch_vals))
            std_val = float(np.std(ch_vals))
        else:
            mean_val, std_val = 0.0, 0.0

        features.append(mean_val)
        names.append(f"hsv_lesion_{ch}_mean")
        features.append(std_val)
        names.append(f"hsv_lesion_{ch}_std")

    return np.array(features, dtype=np.float64), names


def extract_glcm_features(
    image_rgb: np.ndarray,
    leaf_mask: np.ndarray,
    distances: list[int] | None = None,
    angles: list[float] | None = None,
    properties: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Extract Gray-Level Co-occurrence Matrix (GLCM) texture features.

    Parameters
    ----------
    image_rgb : np.ndarray
        Input RGB image array of shape (H, W, 3).
    leaf_mask : np.ndarray
        Binary leaf mask (uint8, 0 or 255) of shape (H, W).
    distances : list[int] | None
        Pixel displacement distances (default [1, 3]).
    angles : list[float] | None
        Pixel pair angles in radians (default [0, 45°, 90°, 135°]).
    properties : list[str] | None
        GLCM statistics to compute (default contrast, homogeneity, energy, correlation, dissimilarity).

    Returns
    -------
    tuple[np.ndarray, list[str]]
        - 20-element 1D float array of GLCM statistics.
        - Corresponding feature names.
    """
    cfg = get_features_config()
    distances = distances or [int(d) for d in cfg.get("glcm_distances", [1, 3])]
    angles = angles or [float(a) for a in cfg.get("glcm_angles", [0, 0.7854, 1.5708, 2.3562])]
    properties = properties or list(
        cfg.get(
            "glcm_properties",
            ["contrast", "homogeneity", "energy", "correlation", "dissimilarity"],
        )
    )

    gray = rgb_to_gray(image_rgb)

    # Crop to leaf bounding box to focus GLCM on leaf tissue
    y_idx, x_idx = np.where(leaf_mask > 0)
    if len(y_idx) > 10 and len(x_idx) > 10:
        ymin, ymax = int(np.min(y_idx)), int(np.max(y_idx))
        xmin, xmax = int(np.min(x_idx)), int(np.max(x_idx))
        gray_crop = gray[ymin : ymax + 1, xmin : xmax + 1]
    else:
        gray_crop = gray

    # Ensure crop is at least 4x4
    if gray_crop.shape[0] < 4 or gray_crop.shape[1] < 4:
        gray_crop = cv2.resize(gray, (32, 32))

    glcm = graycomatrix(
        gray_crop,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True,
    )

    mean_features: list[float] = []
    mean_names: list[str] = []
    std_features: list[float] = []
    std_names: list[str] = []

    for d_idx, d in enumerate(distances):
        for prop in properties:
            vals = graycoprops(glcm, prop)  # shape (len(distances), len(angles))
            row = vals[d_idx, :]
            mean_features.append(float(np.mean(row)))
            mean_names.append(f"glcm_{prop}_d{d}_mean")
            std_features.append(float(np.std(row)))
            std_names.append(f"glcm_{prop}_d{d}_std")

    all_features = mean_features + std_features
    all_names = mean_names + std_names
    return np.array(all_features, dtype=np.float64), all_names


def extract_lbp_features(
    image_rgb: np.ndarray,
    leaf_mask: np.ndarray,
    radius: int = 2,
    n_points: int = 16,
) -> tuple[np.ndarray, list[str]]:
    """Extract Local Binary Pattern (LBP) uniform texture histogram.

    Parameters
    ----------
    image_rgb : np.ndarray
        Input RGB image array of shape (H, W, 3).
    leaf_mask : np.ndarray
        Binary leaf mask (uint8, 0 or 255) of shape (H, W).
    radius : int
        Radius of circular neighborhood (default 2).
    n_points : int
        Number of circularly symmetric neighbor set points (default 16).

    Returns
    -------
    tuple[np.ndarray, list[str]]
        - 18-element 1D float array representing normalized LBP histogram.
        - Corresponding feature names.
    """
    gray = rgb_to_gray(image_rgb)
    lbp = local_binary_pattern(gray, P=n_points, R=radius, method="uniform")

    # Total uniform pattern bins = n_points + 2 (labels 0..n_points are uniform, n_points+1 is non-uniform)
    n_bins = n_points + 2
    names = [f"lbp_bin_{b}" for b in range(n_bins)]

    leaf_pixels = lbp[leaf_mask > 0]
    if len(leaf_pixels) > 0:
        hist, _ = np.histogram(leaf_pixels, bins=n_bins, range=(0, n_bins))
        hist_norm = hist.astype(np.float64) / (hist.sum() + 1e-7)
    else:
        hist_norm = np.zeros(n_bins, dtype=np.float64)

    return hist_norm, names


def extract_hu_moments(leaf_mask: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Extract 7 Hu invariant shape moments with logarithmic transformation.

    Parameters
    ----------
    leaf_mask : np.ndarray
        Binary leaf mask (uint8, 0 or 255) of shape (H, W).

    Returns
    -------
    tuple[np.ndarray, list[str]]
        - 7-element 1D float array of log-transformed Hu moments.
        - Corresponding feature names (`hu_moment_1` to `hu_moment_7`).
    """
    names = [f"hu_moment_{i}" for i in range(1, 8)]

    if not np.any(leaf_mask > 0):
        return np.zeros(7, dtype=np.float64), names

    moments = cv2.moments(leaf_mask)
    raw_hu = cv2.HuMoments(moments).flatten()

    # Log transform for scale invariance and numerical stability: -sign(h) * log10(abs(h) + eps)
    log_hu = -np.sign(raw_hu) * np.log10(np.abs(raw_hu) + 1e-12)
    # Replace any potential NaN or Inf with 0.0
    log_hu = np.nan_to_num(log_hu, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)

    return log_hu, names


def extract_image_features(
    image_rgb: np.ndarray,
    leaf_mask: np.ndarray,
    diseased_mask: np.ndarray,
    severity_pct: float,
) -> np.ndarray:
    """Combine color, GLCM, LBP, Hu moments, and severity into a single 106D feature vector.

    Parameters
    ----------
    image_rgb : np.ndarray
        Preprocessed RGB image of shape (H, W, 3).
    leaf_mask : np.ndarray
        Binary leaf mask (uint8, 0 or 255) of shape (H, W).
    diseased_mask : np.ndarray
        Binary disease lesion mask (uint8, 0 or 255) of shape (H, W).
    severity_pct : float
        Affected lesion area percentage on the leaf (0.0 to 100.0).

    Returns
    -------
    np.ndarray
        106-dimensional float64 feature vector.
    """
    cfg = get_features_config()
    hsv_bins = int(cfg.get("hsv_bins", 8))
    lbp_radius = int(cfg.get("lbp_radius", 2))
    lbp_n_points = int(cfg.get("lbp_n_points", 16))

    color_feats, _ = extract_color_features(image_rgb, leaf_mask, diseased_mask, hsv_bins=hsv_bins)
    glcm_feats, _ = extract_glcm_features(image_rgb, leaf_mask)
    lbp_feats, _ = extract_lbp_features(
        image_rgb, leaf_mask, radius=lbp_radius, n_points=lbp_n_points
    )
    hu_feats, _ = extract_hu_moments(leaf_mask)
    sev_feat = np.array([float(severity_pct)], dtype=np.float64)

    vector = np.concatenate([color_feats, glcm_feats, lbp_feats, hu_feats, sev_feat])
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)


def extract_features_for_image_path(image_path: Path | str) -> np.ndarray | None:
    """Preprocess, segment, and extract the 106D feature vector from an image file.

    Parameters
    ----------
    image_path : Path | str
        Path to the leaf image.

    Returns
    -------
    np.ndarray | None
        106-element float64 feature vector, or None if the image is unreadable.
    """
    rgb = load_and_preprocess_image(image_path)
    if rgb is None:
        return None

    leaf_mask, diseased_mask, _, severity_pct = segment_leaf_and_disease(rgb)
    return extract_image_features(rgb, leaf_mask, diseased_mask, severity_pct)


def _process_single_record(record: dict[str, Any]) -> dict[str, Any] | None:
    """Worker function to process one image manifest record.

    Parameters
    ----------
    record : dict[str, Any]
        Dictionary with record metadata including 'relative_path'.

    Returns
    -------
    dict[str, Any] | None
        Enriched dictionary containing metadata and 106 features, or None on failure.
    """
    rel_path = record.get("relative_path", "")
    full_path = resolve_path(rel_path)

    vector = extract_features_for_image_path(full_path)
    if vector is None:
        return None

    names = get_feature_names()
    result = dict(record)
    for name, val in zip(names, vector, strict=False):
        result[name] = float(val)

    return result


def extract_all_features(
    manifest_path: Path | str | None = None,
    output_csv: Path | str | None = None,
    n_jobs: int | None = None,
    max_per_class: int | None = None,
) -> pd.DataFrame:
    """Extract features for all images in the dataset manifest and save to CSV.

    Parameters
    ----------
    manifest_path : Path | str | None
        Path to split_manifest.json. Defaults to config split_manifest.
    output_csv : Path | str | None
        Path to destination CSV. Defaults to config features_file.
    n_jobs : int | None
        Number of parallel worker processes. Defaults to config n_jobs.
    max_per_class : int | None
        Optional maximum records per class and split to process.

    Returns
    -------
    pd.DataFrame
        DataFrame of extracted features with metadata.
    """
    dcfg = get_dataset_config()
    fcfg = get_features_config()

    m_path = (
        Path(manifest_path)
        if manifest_path is not None
        else resolve_path(dcfg.get("split_manifest", "data/processed/split_manifest.json"))
    )
    out_path = (
        Path(output_csv)
        if output_csv is not None
        else resolve_path(dcfg.get("features_file", "data/processed/features.csv"))
    )
    jobs = n_jobs if n_jobs is not None else int(fcfg.get("n_jobs", -1))
    if jobs == -1:
        jobs = max(1, (os.cpu_count() or 4) - 1)

    if not m_path.exists():
        raise FileNotFoundError(
            f"Manifest file not found at {m_path}. "
            "Run 'python scripts/generate_phase1_artifacts.py' first."
        )

    import json

    with open(m_path, encoding="utf-8") as fh:
        manifest_data = json.load(fh)

    records = manifest_data.get("records", [])

    if max_per_class is not None and max_per_class > 0:
        filtered = []
        counts: dict[str, int] = {}
        for r in records:
            cls = r.get("class_name", "")
            splt = r.get("split", "train")
            key = f"{cls}_{splt}"
            if counts.get(key, 0) < max_per_class:
                filtered.append(r)
                counts[key] = counts.get(key, 0) + 1
        records = filtered

    logger.info(
        "Starting feature extraction for %d images using %d workers...",
        len(records),
        jobs,
    )

    results: list[dict[str, Any]] = []

    # Run in parallel using ProcessPoolExecutor
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = {executor.submit(_process_single_record, rec): rec for rec in records}
            for fut in tqdm(
                as_completed(futures),
                total=len(records),
                desc="Extracting Features",
                unit="img",
            ):
                try:
                    res = fut.result()
                    if res is not None:
                        results.append(res)
                except Exception as exc:
                    rec = futures[fut]
                    logger.warning("Error processing image %s: %s", rec.get("relative_path"), exc)
    else:
        for rec in tqdm(records, desc="Extracting Features", unit="img"):
            res = _process_single_record(rec)
            if res is not None:
                results.append(res)

    df = pd.DataFrame(results)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Successfully extracted features for %d images -> %s", len(df), out_path)
    return df
