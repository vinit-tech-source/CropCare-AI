"""
CropCare AI — Leaf & Disease Segmentation Module.

Uses KMeans clustering (k=3) on CIE Lab a*b* channels to segment leaf tissue
from lesions. Chooses the diseased cluster based on the lowest green ratio
(G / (R + G + B)), isolates lesion boundaries, and computes affected-area
severity percentage.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.config import get_random_seed, get_segmentation_config

logger = logging.getLogger(__name__)

# Silence loky core detection warning on Windows
if "LOKY_MAX_CPU_COUNT" not in os.environ:
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)


def extract_leaf_mask(image_rgb: np.ndarray) -> np.ndarray:
    """Isolate the leaf region from the background using Lab chroma & luminance.

    PlantVillage images contain gray/white/black neutral backgrounds.
    In Lab color space, neutral backgrounds have low chroma (deviation from 128
    on a* and b* axes).

    Parameters
    ----------
    image_rgb : np.ndarray
        RGB image of shape (H, W, 3).

    Returns
    -------
    np.ndarray
        Binary leaf mask (uint8, 0 or 255) of shape (H, W).
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2Lab)
    l_channel = lab[:, :, 0].astype(float)
    a_channel = lab[:, :, 1].astype(float) - 128.0
    b_channel = lab[:, :, 2].astype(float) - 128.0

    chroma = np.sqrt(a_channel**2 + b_channel**2)

    # Chromatic leaf pixels with adequate lighting
    raw_mask = (chroma > 12.0) & (l_channel > 25.0) & (l_channel < 245.0)
    mask_u8 = (raw_mask.astype(np.uint8)) * 255

    # Morphological filtering: close internal gaps, open small specks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    # If mask is too small (e.g. non-standard background), fallback to Otsu on grayscale
    if np.sum(cleaned > 0) < 0.05 * image_rgb.shape[0] * image_rgb.shape[1]:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cleaned = otsu

    return cleaned


def segment_leaf_and_disease(
    image_rgb: np.ndarray,
    k: int | None = None,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Segment leaf and disease lesions using KMeans on Lab a*b* channels.

    Parameters
    ----------
    image_rgb : np.ndarray
        Preprocessed RGB image of shape (H, W, 3).
    k : int | None
        Number of clusters (default 3 from config).
    random_state : int | None
        Random seed for KMeans initialization.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, float]
        - leaf_mask: uint8 binary mask of the entire leaf (0 or 255)
        - diseased_mask: uint8 binary mask of disease lesions (0 or 255)
        - overlay_image: RGB image with disease lesions highlighted in red/amber
        - severity_pct: float percentage (0.0 to 100.0) of leaf area infected
    """
    cfg = get_segmentation_config()
    k = k or cfg.get("n_clusters", 3)
    seed = random_state if random_state is not None else get_random_seed()

    leaf_mask = extract_leaf_mask(image_rgb)
    leaf_indices = np.where(leaf_mask > 0)
    leaf_pixel_count = len(leaf_indices[0])

    # If virtually no leaf detected, return empty masks
    if leaf_pixel_count < 100:
        empty_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        return leaf_mask, empty_mask, image_rgb.copy(), 0.0

    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2Lab)
    # Feature matrix: normalized a* and b* channels of leaf pixels
    features = lab[leaf_indices][:, 1:3].astype(np.float32)

    # Run KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=5)
    cluster_labels = kmeans.fit_predict(features)

    # Identify the diseased cluster:
    # Compute the green ratio for each cluster: G / (R + G + B + 1e-5)
    # Healthy leaf tissue has the highest green ratio.
    # Lesions (brown, necrotic, chlorotic, rust, black rot) have the lowest green ratio.
    rgb_leaf = image_rgb[leaf_indices].astype(np.float32)
    green_ratios = []
    mean_a_values = []

    for c in range(k):
        c_mask = cluster_labels == c
        if np.any(c_mask):
            pts_rgb = rgb_leaf[c_mask]
            r = pts_rgb[:, 0]
            g = pts_rgb[:, 1]
            b = pts_rgb[:, 2]
            gr = np.mean(g / (r + g + b + 1e-5))
            green_ratios.append(gr)
            mean_a_values.append(kmeans.cluster_centers_[c, 0])
        else:
            green_ratios.append(999.0)
            mean_a_values.append(0.0)

    # Diseased cluster has lowest green ratio (highest discoloration)
    diseased_cluster = int(np.argmin(green_ratios))

    # Construct diseased mask
    diseased_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    disease_pixel_locs = (
        leaf_indices[0][cluster_labels == diseased_cluster],
        leaf_indices[1][cluster_labels == diseased_cluster],
    )
    diseased_mask[disease_pixel_locs] = 255

    # Morphological cleanup on diseased mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    diseased_mask = cv2.morphologyEx(diseased_mask, cv2.MORPH_OPEN, kernel)

    # Compute severity percentage
    diseased_pixel_count = int(np.sum(diseased_mask > 0))
    severity_pct = float((diseased_pixel_count / max(leaf_pixel_count, 1)) * 100.0)
    severity_pct = min(max(severity_pct, 0.0), 100.0)

    # Create visual overlay: highlight diseased lesions in crimson/red with transparency
    overlay_image = image_rgb.copy()
    red_highlight = np.zeros_like(image_rgb)
    red_highlight[:, :] = [220, 20, 60]  # Crimson red

    # Blend 60% original image + 40% red overlay over diseased regions
    lesion_pixels = diseased_mask > 0
    if np.any(lesion_pixels):
        blended = cv2.addWeighted(image_rgb, 0.60, red_highlight, 0.40, 0)
        overlay_image[lesion_pixels] = blended[lesion_pixels]

        # Draw thin lesion contours in bright yellow for high visibility
        contours, _ = cv2.findContours(diseased_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_image, contours, -1, (255, 235, 59), 1)

    return leaf_mask, diseased_mask, overlay_image, round(severity_pct, 2)


def generate_elbow_silhouette_plot(
    images_rgb: list[np.ndarray],
    max_k: int = 8,
    sample_pixels: int = 6000,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Compute and plot the KMeans Elbow (Inertia) and Silhouette scores for k=2..max_k.

    Justifies the choice of k=3 for leaf/disease segmentation.

    Parameters
    ----------
    images_rgb : list[np.ndarray]
        List of preprocessed RGB leaf images.
    max_k : int
        Maximum number of clusters to evaluate (default 8).
    sample_pixels : int
        Number of leaf pixels to sub-sample for silhouette computation.
    output_path : Path | str | None
        Path to save figure. If None, saves to reports/figures/kmeans_elbow_silhouette.png.

    Returns
    -------
    dict[str, Any]
        Dictionary with k values, inertia scores, and silhouette scores.
    """
    logger.info("Computing KMeans elbow and silhouette scores for k=2..%d...", max_k)
    pixel_pool = []

    for img in images_rgb:
        mask = extract_leaf_mask(img)
        if np.any(mask > 0):
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
            pts = lab[mask > 0][:, 1:3].astype(np.float32)
            if len(pts) > 200:
                idx = np.random.choice(len(pts), size=min(len(pts), 500), replace=False)
                pixel_pool.append(pts[idx])

    if not pixel_pool:
        raise ValueError("No leaf pixels found in provided images.")

    all_pixels = np.vstack(pixel_pool)
    if len(all_pixels) > sample_pixels:
        sub_idx = np.random.choice(len(all_pixels), size=sample_pixels, replace=False)
        all_pixels = all_pixels[sub_idx]

    k_values = list(range(2, max_k + 1))
    inertias = []
    silhouettes = []

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        labels = km.fit_predict(all_pixels)
        inertias.append(float(km.inertia_))
        sil_score = float(
            silhouette_score(all_pixels, labels, sample_size=min(2000, len(all_pixels)))
        )
        silhouettes.append(sil_score)
        logger.info("  k=%d | Inertia=%.1f | Silhouette=%.3f", k, km.inertia_, sil_score)

    # Plot dual-axis figure
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    line1 = ax1.plot(
        k_values,
        inertias,
        "o-",
        color="#1976D2",
        linewidth=2.5,
        markersize=7,
        label="Inertia (SSE)",
    )
    line2 = ax2.plot(
        k_values,
        silhouettes,
        "s--",
        color="#388E3C",
        linewidth=2.5,
        markersize=7,
        label="Silhouette Score",
    )

    # Mark chosen k=3
    ax1.axvline(
        x=3, color="#D32F2F", linestyle=":", linewidth=2, label="Selected k=3 (Elbow Point)"
    )

    ax1.set_xlabel("Number of Clusters (k)", fontsize=11, fontweight="bold")
    ax1.set_ylabel(
        "Inertia (Sum of Squared Errors)", color="#1976D2", fontsize=11, fontweight="bold"
    )
    ax2.set_ylabel("Silhouette Coefficient", color="#388E3C", fontsize=11, fontweight="bold")
    ax1.set_xticks(k_values)

    lines = line1 + line2 + [ax1.get_lines()[-1]]
    labels = [line_obj.get_label() for line_obj in lines]
    ax1.legend(lines, labels, loc="upper right", frameon=True)

    plt.title(
        "KMeans Cluster Justification: Elbow Curve & Silhouette Analysis (k=3)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    plt.tight_layout()

    out_file = (
        Path(output_path) if output_path else Path("reports/figures/kmeans_elbow_silhouette.png")
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300)
    plt.close()
    logger.info("Saved elbow & silhouette justification figure to %s", out_file)

    return {
        "k_values": k_values,
        "inertias": inertias,
        "silhouettes": silhouettes,
        "selected_k": 3,
    }
