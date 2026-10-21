"""
CropCare AI — Phase 1 Data Artifacts & Manifest Generator.

Generates:
1. data/processed/split_manifest.json — full audit trail of train & test images
2. reports/figures/class_balance.png — leaf disease dataset class distribution plot
3. reports/figures/fertilizer_class_balance.png — fertilizer dataset distribution plots
4. reports/figures/rgb_channel_distribution.png — pixel intensity statistics across sample images
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Ensure headless plotting
plt.switch_backend("Agg")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
SUBSET_DIR = ROOT_DIR / "data" / "subset"
SUBSET_TEST_DIR = ROOT_DIR / "data" / "subset_test"
RAW_FERTILIZER = ROOT_DIR / "data" / "raw" / "fertilizer.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Generate Split Manifest
# ---------------------------------------------------------------------------
logger.info("Generating split manifest...")
records = []

for split_name, folder in [("train", SUBSET_DIR), ("test", SUBSET_TEST_DIR)]:
    if not folder.exists():
        logger.error(f"Folder not found: {folder}")
        continue
    for class_dir in sorted(folder.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        crop = class_name.split("___")[0]
        disease = class_name.split("___")[1] if "___" in class_name else class_name
        is_healthy = "healthy" in class_name.lower()

        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                records.append(
                    {
                        "relative_path": str(img_path.relative_to(ROOT_DIR)).replace("\\", "/"),
                        "filename": img_path.name,
                        "crop": crop,
                        "class_name": class_name,
                        "disease": disease,
                        "is_healthy": is_healthy,
                        "split": split_name,
                    }
                )

manifest_df = pd.DataFrame(records)
logger.info(f"Total images cataloged in manifest: {len(manifest_df)}")
logger.info(f"Split breakdown:\n{manifest_df['split'].value_counts()}")

manifest_path = PROCESSED_DIR / "split_manifest.json"
manifest_summary = {
    "total_images": len(manifest_df),
    "train_images": int((manifest_df["split"] == "train").sum()),
    "test_images": int((manifest_df["split"] == "test").sum()),
    "total_classes": int(manifest_df["class_name"].nunique()),
    "crops": sorted(manifest_df["crop"].unique().tolist()),
    "classes": sorted(manifest_df["class_name"].unique().tolist()),
    "records": records,
}

with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest_summary, f, indent=2)
logger.info(f"Saved split manifest to {manifest_path}")

# ---------------------------------------------------------------------------
# 2. Generate Class Balance Plot
# ---------------------------------------------------------------------------
logger.info("Generating leaf class balance plot...")
class_counts = (
    manifest_df.groupby(["class_name", "split"]).size().unstack(fill_value=0).reset_index()
)

plt.figure(figsize=(12, 9))
sns.set_theme(style="whitegrid", font="sans-serif")

# Color palette: distinct bars for train and test
y_pos = np.arange(len(class_counts))
bar_height = 0.4

plt.barh(
    y_pos - bar_height / 2,
    class_counts["train"],
    height=bar_height,
    label="Train Subset",
    color="#2E7D32",
    alpha=0.9,
)
plt.barh(
    y_pos + bar_height / 2,
    class_counts["test"],
    height=bar_height,
    label="Test Subset",
    color="#1565C0",
    alpha=0.9,
)

plt.yticks(y_pos, class_counts["class_name"], fontsize=9)
plt.xlabel("Number of Images", fontsize=11, fontweight="bold")
plt.title(
    "KrishiMitra: Class Balance Distribution Across 21 Classes (Train vs Test)",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
plt.legend(loc="lower right", frameon=True, fontsize=10)
plt.axvline(x=300, color="#2E7D32", linestyle="--", alpha=0.5, label="Train target (300)")
plt.axvline(x=100, color="#1565C0", linestyle="--", alpha=0.5, label="Test target (100)")
plt.tight_layout()

class_balance_fig = FIGURES_DIR / "class_balance.png"
plt.savefig(class_balance_fig, dpi=300)
plt.close()
logger.info(f"Saved class balance figure to {class_balance_fig}")

# ---------------------------------------------------------------------------
# 3. Generate Fertilizer Class Balance & Feature Distribution
# ---------------------------------------------------------------------------
logger.info("Generating fertilizer figures...")
if RAW_FERTILIZER.exists():
    fert_df = pd.read_csv(RAW_FERTILIZER)
    fert_df.columns = [c.strip() for c in fert_df.columns]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "KrishiMitra: Fertilizer Dataset Exploratory Analysis", fontsize=14, fontweight="bold"
    )

    # 1. Target Fertilizer Distribution
    fert_order = fert_df["Fertilizer Name"].value_counts().index
    sns.countplot(
        data=fert_df,
        y="Fertilizer Name",
        order=fert_order,
        ax=axes[0, 0],
        palette="viridis",
    )
    axes[0, 0].set_title("Target Fertilizer Distribution (99 Samples)", fontweight="bold")
    axes[0, 0].set_xlabel("Count")

    # 2. Crop Type Distribution
    crop_order = fert_df["Crop Type"].value_counts().index
    sns.countplot(
        data=fert_df,
        y="Crop Type",
        order=crop_order,
        ax=axes[0, 1],
        palette="mako",
    )
    axes[0, 1].set_title("Crop Type Representation", fontweight="bold")
    axes[0, 1].set_xlabel("Count")

    # 3. Soil Type Distribution
    soil_order = fert_df["Soil Type"].value_counts().index
    sns.countplot(
        data=fert_df,
        x="Soil Type",
        order=soil_order,
        ax=axes[1, 0],
        palette="magma",
    )
    axes[1, 0].set_title("Soil Type Representation", fontweight="bold")
    axes[1, 0].set_ylabel("Count")

    # 4. N-P-K Correlation Heatmap
    num_cols = ["Nitrogen", "Potassium", "Phosphorous", "Temparature", "Humidity", "Moisture"]
    corr = fert_df[num_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=axes[1, 1], cbar=True, square=True)
    axes[1, 1].set_title("Soil & Climate Feature Correlation Matrix", fontweight="bold")

    plt.tight_layout()
    fert_fig = FIGURES_DIR / "fertilizer_class_balance.png"
    plt.savefig(fert_fig, dpi=300)
    plt.close()
    logger.info(f"Saved fertilizer balance figure to {fert_fig}")

# ---------------------------------------------------------------------------
# 4. Generate RGB Color Distribution on Sample Leaves
# ---------------------------------------------------------------------------
logger.info("Computing RGB channel distribution across sample leaves...")

sample_images = list(SUBSET_DIR.rglob("*.JPG"))[:100]
if not sample_images:
    sample_images = list(SUBSET_DIR.rglob("*.jpg"))[:100]

r_vals, g_vals, b_vals = [], [], []
for p in sample_images:
    bgr = cv2.imread(str(p))
    if bgr is not None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        r_vals.append(rgb[:, :, 0].mean())
        g_vals.append(rgb[:, :, 1].mean())
        b_vals.append(rgb[:, :, 2].mean())

plt.figure(figsize=(9, 5))
sns.kdeplot(r_vals, color="red", label="Red Channel Mean", fill=True, alpha=0.3)
sns.kdeplot(g_vals, color="green", label="Green Channel Mean", fill=True, alpha=0.3)
sns.kdeplot(b_vals, color="blue", label="Blue Channel Mean", fill=True, alpha=0.3)
plt.title("Distribution of Mean RGB Channel Intensities Across Leaf Samples", fontweight="bold")
plt.xlabel("Mean Pixel Intensity (0 - 255)")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()

rgb_fig = FIGURES_DIR / "rgb_channel_distribution.png"
plt.savefig(rgb_fig, dpi=300)
plt.close()
logger.info(f"Saved RGB distribution figure to {rgb_fig}")

logger.info("Phase 1 artifacts generation complete!")
