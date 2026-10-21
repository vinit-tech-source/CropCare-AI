"""
CropCare AI — Create a manageable subset from the PlantVillage dataset.

This script auto-detects the folder structure under ``data/raw/``, identifies
the four target crops (Apple, Corn, Potato, Tomato) with all their disease
classes plus healthy, and copies a configurable number of images per class
into ``data/subset/`` (train) and ``data/subset_test/`` (test).

Usage
-----
::

    python scripts/make_subset.py [--train-per-class 300] [--test-per-class 100]

The script is idempotent: running it again will overwrite the previous subset.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap — add project root to sys.path so we can import src.*
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import get_dataset_config, get_random_seed  # noqa: E402
from src.utils import ensure_dir, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_CROPS = {"Apple", "Corn", "Potato", "Tomato"}

# We accept several common names for the PlantVillage top-level folder.
_CANDIDATE_RAW_DIRS = [
    "New Plant Diseases Dataset(Augmented)",
    "New Plant Diseases Dataset",
    "PlantVillage",
    "plantvillage",
    "new-plant-diseases-dataset",
    "plant-diseases",
]


def _find_raw_root(raw_dir: Path) -> tuple[Path | None, Path | None]:
    """Locate the ``train`` and ``valid`` folders inside *raw_dir*.

    Returns
    -------
    tuple[Path | None, Path | None]
        ``(train_root, valid_root)`` — either may be *None* if not found.
    """
    # Direct children first
    for name in ("train", "valid", "validation", "test"):
        candidate = raw_dir / name
        if candidate.is_dir():
            if name in ("train",):
                train = candidate
            else:
                valid = candidate
            # Quick return if both found at top level
            try:
                return train, valid  # type: ignore[possibly-undefined]
            except NameError:
                continue

    # Search inside known sub-folder names
    for sub in _CANDIDATE_RAW_DIRS:
        subdir = raw_dir / sub
        if subdir.is_dir():
            result = _find_raw_root(subdir)
            if result[0] is not None:
                return result

    # Recursive single-level scan
    for child in sorted(raw_dir.iterdir()):
        if child.is_dir() and child.name not in {"__MACOSX", ".ipynb_checkpoints"}:
            result = _find_raw_root(child)
            if result[0] is not None:
                return result

    return None, None


def _find_raw_zip(raw_dir: Path, project_root: Path) -> Path | None:
    """Find a dataset zip file in *raw_dir* or *project_root*."""
    # Check data/raw/ first
    if raw_dir.is_dir():
        for z in sorted(raw_dir.glob("*.zip")):
            return z
    # Check project root
    for z in sorted(project_root.glob("*.zip")):
        return z
    return None


def _is_valid_image(path: Path) -> bool:
    """Return *True* if *path* looks like a valid image file."""
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _class_matches_crop(class_name: str) -> str | None:
    """Return the crop name if *class_name* belongs to a target crop, else *None*.

    PlantVillage class folder names look like ``Tomato___Late_blight`` or
    ``Corn_(maize)___healthy``. We match on the prefix before ``___``.
    """
    # Normalise: strip, lowercase for matching
    lower = class_name.lower().replace("(", "").replace(")", "")
    for crop in TARGET_CROPS:
        crop_lower = crop.lower()
        if lower.startswith(crop_lower) or lower.startswith(
            crop_lower.replace("corn", "corn_maize")
        ):
            return crop
        # Handle "Corn_(maize)" alias
        if crop_lower == "corn" and lower.startswith("corn"):
            return crop
    return None


def _copy_subset(
    src_dir: Path,
    dst_dir: Path,
    max_per_class: int,
    seed: int,
) -> dict[str, int]:
    """Copy up to *max_per_class* images from each target-crop class folder.

    Parameters
    ----------
    src_dir : Path
        Source directory containing class sub-folders.
    dst_dir : Path
        Destination directory.
    max_per_class : int
        Maximum number of images to copy per class.
    seed : int
        Random seed for reproducible sampling.

    Returns
    -------
    dict[str, int]
        Mapping ``{class_name: count_copied}``.
    """
    rng = np.random.RandomState(seed)
    counts: dict[str, int] = {}

    if not src_dir.is_dir():
        logger.warning("Source directory does not exist: %s", src_dir)
        return counts

    class_dirs = sorted(
        [d for d in src_dir.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )

    for class_dir in class_dirs:
        crop = _class_matches_crop(class_dir.name)
        if crop is None:
            logger.debug("Skipping class (not a target crop): %s", class_dir.name)
            continue

        images = sorted([f for f in class_dir.iterdir() if _is_valid_image(f)])
        if not images:
            logger.warning("No valid images in %s", class_dir)
            continue

        n_available = len(images)
        n_to_copy = min(max_per_class, n_available)

        if n_available < max_per_class:
            logger.warning(
                "Class '%s' has only %d images (requested %d). Using all.",
                class_dir.name,
                n_available,
                max_per_class,
            )

        # Random sample without replacement
        indices = rng.choice(n_available, size=n_to_copy, replace=False)
        selected = [images[i] for i in sorted(indices)]

        # Copy
        dest_class_dir = ensure_dir(dst_dir / class_dir.name)
        copied = 0
        for img_path in selected:
            try:
                shutil.copy2(img_path, dest_class_dir / img_path.name)
                copied += 1
            except Exception:
                logger.exception("Failed to copy %s — skipping.", img_path)

        counts[class_dir.name] = copied
        logger.info(
            "  %-45s  %4d / %4d images copied",
            class_dir.name,
            copied,
            n_available,
        )

    return counts


def _extract_subset_from_zip(
    zip_path: Path,
    subset_dir: Path,
    subset_test_dir: Path,
    train_per_class: int,
    test_per_class: int,
    seed: int,
) -> tuple[dict[str, int], dict[str, int]]:
    """Extract train and test subsets directly from a PlantVillage zip archive."""
    logger.info("Opening dataset zip file: %s", zip_path)
    rng = np.random.RandomState(seed)

    with zipfile.ZipFile(zip_path, "r") as z:
        all_names = z.namelist()

        train_pool: dict[str, list[str]] = {}
        valid_pool: dict[str, list[str]] = {}

        for name in all_names:
            if name.endswith("/"):
                continue
            lower = name.lower()
            if not any(lower.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".bmp")):
                continue
            parts = name.split("/")
            for i, part in enumerate(parts):
                p_lower = part.lower()
                if p_lower == "train" and i + 1 < len(parts):
                    cls_name = parts[i + 1]
                    if _class_matches_crop(cls_name):
                        train_pool.setdefault(cls_name, []).append(name)
                    break
                elif p_lower in ("valid", "validation") and i + 1 < len(parts):
                    cls_name = parts[i + 1]
                    if _class_matches_crop(cls_name):
                        valid_pool.setdefault(cls_name, []).append(name)
                    break

        # Deduplicate paths by file basename to avoid duplicate casing in some zips
        def dedupe(pool: dict[str, list[str]]) -> dict[str, list[str]]:
            clean_pool = {}
            for cls, items in pool.items():
                seen = set()
                unique = []
                for it in items:
                    base = Path(it).name
                    if base not in seen:
                        seen.add(base)
                        unique.append(it)
                clean_pool[cls] = sorted(unique)
            return clean_pool

        train_pool = dedupe(train_pool)
        valid_pool = dedupe(valid_pool)

        train_counts: dict[str, int] = {}
        test_counts: dict[str, int] = {}

        logger.info("\n📂 Extracting TRAIN subset directly from zip archive...")
        for cls in sorted(train_pool.keys()):
            items = train_pool[cls]
            n_avail = len(items)
            n_copy = min(train_per_class, n_avail)
            chosen_idx = rng.choice(n_avail, size=n_copy, replace=False)
            dest_cls_dir = ensure_dir(subset_dir / cls)

            for idx in chosen_idx:
                src_path = items[idx]
                dest_file = dest_cls_dir / Path(src_path).name
                with z.open(src_path) as src_f, open(dest_file, "wb") as dst_f:
                    shutil.copyfileobj(src_f, dst_f)
            train_counts[cls] = n_copy
            logger.info("  %-45s  %4d / %4d images extracted", cls, n_copy, n_avail)

        logger.info("\n📂 Extracting TEST subset directly from zip archive...")
        for cls in sorted(valid_pool.keys()):
            items = valid_pool[cls]
            n_avail = len(items)
            n_copy = min(test_per_class, n_avail)
            chosen_idx = rng.choice(n_avail, size=n_copy, replace=False)
            dest_cls_dir = ensure_dir(subset_test_dir / cls)

            for idx in chosen_idx:
                src_path = items[idx]
                dest_file = dest_cls_dir / Path(src_path).name
                with z.open(src_path) as src_f, open(dest_file, "wb") as dst_f:
                    shutil.copyfileobj(src_f, dst_f)
            test_counts[cls] = n_copy
            logger.info("  %-45s  %4d / %4d images extracted", cls, n_copy, n_avail)

        return train_counts, test_counts


def make_subset(
    train_per_class: int | None = None,
    test_per_class: int | None = None,
) -> None:
    """Main entry point: create ``data/subset/`` and ``data/subset_test/``.

    Parameters
    ----------
    train_per_class : int | None
        Override for images per class (train). Defaults to config value.
    test_per_class : int | None
        Override for images per class (test). Defaults to config value.
    """
    cfg = get_dataset_config()
    seed = get_random_seed()

    raw_dir = _PROJECT_ROOT / cfg["raw_dir"]
    subset_dir = _PROJECT_ROOT / cfg["subset_dir"]
    subset_test_dir = _PROJECT_ROOT / cfg["subset_test_dir"]

    train_per_class = train_per_class or cfg["max_images_per_class_train"]
    test_per_class = test_per_class or cfg["max_images_per_class_test"]

    logger.info("=" * 60)
    logger.info("CropCare AI — Subset Creator")
    logger.info("=" * 60)
    logger.info("Raw data root     : %s", raw_dir)
    logger.info("Train subset dest : %s", subset_dir)
    logger.info("Test subset dest  : %s", subset_test_dir)
    logger.info("Train per class   : %d", train_per_class)
    logger.info("Test per class    : %d", test_per_class)
    logger.info("Target crops      : %s", sorted(TARGET_CROPS))
    logger.info("Random seed       : %d", seed)
    logger.info("-" * 60)

    # ---- Locate train / valid folders or zip archive ----
    train_root, valid_root = _find_raw_root(raw_dir)
    zip_path = None

    if train_root is None:
        zip_path = _find_raw_zip(raw_dir, _PROJECT_ROOT)
        if zip_path is None:
            logger.error(
                "Could not find a 'train' folder under %s or any dataset .zip file. "
                "Please ensure the PlantVillage dataset is extracted or placed in data/raw/ or project root.",
                raw_dir,
            )
            sys.exit(1)
        logger.info("Found dataset archive: %s", zip_path)
    else:
        logger.info("Found train root: %s", train_root)
        if valid_root:
            logger.info("Found valid root: %s", valid_root)
        else:
            logger.warning(
                "No 'valid' or 'validation' folder found. "
                "Test subset will be sampled from the train folder using a different seed."
            )

    # ---- Clean previous subsets ----
    if subset_dir.exists():
        shutil.rmtree(subset_dir)
        logger.info("Removed previous train subset.")
    if subset_test_dir.exists():
        shutil.rmtree(subset_test_dir)
        logger.info("Removed previous test subset.")

    if zip_path is not None:
        train_counts, test_counts = _extract_subset_from_zip(
            zip_path=zip_path,
            subset_dir=subset_dir,
            subset_test_dir=subset_test_dir,
            train_per_class=train_per_class,
            test_per_class=test_per_class,
            seed=seed,
        )
    else:
        # ---- Copy train subset ----
        logger.info("\n📂 Creating TRAIN subset from: %s", train_root)
        train_counts = _copy_subset(train_root, subset_dir, train_per_class, seed)

        # ---- Copy test subset ----
        if valid_root:
            logger.info("\n📂 Creating TEST subset from: %s", valid_root)
            test_counts = _copy_subset(valid_root, subset_test_dir, test_per_class, seed)
        else:
            logger.info("\n📂 Creating TEST subset from train folder (seed+1 to avoid overlap)")
            test_counts = _copy_subset(train_root, subset_test_dir, test_per_class, seed + 1)

    # ---- Summary ----
    logger.info("\n" + "=" * 60)
    logger.info("SUBSET SUMMARY")
    logger.info("=" * 60)

    logger.info("\nTrain subset (%s):", subset_dir)
    total_train = 0
    for cls, cnt in sorted(train_counts.items()):
        logger.info("  %-45s %4d", cls, cnt)
        total_train += cnt
    logger.info("  %-45s %4d", "TOTAL", total_train)

    logger.info("\nTest subset (%s):", subset_test_dir)
    total_test = 0
    for cls, cnt in sorted(test_counts.items()):
        logger.info("  %-45s %4d", cls, cnt)
        total_test += cnt
    logger.info("  %-45s %4d", "TOTAL", total_test)

    logger.info("\nTotal classes (train): %d", len(train_counts))
    logger.info("Total classes (test) : %d", len(test_counts))

    # Warn about mismatches
    train_classes = set(train_counts.keys())
    test_classes = set(test_counts.keys())
    if train_classes != test_classes:
        missing = train_classes - test_classes
        extra = test_classes - train_classes
        if missing:
            logger.warning("Classes in train but not test: %s", missing)
        if extra:
            logger.warning("Classes in test but not train: %s", extra)

    logger.info("\n✅ Subset creation complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a subset from the PlantVillage dataset.",
    )
    parser.add_argument(
        "--train-per-class",
        type=int,
        default=None,
        help="Max images per class for train subset (default: from config).",
    )
    parser.add_argument(
        "--test-per-class",
        type=int,
        default=None,
        help="Max images per class for test subset (default: from config).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = _parse_args()
    make_subset(
        train_per_class=args.train_per_class,
        test_per_class=args.test_per_class,
    )
