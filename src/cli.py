"""
CropCare AI — Command-line interface.

All user-facing commands are defined here using :mod:`argparse`.

Usage
-----
::

    python -m src.cli <command> [options]

Commands
--------
- ``predict``        — Predict disease and recommend fertilizer for a leaf image.
- ``train``          — Train disease and fertilizer models.
- ``evaluate``       — Evaluate trained models and generate reports.
- ``extract-features`` — Extract features from subset images.
- ``compare-models`` — Compare all trained models side by side.
- ``list-classes``   — List all supported crop–disease classes.
- ``demo``           — Run prediction on bundled sample images.
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.utils import setup_logging

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="cropcare-ai",
        description=(
            "🌿 CropCare AI — Crop Disease Detection and Fertilizer "
            "Recommendation System Using Machine Learning"
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ---- predict ----
    pred = sub.add_parser("predict", help="Predict disease + recommend fertilizer.")
    pred.add_argument("--image", required=True, help="Path to a leaf image.")
    pred.add_argument("--crop", required=True, help="Crop type (e.g. Tomato).")
    pred.add_argument("--soil", required=True, help="Soil type (e.g. Loamy).")
    pred.add_argument("--N", type=float, required=True, help="Soil nitrogen level.")
    pred.add_argument("--P", type=float, required=True, help="Soil phosphorus level.")
    pred.add_argument("--K", type=float, required=True, help="Soil potassium level.")
    pred.add_argument("--temp", type=float, required=True, help="Temperature (°C).")
    pred.add_argument("--humidity", type=float, required=True, help="Humidity (%%).")
    pred.add_argument("--moisture", type=float, required=True, help="Soil moisture (%%).")

    # ---- train ----
    sub.add_parser("train", help="Train disease and fertilizer models.")

    # ---- evaluate ----
    sub.add_parser("evaluate", help="Evaluate models and generate reports.")

    # ---- extract-features ----
    sub.add_parser("extract-features", help="Extract features from subset images.")

    # ---- compare-models ----
    sub.add_parser("compare-models", help="Compare all trained model metrics.")

    # ---- list-classes ----
    sub.add_parser("list-classes", help="List all supported crop–disease classes.")

    # ---- demo ----
    sub.add_parser("demo", help="Run prediction on bundled sample images.")

    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and dispatch to the appropriate handler."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    logger.info("CropCare AI — command: %s", args.command)

    # Lazy imports to keep startup fast and allow partial installs
    if args.command == "predict":
        from src.predict import run_prediction

        run_prediction(args)

    elif args.command == "train":
        from src.train_disease import train_disease_models
        from src.train_fertilizer import train_fertilizer_model

        train_disease_models()
        train_fertilizer_model()

    elif args.command == "evaluate":
        from src.evaluate import run_evaluation

        run_evaluation()

    elif args.command == "extract-features":
        from src.features import extract_all_features

        extract_all_features()

    elif args.command == "compare-models":
        from src.evaluate import compare_models

        compare_models()

    elif args.command == "list-classes":
        from src.predict import list_classes

        list_classes()

    elif args.command == "demo":
        from src.predict import run_demo

        run_demo()

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
