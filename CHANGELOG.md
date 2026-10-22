# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project skeleton and repository structure (Phase 0)
- Configuration system (`configs/config.yaml`, `src/config.py`)
- CLI framework with argparse (`src/cli.py`)
- Utility module with logging, seeding, and path helpers (`src/utils.py`)
- Subset creation script (`scripts/make_subset.py`)
- Disease knowledge base with 21 classes (`data/knowledge_base/diseases.json`)
- Fertilizer rules knowledge base (`data/knowledge_base/fertilizer_rules.json`)
- GitHub Actions CI pipeline (`.github/workflows/ci.yml`)
- Project documentation stubs (README, CONTRIBUTING, CHANGELOG, etc.)
- Pre-commit hooks configuration
- Data download instructions (`scripts/download_data.md`)
- Split manifest, EDA notebook, and class balance analysis (Phase 1)
- Image preprocessing and Gaussian filtering (`src/preprocessing.py`) (Phase 2)
- KMeans leaf and disease lesion segmentation (`src/segmentation.py`) (Phase 2)
- 106D handcrafted feature extraction combining HSV histograms/moments, GLCM, LBP, Hu moments, and lesion severity (`src/features.py`) (Phase 2)
- Comprehensive test suite for preprocessing, segmentation, and feature extraction reaching 86% core coverage (Phase 2)
