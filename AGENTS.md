# AGENTS.md — CropCare AI

## Project Overview
CropCare AI is a classical machine learning system for crop disease detection
from leaf images and fertilizer recommendation. It uses hand-crafted features
(color, texture, shape) with SVM, KNN, Logistic Regression, and Random Forest
classifiers. No deep learning or neural networks.

## Repository Layout
- `src/` — All source code (config, preprocessing, segmentation, features,
  training, evaluation, recommendation, prediction, CLI, utilities)
- `configs/config.yaml` — Single source of truth for all tunables
- `data/knowledge_base/` — Disease and fertilizer rule JSON files (tracked)
- `data/raw/`, `data/subset/`, `data/processed/` — Data directories (git-ignored)
- `scripts/` — Data preparation scripts
- `tests/` — Pytest test suite
- `models/` — Saved model artifacts (git-ignored except model_card.json)
- `reports/` — Generated figures, CSVs, and markdown reports
- `docs/` — Architecture, methodology, dataset, and viva documentation

## Conventions
- **Python 3.11**, type hints required, Google-style docstrings
- **Formatting**: Black (line length 100), Ruff for linting
- **Commits**: Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `chore:`)
- **Random seed**: 42 everywhere (`random_state=42`)
- **Logging**: Use `logging` module, never bare `print()`
- **Config**: All tunables in `configs/config.yaml`, accessed via `src/config.py`
- **No data in git**: Raw data, processed features, model files, and outputs
  are git-ignored. Document download/generation steps instead.

## Key Design Decisions
1. Classical ML only — no CNN/transfer learning
2. Feature vector (~105D): HSV histograms + GLCM + LBP + Hu moments + severity
3. KMeans segmentation on Lab a*b* channels for disease region isolation
4. sklearn Pipeline (StandardScaler → PCA → Classifier) to prevent data leakage
5. Fertilizer ML model + rule-based disease-specific adjustments
6. CLI-only interface with rich terminal output

## Testing
- Run: `python -m pytest tests/ -v`
- Target: ≥ 70% coverage on core modules
- Tests use small synthetic data, not full dataset

## Dependencies
Pinned in `requirements.txt`. Do not add packages without discussion.
Core: numpy, pandas, scikit-learn, opencv-python, scikit-image, scipy,
matplotlib, seaborn, joblib, tqdm, rich, reportlab, PyYAML.
