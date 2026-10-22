"""CropCare AI — Disease classification model training.

Trains SVM, KNN, Logistic Regression, and Random Forest classifiers
using GridSearchCV with Stratified K-Fold cross-validation.  Each
classifier is wrapped in a ``sklearn.pipeline.Pipeline``::

    StandardScaler → PCA(0.95 variance) → Classifier

This prevents data leakage by ensuring scaling and dimensionality
reduction are fitted only on training folds.

Implemented in Phase 3.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from src.config import (
    get_dataset_config,
    get_paths_config,
    get_training_config,
    resolve_path,
)
from src.utils import ensure_dir, save_json, set_global_seed, timestamp_str

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: dict[str, type] = {
    "svm": SVC,
    "knn": KNeighborsClassifier,
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
}


def _build_pipeline(
    classifier_cls: type,
    pca_variance: float = 0.95,
    random_state: int = 42,
) -> Pipeline:
    """Construct a StandardScaler → PCA → Classifier pipeline.

    Parameters
    ----------
    classifier_cls : type
        A scikit-learn classifier class (e.g. ``SVC``).
    pca_variance : float
        Cumulative explained variance ratio retained by PCA (default 0.95).
    random_state : int
        Random seed passed to the classifier (if it accepts one).

    Returns
    -------
    Pipeline
        Unfitted sklearn pipeline.
    """
    # Determine constructor kwargs based on the classifier
    clf_params: dict[str, Any] = {}
    if classifier_cls is SVC:
        clf_params = {"random_state": random_state, "probability": True}
    elif classifier_cls is LogisticRegression or classifier_cls is RandomForestClassifier:
        clf_params = {"random_state": random_state}
    # KNN has no random_state parameter

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=pca_variance, random_state=random_state)),
            ("classifier", classifier_cls(**clf_params)),
        ]
    )


def load_feature_data(
    features_path: Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray, LabelEncoder, list[str]]:
    """Load the feature CSV and return X, y arrays with a fitted LabelEncoder.

    Parameters
    ----------
    features_path : Path | str | None
        Path to ``features.csv``. If *None*, read from config.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, LabelEncoder, list[str]]
        - X: float64 feature matrix of shape (n_samples, n_features).
        - y: int64 encoded label array of shape (n_samples,).
        - le: Fitted LabelEncoder mapping class names → integers.
        - feature_names: List of feature column names.
    """
    dcfg = get_dataset_config()
    feat_path = (
        Path(features_path)
        if features_path is not None
        else resolve_path(dcfg.get("features_file", "data/processed/features.csv"))
    )

    if not feat_path.exists():
        raise FileNotFoundError(
            f"Feature file not found at {feat_path}. "
            "Run 'python -m src.cli extract-features' first."
        )

    df = pd.read_csv(feat_path)
    logger.info(
        "Loaded feature CSV: %d samples, %d columns from %s", len(df), len(df.columns), feat_path
    )

    # Identify feature columns (everything that is not metadata)
    meta_cols = {
        "relative_path",
        "filename",
        "crop",
        "class_name",
        "disease",
        "is_healthy",
        "split",
    }
    feature_cols = [c for c in df.columns if c not in meta_cols]
    logger.info("Feature columns: %d", len(feature_cols))

    X = df[feature_cols].values.astype(np.float64)
    # Replace NaN / Inf with 0.0 for robustness
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    le = LabelEncoder()
    y = le.fit_transform(df["class_name"].values)
    logger.info("Classes: %d → %s", len(le.classes_), list(le.classes_))

    return X, y, le, feature_cols


def train_single_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: dict[str, list[Any]],
    cv_folds: int = 5,
    pca_variance: float = 0.95,
    random_state: int = 42,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Train one classifier using GridSearchCV and return results.

    Parameters
    ----------
    model_name : str
        Key from ``_MODEL_REGISTRY`` (e.g. ``"svm"``).
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training encoded labels.
    param_grid : dict[str, list[Any]]
        Hyperparameter grid for ``GridSearchCV``.
    cv_folds : int
        Number of stratified folds for tuning (default 5).
    pca_variance : float
        PCA variance to retain (default 0.95).
    random_state : int
        Global random seed.
    n_jobs : int
        Parallel jobs for GridSearchCV (-1 = all cores).

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - ``"model_name"``: str
        - ``"best_pipeline"``: fitted Pipeline
        - ``"best_params"``: dict of best hyperparameters
        - ``"best_cv_score"``: float (mean CV accuracy)
        - ``"cv_results"``: dict from GridSearchCV
        - ``"training_time_sec"``: float
    """
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(_MODEL_REGISTRY)}")

    classifier_cls = _MODEL_REGISTRY[model_name]
    pipeline = _build_pipeline(classifier_cls, pca_variance=pca_variance, random_state=random_state)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    logger.info(
        "Training '%s' with GridSearchCV (%d-fold, %d param combos)...",
        model_name,
        cv_folds,
        _count_param_combos(param_grid),
    )

    t0 = time.perf_counter()
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=n_jobs,
        verbose=0,
        refit=True,
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    logger.info(
        "  ✓ %s — Best CV accuracy: %.4f | Time: %.1fs | Best params: %s",
        model_name.upper(),
        grid_search.best_score_,
        elapsed,
        grid_search.best_params_,
    )

    return {
        "model_name": model_name,
        "best_pipeline": grid_search.best_estimator_,
        "best_params": grid_search.best_params_,
        "best_cv_score": float(grid_search.best_score_),
        "cv_results": {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in grid_search.cv_results_.items()
        },
        "training_time_sec": round(elapsed, 2),
    }


def _count_param_combos(param_grid: dict[str, list[Any]]) -> int:
    """Count total combinations in a parameter grid."""
    count = 1
    for values in param_grid.values():
        count *= len(values)
    return count


def save_model_artifacts(
    result: dict[str, Any],
    label_encoder: LabelEncoder,
    feature_names: list[str],
    models_dir: Path | None = None,
) -> Path:
    """Save the trained pipeline, label encoder, and metadata to disk.

    Parameters
    ----------
    result : dict[str, Any]
        Output from ``train_single_model``.
    label_encoder : LabelEncoder
        Fitted LabelEncoder for class names.
    feature_names : list[str]
        Feature column names used during training.
    models_dir : Path | None
        Directory to save into. Defaults to ``models/`` from config.

    Returns
    -------
    Path
        Path to the saved model file.
    """
    pcfg = get_paths_config()
    models_dir = models_dir or resolve_path(pcfg.get("models_dir", "models"))
    ensure_dir(models_dir)

    model_name = result["model_name"]

    # Save the pipeline
    pipeline_path = models_dir / f"disease_{model_name}_pipeline.joblib"
    joblib.dump(result["best_pipeline"], pipeline_path)
    logger.info("Saved pipeline → %s", pipeline_path)

    # Save the label encoder (shared across all models)
    le_path = models_dir / "disease_label_encoder.joblib"
    joblib.dump(label_encoder, le_path)

    # Save training metadata
    meta = {
        "model_name": model_name,
        "best_params": result["best_params"],
        "best_cv_score": result["best_cv_score"],
        "training_time_sec": result["training_time_sec"],
        "n_classes": len(label_encoder.classes_),
        "classes": list(label_encoder.classes_),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "timestamp": timestamp_str(),
    }
    meta_path = models_dir / f"disease_{model_name}_meta.json"
    save_json(meta, meta_path)

    return pipeline_path


def train_disease_models(
    features_path: Path | str | None = None,
    models_dir: Path | str | None = None,
    model_names: list[str] | None = None,
    cv_folds: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Train all (or selected) disease classification models end-to-end.

    This is the main entry point called by the CLI ``train`` command.

    Parameters
    ----------
    features_path : Path | str | None
        Path to ``features.csv``. Defaults to config value.
    models_dir : Path | str | None
        Directory to save trained models. Defaults to config value.
    model_names : list[str] | None
        List of model names to train. Defaults to all four models.
    cv_folds : int | None
        Optional cross-validation folds. Defaults to config cv_folds_tuning.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping from model name to its training result dictionary.
    """
    seed = set_global_seed()
    tcfg = get_training_config()

    if cv_folds is None:
        cv_folds = int(tcfg.get("cv_folds_tuning", 5))
    pca_variance = float(tcfg.get("pca_variance", 0.95))
    param_grids = tcfg.get("param_grids", {})

    # Load features
    X, y, le, feature_names = load_feature_data(features_path)
    logger.info(
        "Training dataset: %d samples, %d features, %d classes",
        X.shape[0],
        X.shape[1],
        len(le.classes_),
    )

    min_class_samples = int(np.min(np.bincount(y)))
    if cv_folds > min_class_samples:
        logger.warning(
            "Clamping cv_folds from %d to %d based on smallest class count (%d).",
            cv_folds,
            min_class_samples,
            min_class_samples,
        )
        cv_folds = max(2, min_class_samples)

    # Resolve models dir
    if models_dir is not None:
        m_dir = Path(models_dir)
    else:
        pcfg = get_paths_config()
        m_dir = resolve_path(pcfg.get("models_dir", "models"))

    # Determine which models to train
    names = model_names or list(_MODEL_REGISTRY.keys())
    results: dict[str, dict[str, Any]] = {}

    logger.info("=" * 60)
    logger.info("CropCare AI — Disease Model Training")
    logger.info("Models to train: %s", names)
    logger.info("CV folds (tuning): %d | PCA variance: %.2f", cv_folds, pca_variance)
    logger.info("=" * 60)

    for name in names:
        grid = param_grids.get(name, {})
        if not grid:
            logger.warning("No param grid found for '%s' in config. Using defaults.", name)

        result = train_single_model(
            model_name=name,
            X_train=X,
            y_train=y,
            param_grid=grid,
            cv_folds=cv_folds,
            pca_variance=pca_variance,
            random_state=seed,
            n_jobs=-1,
        )
        results[name] = result

        # Save model artifacts
        save_model_artifacts(result, le, feature_names, models_dir=m_dir)

    # Save a summary comparison JSON
    summary = {
        "training_timestamp": timestamp_str(),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": len(le.classes_),
        "classes": list(le.classes_),
        "models": {
            name: {
                "best_cv_accuracy": res["best_cv_score"],
                "best_params": res["best_params"],
                "training_time_sec": res["training_time_sec"],
            }
            for name, res in results.items()
        },
    }
    summary_path = m_dir / "training_summary.json"
    save_json(summary, summary_path)

    # Log final comparison table
    logger.info("=" * 60)
    logger.info("Training Summary:")
    logger.info("%-25s  %12s  %10s", "Model", "CV Accuracy", "Time (s)")
    logger.info("-" * 52)
    for name, res in results.items():
        logger.info(
            "%-25s  %12.4f  %10.1f",
            name.upper(),
            res["best_cv_score"],
            res["training_time_sec"],
        )
    logger.info("=" * 60)
    logger.info("All models and metadata saved to %s", m_dir)

    return results


def load_trained_pipeline(
    model_name: str = "svm",
    models_dir: Path | str | None = None,
) -> tuple[Pipeline, LabelEncoder, dict[str, Any]]:
    """Load a previously trained disease model pipeline from disk.

    Parameters
    ----------
    model_name : str
        Name of the model (e.g. ``"svm"``, ``"random_forest"``).
    models_dir : Path | str | None
        Directory containing saved models. Defaults to config value.

    Returns
    -------
    tuple[Pipeline, LabelEncoder, dict[str, Any]]
        - Fitted Pipeline ready for ``.predict()``.
        - Fitted LabelEncoder for decoding predictions.
        - Training metadata dictionary.
    """
    if models_dir is not None:
        m_dir = Path(models_dir)
    else:
        pcfg = get_paths_config()
        m_dir = resolve_path(pcfg.get("models_dir", "models"))

    pipeline_path = m_dir / f"disease_{model_name}_pipeline.joblib"
    le_path = m_dir / "disease_label_encoder.joblib"
    meta_path = m_dir / f"disease_{model_name}_meta.json"

    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {pipeline_path}. " "Run 'python -m src.cli train' first."
        )

    pipeline = joblib.load(pipeline_path)
    le = joblib.load(le_path)
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)

    logger.info("Loaded trained '%s' pipeline from %s", model_name, pipeline_path)
    return pipeline, le, meta
