"""
CropCare AI — Fertilizer recommendation model training.

Trains classical ML classifiers (Random Forest, Logistic Regression, SVM, KNN)
on soil and climate parameters to predict optimal fertilizer type.
The trained pipeline combines a ColumnTransformer (StandardScaler for numerical,
OneHotEncoder for categorical) and a classifier, serialized via joblib.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import get_config, resolve_path
from src.utils import ensure_dir, save_json, timestamp_str

logger = logging.getLogger(__name__)

NUMERICAL_FEATURES: list[str] = [
    "Temparature",
    "Humidity",
    "Moisture",
    "Nitrogen",
    "Potassium",
    "Phosphorous",
]

CATEGORICAL_FEATURES: list[str] = [
    "Soil Type",
    "Crop Type",
]

TARGET_COLUMN: str = "Fertilizer Name"


def load_fertilizer_data(
    csv_path: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load and clean the fertilizer dataset from CSV.

    Args:
        csv_path: Optional path to fertilizer CSV. If None, resolves from config.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Feature matrix X and target series y.

    Raises:
        FileNotFoundError: If the specified CSV file does not exist.
        ValueError: If required columns are missing.
    """
    if csv_path is None:
        cfg = get_config()
        fert_rel = cfg.get("dataset", {}).get("fertilizer_csv", "data/raw/fertilizer.csv")
        resolved = resolve_path(fert_rel)
    else:
        resolved = Path(csv_path).resolve()

    if not resolved.exists():
        raise FileNotFoundError(f"Fertilizer dataset not found at: {resolved}")

    logger.info("Loading fertilizer data from: %s", resolved)
    df = pd.read_csv(resolved)

    # Strip any leading/trailing whitespace from column names (e.g. 'Humidity ')
    df.columns = [str(col).strip() for col in df.columns]

    missing_cols = [
        col
        for col in NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]
        if col not in df.columns
    ]
    if missing_cols:
        raise ValueError(f"Missing required columns in fertilizer dataset: {missing_cols}")

    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET_COLUMN].astype(str).str.strip()

    logger.info("Fertilizer data loaded: %d samples, %d unique classes", len(df), y.nunique())
    return X, y


def build_fertilizer_pipeline(
    classifier_type: str = "random_forest",
    random_state: int = 42,
) -> Pipeline:
    """Build a scikit-learn Pipeline for fertilizer recommendation.

    Pipeline consists of:
      1. preprocessor: ColumnTransformer
         - StandardScaler on numerical features
         - OneHotEncoder(handle_unknown='ignore') on categorical features
      2. classifier: RandomForestClassifier (or selected alternative)

    Args:
        classifier_type: Model type ('random_forest', 'logistic_regression', 'svm', 'knn').
        random_state: Random state for reproducibility.

    Returns:
        Pipeline: Assembled, un-fitted scikit-learn pipeline.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    clf_lower = classifier_type.lower()
    if clf_lower == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=random_state,
        )
    elif clf_lower == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        )
    elif clf_lower == "svm":
        from sklearn.svm import SVC

        clf = SVC(
            probability=True,
            random_state=random_state,
        )
    elif clf_lower == "knn":
        from sklearn.neighbors import KNeighborsClassifier

        clf = KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError(
            f"Unsupported classifier_type '{classifier_type}'. "
            "Choose from: 'random_forest', 'logistic_regression', 'svm', 'knn'."
        )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )


def train_fertilizer_model(
    csv_path: Path | str | None = None,
    model_save_path: Path | str | None = None,
    classifier_type: str = "random_forest",
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train, evaluate with CV, and serialize the fertilizer classification pipeline.

    Args:
        csv_path: Optional path to fertilizer CSV dataset.
        model_save_path: Optional destination path for model artifact (.joblib).
        classifier_type: Type of classifier to train.
        cv_folds: Number of Stratified K-Fold CV splits.
        random_state: Random seed for reproducibility.

    Returns:
        dict[str, Any]: Training summary including CV scores and artifact path.
    """
    X, y = load_fertilizer_data(csv_path)

    pipeline = build_fertilizer_pipeline(
        classifier_type=classifier_type,
        random_state=random_state,
    )

    # Cross-validation evaluation
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(pipeline, X, y, cv=skf, scoring="accuracy")

    logger.info(
        "Fertilizer model (%s) %d-fold CV Accuracy: %.4f (+/- %.4f)",
        classifier_type,
        cv_folds,
        float(np.mean(cv_scores)),
        float(np.std(cv_scores)),
    )

    # Fit on all available data for production inference
    pipeline.fit(X, y)
    train_preds = pipeline.predict(X)
    train_acc = float(accuracy_score(y, train_preds))
    cls_report = classification_report(y, train_preds, output_dict=True)

    # Determine save path
    if model_save_path is None:
        cfg = get_config()
        models_rel = cfg.get("paths", {}).get("models_dir", "models")
        models_dir = resolve_path(models_rel)
        ensure_dir(models_dir)
        save_target = models_dir / "fertilizer_model.joblib"
    else:
        save_target = Path(model_save_path).resolve()
        ensure_dir(save_target.parent)

    artifact_payload = {
        "pipeline": pipeline,
        "classes": list(pipeline.classes_),
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_column": TARGET_COLUMN,
        "classifier_type": classifier_type,
        "cv_accuracy_mean": float(np.mean(cv_scores)),
        "cv_accuracy_std": float(np.std(cv_scores)),
        "train_accuracy": train_acc,
        "timestamp": timestamp_str(),
    }

    joblib.dump(artifact_payload, save_target)
    logger.info("Saved fertilizer pipeline artifact to: %s", save_target)

    # Save metrics report JSON if reports directory exists
    try:
        cfg = get_config()
        reports_dir = resolve_path(cfg.get("paths", {}).get("reports_dir", "reports"))
        ensure_dir(reports_dir)
        report_data = {
            "model_type": classifier_type,
            "cv_folds": cv_folds,
            "cv_accuracy_scores": [float(s) for s in cv_scores],
            "cv_accuracy_mean": float(np.mean(cv_scores)),
            "cv_accuracy_std": float(np.std(cv_scores)),
            "train_accuracy": train_acc,
            "classes": list(pipeline.classes_),
            "classification_report": cls_report,
            "timestamp": timestamp_str(),
        }
        save_json(report_data, reports_dir / "fertilizer_training_report.json")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not save training report json: %s", exc)

    return {
        "model_path": str(save_target),
        "classifier_type": classifier_type,
        "cv_accuracy_mean": float(np.mean(cv_scores)),
        "cv_accuracy_std": float(np.std(cv_scores)),
        "train_accuracy": train_acc,
        "classes": list(pipeline.classes_),
    }


def load_fertilizer_pipeline(
    model_path: Path | str | None = None,
) -> tuple[Pipeline, list[str]]:
    """Load a trained fertilizer model pipeline and its class names.

    Args:
        model_path: Optional path to .joblib artifact. If None, resolves default.

    Returns:
        tuple[Pipeline, list[str]]: The fitted Pipeline and list of class labels.

    Raises:
        FileNotFoundError: If model file does not exist.
        KeyError: If artifact is missing required keys.
    """
    if model_path is None:
        cfg = get_config()
        models_rel = cfg.get("paths", {}).get("models_dir", "models")
        target = resolve_path(models_rel) / "fertilizer_model.joblib"
    else:
        target = Path(model_path).resolve()

    if not target.exists():
        raise FileNotFoundError(
            f"Fertilizer model not found at {target}. Run 'python -m src.cli train' first."
        )

    data = joblib.load(target)
    if isinstance(data, dict) and "pipeline" in data:
        pipeline = data["pipeline"]
        classes = data.get("classes", list(pipeline.classes_))
        return pipeline, classes
    if isinstance(data, Pipeline):
        return data, list(data.classes_)

    raise ValueError(f"Unrecognized artifact structure in {target}")
