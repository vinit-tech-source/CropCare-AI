"""CropCare AI — Model evaluation and report generation.

Generates confusion matrices, per-class classification reports,
ROC curves (one-vs-rest), model comparison tables, and stratified
K-fold cross-validation statistics.

Implemented in Phase 3.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from src.config import get_paths_config, get_random_seed, get_training_config, resolve_path
from src.train_disease import _MODEL_REGISTRY, load_feature_data, load_trained_pipeline
from src.utils import ensure_dir, save_json, set_global_seed

logger = logging.getLogger(__name__)

# Ensure headless plotting
plt.switch_backend("Agg")


def evaluate_single_model(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    le: LabelEncoder,
    cv_folds: int | None = None,
    models_dir: Path | None = None,
    figures_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one trained model and generate all figures and reports.

    Parameters
    ----------
    model_name : str
        Name key (e.g. ``"svm"``).
    X : np.ndarray
        Feature matrix (all data used for K-fold evaluation).
    y : np.ndarray
        Encoded label array.
    le : LabelEncoder
        Fitted label encoder.
    cv_folds : int | None
        Number of CV folds. Clamped to min class size if smaller.
    models_dir : Path | None
        Directory with saved models.
    figures_dir : Path | None
        Directory for output figures.
    reports_dir : Path | None
        Directory for output reports.

    Returns
    -------
    dict[str, Any]
        Evaluation metrics dictionary.
    """
    pcfg = get_paths_config()
    figures_dir = figures_dir or resolve_path(pcfg.get("figures_dir", "reports/figures"))
    reports_dir = reports_dir or resolve_path(pcfg.get("reports_dir", "reports"))
    ensure_dir(figures_dir)
    ensure_dir(reports_dir)

    pipeline, _, meta = load_trained_pipeline(model_name, models_dir=models_dir)

    # --- Stratified CV for reporting ---
    tcfg = get_training_config()
    if cv_folds is None:
        cv_folds = int(tcfg.get("cv_folds_reporting", 10))
    seed = get_random_seed()

    min_class_samples = int(np.min(np.bincount(y))) if len(y) > 0 else 0
    if cv_folds > min_class_samples:
        logger.warning(
            "Clamping cv_folds from %d to %d based on smallest class count (%d).",
            cv_folds,
            min_class_samples,
            min_class_samples,
        )
        cv_folds = max(2, min_class_samples) if min_class_samples >= 2 else 0

    if cv_folds >= 2:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        logger.info("Running %d-fold stratified CV for '%s'...", cv_folds, model_name)
        cv_results = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=["accuracy", "f1_macro", "precision_macro", "recall_macro"],
            n_jobs=1,
            return_train_score=True,
        )

        metrics = {
            "model_name": model_name,
            "cv_folds": cv_folds,
            "cv_accuracy_mean": float(np.mean(cv_results["test_accuracy"])),
            "cv_accuracy_std": float(np.std(cv_results["test_accuracy"])),
            "cv_f1_macro_mean": float(np.mean(cv_results["test_f1_macro"])),
            "cv_f1_macro_std": float(np.std(cv_results["test_f1_macro"])),
            "cv_precision_macro_mean": float(np.mean(cv_results["test_precision_macro"])),
            "cv_precision_macro_std": float(np.std(cv_results["test_precision_macro"])),
            "cv_recall_macro_mean": float(np.mean(cv_results["test_recall_macro"])),
            "cv_recall_macro_std": float(np.std(cv_results["test_recall_macro"])),
            "train_accuracy_mean": float(np.mean(cv_results["train_accuracy"])),
            "train_accuracy_std": float(np.std(cv_results["train_accuracy"])),
        }
    else:
        logger.warning(
            "Class count too small (%d) for stratified CV. Using full-data metrics.",
            min_class_samples,
        )
        metrics = {
            "model_name": model_name,
            "cv_folds": 0,
            "cv_accuracy_mean": 0.0,
            "cv_accuracy_std": 0.0,
            "cv_f1_macro_mean": 0.0,
            "cv_f1_macro_std": 0.0,
            "cv_precision_macro_mean": 0.0,
            "cv_precision_macro_std": 0.0,
            "cv_recall_macro_mean": 0.0,
            "cv_recall_macro_std": 0.0,
            "train_accuracy_mean": 0.0,
            "train_accuracy_std": 0.0,
        }

    logger.info(
        "  %s — CV Accuracy: %.4f ± %.4f | Macro F1: %.4f ± %.4f",
        model_name.upper(),
        metrics["cv_accuracy_mean"],
        metrics["cv_accuracy_std"],
        metrics["cv_f1_macro_mean"],
        metrics["cv_f1_macro_std"],
    )

    # --- Full-data predictions for confusion matrix and classification report ---
    y_pred = pipeline.predict(X)
    class_names = list(le.classes_)

    # Classification report
    report_str = classification_report(y, y_pred, target_names=class_names)
    report_dict = classification_report(y, y_pred, target_names=class_names, output_dict=True)
    logger.info("\n%s Classification Report:\n%s", model_name.upper(), report_str)

    report_path = reports_dir / f"classification_report_{model_name}.json"
    save_json(report_dict, report_path)

    # --- Confusion Matrix ---
    cm = confusion_matrix(y, y_pred)
    _plot_confusion_matrix(cm, class_names, model_name, figures_dir)

    # --- ROC Curves (if model supports probability) ---
    if hasattr(pipeline, "predict_proba"):
        try:
            y_prob = pipeline.predict_proba(X)
            _plot_roc_curves(y, y_prob, class_names, model_name, figures_dir)
            # Compute macro AUC
            lb = LabelBinarizer()
            y_bin = lb.fit_transform(y)
            if y_bin.shape[1] > 1:
                metrics["roc_auc_macro"] = float(
                    roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")
                )
        except Exception as exc:
            logger.warning("Could not compute ROC for %s: %s", model_name, exc)

    metrics["full_data_accuracy"] = float(accuracy_score(y, y_pred))
    metrics["full_data_f1_macro"] = float(f1_score(y, y_pred, average="macro"))
    metrics["full_data_precision_macro"] = float(precision_score(y, y_pred, average="macro"))
    metrics["full_data_recall_macro"] = float(recall_score(y, y_pred, average="macro"))

    # Save per-model evaluation
    eval_path = reports_dir / f"evaluation_{model_name}.json"
    save_json(metrics, eval_path)

    return metrics


def _plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    model_name: str,
    figures_dir: Path,
) -> None:
    """Plot and save a heatmap confusion matrix."""
    fig, ax = plt.subplots(figsize=(14, 12))

    # Shorten class names for readability
    short_names = [n.replace("___", "\n").replace("_", " ") for n in class_names]

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=short_names,
        yticklabels=short_names,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("True", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Confusion Matrix — {model_name.upper()}",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()

    out_path = figures_dir / f"confusion_matrix_{model_name}.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info("Saved confusion matrix → %s", out_path)


def _plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    model_name: str,
    figures_dir: Path,
) -> None:
    """Plot One-vs-Rest ROC curves for all classes."""
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y_true)
    n_classes = y_bin.shape[1]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, n_classes))

    for i in range(min(n_classes, y_prob.shape[1])):
        if len(np.unique(y_bin[:, i])) < 2:
            continue
        try:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            auc = roc_auc_score(y_bin[:, i], y_prob[:, i])
            short_name = class_names[i].split("___")[-1].replace("_", " ")
            ax.plot(
                fpr,
                tpr,
                color=colors[i % len(colors)],
                linewidth=1.2,
                label=f"{short_name} (AUC={auc:.3f})",
            )
        except Exception:
            continue

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=11, fontweight="bold")
    ax.set_title(
        f"One-vs-Rest ROC Curves — {model_name.upper()}",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.legend(fontsize=6, loc="lower right", ncol=2)
    plt.tight_layout()

    out_path = figures_dir / f"roc_curves_{model_name}.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info("Saved ROC curves → %s", out_path)


def compare_models(
    features_path: Path | str | None = None,
    models_dir: Path | str | None = None,
    cv_folds: int | None = None,
) -> pd.DataFrame:
    """Compare all trained models side-by-side and generate a summary CSV.

    Parameters
    ----------
    features_path : Path | str | None
        Path to feature CSV. Defaults to config.
    models_dir : Path | str | None
        Path to models directory. Defaults to config.
    cv_folds : int | None
        Number of CV folds to run.

    Returns
    -------
    pd.DataFrame
        Comparison table with CV accuracy, F1, precision, recall per model.
    """
    pcfg = get_paths_config()
    reports_dir = resolve_path(pcfg.get("reports_dir", "reports"))
    figures_dir = resolve_path(pcfg.get("figures_dir", "reports/figures"))
    ensure_dir(reports_dir)
    ensure_dir(figures_dir)

    m_dir = Path(models_dir) if models_dir else resolve_path(pcfg.get("models_dir", "models"))

    X, y, le, _ = load_feature_data(features_path)

    # Discover trained models
    trained = []
    for name in _MODEL_REGISTRY:
        pipeline_file = m_dir / f"disease_{name}_pipeline.joblib"
        if pipeline_file.exists():
            trained.append(name)

    if not trained:
        logger.warning("No trained models found in %s", m_dir)
        return pd.DataFrame()

    logger.info("Comparing models: %s", trained)
    all_metrics: list[dict[str, Any]] = []

    for name in trained:
        metrics = evaluate_single_model(
            name,
            X,
            y,
            le,
            cv_folds=cv_folds,
            models_dir=m_dir,
            figures_dir=figures_dir,
            reports_dir=reports_dir,
        )
        all_metrics.append(metrics)

    df = pd.DataFrame(all_metrics)

    # Select display columns
    display_cols = [
        "model_name",
        "cv_accuracy_mean",
        "cv_accuracy_std",
        "cv_f1_macro_mean",
        "cv_f1_macro_std",
        "cv_precision_macro_mean",
        "cv_recall_macro_mean",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    df_display = df[display_cols].sort_values("cv_accuracy_mean", ascending=False)

    # Save comparison CSV
    csv_path = reports_dir / "model_comparison.csv"
    df_display.to_csv(csv_path, index=False)
    logger.info("Saved model comparison → %s", csv_path)

    # Plot comparison bar chart
    _plot_model_comparison(df_display, figures_dir)

    return df_display


def _plot_model_comparison(df: pd.DataFrame, figures_dir: Path) -> None:
    """Generate a grouped bar chart comparing model performance."""
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    models = df["model_name"].str.upper().tolist()
    x = np.arange(len(models))
    width = 0.25

    acc = df["cv_accuracy_mean"].values
    f1 = df["cv_f1_macro_mean"].values

    bars1 = ax.bar(x - width / 2, acc, width, label="CV Accuracy", color="#1976D2", alpha=0.85)
    bars2 = ax.bar(x + width / 2, f1, width, label="Macro F1", color="#388E3C", alpha=0.85)

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.3f}", ha="center", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.3f}", ha="center", fontsize=8)

    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title(
        "Disease Model Comparison — CV Accuracy & Macro F1",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()

    out_path = figures_dir / "model_comparison.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info("Saved model comparison chart → %s", out_path)


def run_evaluation(
    features_path: Path | str | None = None,
    models_dir: Path | str | None = None,
    cv_folds: int | None = None,
) -> None:
    """CLI entry point: evaluate all trained models and generate reports.

    Parameters
    ----------
    features_path : Path | str | None
        Path to feature CSV.
    models_dir : Path | str | None
        Path to models directory.
    cv_folds : int | None
        Number of CV folds to run.
    """
    set_global_seed()
    logger.info("=" * 60)
    logger.info("CropCare AI — Model Evaluation")
    logger.info("=" * 60)
    compare_models(features_path=features_path, models_dir=models_dir, cv_folds=cv_folds)
    logger.info("Evaluation complete.")
