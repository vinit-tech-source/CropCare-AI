"""
CropCare AI — Prediction pipeline, demo functions, and CLI presentation.

End-to-end pipeline:
1. Load & preprocess leaf image (resize to 256x256, Gaussian blur).
2. Segment leaf & disease regions using KMeans clustering in Lab color space.
3. Extract 106-dimensional hand-crafted feature vector (HSV, GLCM, LBP, Hu, Severity).
4. Classify disease using trained classical ML pipeline (StandardScaler -> PCA -> Classifier).
5. Generate fertilizer & agronomic prescription using ML model and rule knowledge base.
6. Display rich interactive terminal output and save overlay visualizations.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.config import get_config, get_prediction_config, resolve_path
from src.features import extract_image_features
from src.preprocessing import load_and_preprocess_image
from src.recommend import FertilizerRecommendation, recommend_fertilizer
from src.segmentation import segment_leaf_and_disease
from src.train_disease import load_trained_pipeline
from src.utils import ensure_dir, load_json

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class DiseasePrediction:
    """Output of the disease classification stage."""

    predicted_class: str
    display_name: str
    confidence: float
    top_k: list[dict[str, Any]] = field(default_factory=list)
    severity_pct: float = 0.0
    is_healthy: bool = False
    low_confidence_warning: bool = False
    model_used: str = ""
    feature_vector: np.ndarray | None = None
    overlay_image: np.ndarray | None = None


@dataclass
class PredictionResult:
    """Full end-to-end output containing disease diagnosis and fertilizer prescription."""

    image_path: str
    disease: DiseasePrediction
    recommendation: FertilizerRecommendation
    overlay_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert complete result to a JSON-serializable dictionary."""
        data = asdict(self)
        if "feature_vector" in data.get("disease", {}):
            data["disease"]["feature_vector"] = None
        if "overlay_image" in data.get("disease", {}):
            data["disease"]["overlay_image"] = None
        return data


def _find_available_disease_model() -> tuple[str, Path] | None:
    """Scan models directory for a trained disease classification pipeline."""
    cfg = get_config()
    models_rel = cfg.get("paths", {}).get("models_dir", "models")
    models_dir = resolve_path(models_rel)

    priority = ["random_forest", "svm", "logistic_regression", "knn"]
    for name in priority:
        candidate = models_dir / f"disease_{name}_pipeline.joblib"
        if candidate.exists():
            return name, candidate
        alt = models_dir / f"{name}_model.joblib"
        if alt.exists():
            return name, alt

    # Check for any pipeline or model joblib except fertilizer
    if models_dir.exists():
        for f in models_dir.glob("*_pipeline.joblib"):
            if "fertilizer" not in f.name:
                model_name = f.stem.replace("disease_", "").replace("_pipeline", "")
                return model_name, f
        for f in models_dir.glob("*_model.joblib"):
            if "fertilizer" not in f.name:
                model_name = f.stem.replace("_model", "")
                return model_name, f

    return None


def predict_disease(
    image_path: Path | str,
    model_name: str | None = None,
    confidence_threshold: float | None = None,
    top_k_count: int | None = None,
) -> DiseasePrediction:
    """Run preprocessing, segmentation, feature extraction, and disease classification.

    Args:
        image_path: Path to the leaf image.
        model_name: Optional classifier name ('svm', 'knn', 'logistic_regression', 'random_forest').
        confidence_threshold: Minimum confidence score before flagging low confidence.
        top_k_count: Number of top candidate classes to return.

    Returns:
        DiseasePrediction: Structured disease diagnosis.

    Raises:
        FileNotFoundError: If image file does not exist.
        ValueError: If image cannot be read or processed.
    """
    img_p = Path(image_path).resolve()
    if not img_p.exists():
        raise FileNotFoundError(f"Image not found at: {img_p}")

    pcfg = get_prediction_config()
    conf_thresh = (
        confidence_threshold
        if confidence_threshold is not None
        else float(pcfg.get("confidence_threshold", 0.50))
    )
    k_count = top_k_count if top_k_count is not None else int(pcfg.get("top_k", 3))

    # 1. Preprocess
    rgb = load_and_preprocess_image(img_p)
    if rgb is None:
        raise ValueError(f"Failed to read or decode image at {img_p}")

    # 2. Segment
    leaf_mask, diseased_mask, overlay, severity_pct = segment_leaf_and_disease(rgb)

    # 3. Extract 106D Features
    features = extract_image_features(rgb, leaf_mask, diseased_mask, severity_pct)

    # 4. Locate model
    if model_name is None:
        found = _find_available_disease_model()
        if found is None:
            logger.warning(
                "No trained disease model found in models/ directory. "
                "Using heuristic / uncalibrated fallback."
            )
            return DiseasePrediction(
                predicted_class="Apple___healthy" if severity_pct < 2.0 else "Apple___Apple_scab",
                display_name="Healthy Plant" if severity_pct < 2.0 else "Suspected Leaf Infection",
                confidence=0.50,
                top_k=[
                    {"class": "Healthy", "confidence": 0.50},
                    {"class": "Infected", "confidence": 0.50},
                ],
                severity_pct=severity_pct,
                is_healthy=severity_pct < 2.0,
                low_confidence_warning=True,
                model_used="heuristic_fallback",
                feature_vector=features,
                overlay_image=overlay,
            )
        resolved_model_name, _ = found
    else:
        resolved_model_name = model_name

    # 5. Classify
    pipeline, label_encoder, _ = load_trained_pipeline(resolved_model_name)
    X = features.reshape(1, -1)

    predicted_encoded = pipeline.predict(X)[0]
    predicted_class = str(label_encoder.inverse_transform([predicted_encoded])[0])

    top_candidates: list[dict[str, Any]] = []
    top_confidence = 1.0

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X)[0]
        top_indices = np.argsort(probabilities)[::-1][:k_count]
        top_confidence = float(probabilities[top_indices[0]])

        for idx in top_indices:
            cls_name = str(label_encoder.inverse_transform([idx])[0])
            top_candidates.append(
                {
                    "class": cls_name,
                    "confidence": round(float(probabilities[idx]), 4),
                }
            )
    else:
        top_candidates.append({"class": predicted_class, "confidence": 1.0})

    is_healthy = "healthy" in predicted_class.lower()
    low_conf = top_confidence < conf_thresh

    # Pretty display name
    clean_name = predicted_class.split("___")[-1].replace("_", " ").strip()

    return DiseasePrediction(
        predicted_class=predicted_class,
        display_name=clean_name,
        confidence=round(top_confidence, 4),
        top_k=top_candidates,
        severity_pct=round(severity_pct, 2),
        is_healthy=is_healthy,
        low_confidence_warning=low_conf,
        model_used=resolved_model_name,
        feature_vector=features,
        overlay_image=overlay,
    )


def predict_crop_disease_and_fertilizer(
    image_path: Path | str,
    crop: str,
    soil: str,
    N: float,
    P: float,
    K: float,
    temp: float,
    humidity: float,
    moisture: float,
    model_name: str | None = None,
    save_overlay: bool = True,
) -> PredictionResult:
    """Execute end-to-end crop diagnosis and fertilizer advisory.

    Args:
        image_path: Path to input leaf image.
        crop: Plant crop type (e.g. 'Tomato', 'Apple', 'Corn', 'Potato').
        soil: Soil type (e.g. 'Loamy', 'Sandy', 'Clayey', 'Black', 'Red').
        N: Soil Nitrogen in kg/ha.
        P: Soil Phosphorus in kg/ha.
        K: Soil Potassium in kg/ha.
        temp: Temperature in °C.
        humidity: Humidity percentage.
        moisture: Soil moisture percentage.
        model_name: Optional disease classifier name.
        save_overlay: Whether to write the segmentation overlay image to disk.

    Returns:
        PredictionResult: Full diagnosis and advisory container.
    """
    img_path = Path(image_path).resolve()

    disease_res = predict_disease(
        image_path=img_path,
        model_name=model_name,
    )

    fert_rec = recommend_fertilizer(
        crop=crop,
        disease=disease_res.predicted_class,
        soil=soil,
        N=N,
        P=P,
        K=K,
        temp=temp,
        humidity=humidity,
        moisture=moisture,
    )

    overlay_saved_path: str | None = None
    if save_overlay and disease_res.overlay_image is not None:
        cfg = get_config()
        out_rel = cfg.get("paths", {}).get("outputs_dir", "outputs")
        out_dir = resolve_path(out_rel) / "overlays"
        ensure_dir(out_dir)
        target = out_dir / f"{img_path.stem}_segmented.png"
        cv2.imwrite(str(target), cv2.cvtColor(disease_res.overlay_image, cv2.COLOR_RGB2BGR))
        overlay_saved_path = str(target)

    return PredictionResult(
        image_path=str(img_path),
        disease=disease_res,
        recommendation=fert_rec,
        overlay_path=overlay_saved_path,
    )


def format_prediction_terminal(result: PredictionResult) -> None:
    """Render terminal report with rich formatting, tables, and colors."""
    dis = result.disease
    rec = result.recommendation

    # Header
    console.print()
    header_text = Text(
        "🌿 CropCare AI — Crop Disease Detection & Agronomic Advisory 🌿",
        style="bold green on black",
        justify="center",
    )
    console.print(Panel(header_text, box=ROUNDED, border_style="bright_green"))

    # Section 1: Leaf & Disease Diagnosis
    diag_table = Table(box=ROUNDED, expand=True, title="🔬 Disease Diagnosis & Vision Pipeline")
    diag_table.add_column("Property", style="cyan", width=26)
    diag_table.add_column("Result", style="white")

    diag_table.add_row("Input Leaf Image", Path(result.image_path).name)
    diag_table.add_row("Crop Tested", rec.crop)

    status_color = "bright_green" if dis.is_healthy else "bright_red"
    diag_table.add_row(
        "Primary Classification",
        Text(f"{dis.predicted_class} ({dis.display_name})", style=f"bold {status_color}"),
    )
    diag_table.add_row("Model Confidence", f"{dis.confidence * 100:.1f}% ({dis.model_used})")
    diag_table.add_row("Lesion Area Severity", f"{dis.severity_pct:.2f}% of leaf area")

    if result.overlay_path:
        diag_table.add_row("Segmentation Overlay", result.overlay_path)

    console.print(diag_table)

    # Top-K Candidates table
    if dis.top_k and len(dis.top_k) > 1:
        cand_table = Table(box=ROUNDED, expand=True, title="📊 Top Alternative Predictions")
        cand_table.add_column("Rank", style="dim", width=8)
        cand_table.add_column("Disease Class", style="cyan")
        cand_table.add_column("Confidence", justify="right", style="yellow")

        for rank, cand in enumerate(dis.top_k, 1):
            cand_table.add_row(str(rank), cand["class"], f"{cand['confidence'] * 100:.1f}%")
        console.print(cand_table)

    # Low confidence warning alert
    if dis.low_confidence_warning:
        console.print(
            Panel(
                "⚠️ [bold yellow]LOW CONFIDENCE WARNING[/bold yellow]: Prediction confidence is below "
                "the 50% reliability threshold. Please take a clearer photo with uniform lighting, "
                "ensuring the leaf fills the majority of the frame.",
                border_style="yellow",
            )
        )

    # Section 2: Soil & Nutrient Status
    soil_table = Table(box=ROUNDED, expand=True, title="🧪 Soil & Environmental Assessment")
    soil_table.add_column("Parameter", style="cyan", width=26)
    soil_table.add_column("Value Tested", style="white", width=18)
    soil_table.add_column("Status / Guidance", style="magenta")

    params = rec.soil_parameters
    n_status = rec.nutrient_status.get("N_status", "N/A")
    p_status = rec.nutrient_status.get("P_status", "N/A")
    k_status = rec.nutrient_status.get("K_status", "N/A")

    soil_table.add_row(
        "Soil Type",
        rec.soil_type,
        rec.soil_guidance.get("nutrient_retention", "Standard retention"),
    )
    soil_table.add_row("Nitrogen (N)", f"{params.get('N', 0):.1f} kg/ha", n_status)
    soil_table.add_row("Phosphorus (P)", f"{params.get('P', 0):.1f} kg/ha", p_status)
    soil_table.add_row("Potassium (K)", f"{params.get('K', 0):.1f} kg/ha", k_status)
    soil_table.add_row(
        "Climate Conditions",
        f"{params.get('temperature', 0):.1f}°C, {params.get('humidity', 0):.0f}% RH",
        f"Soil Moisture: {params.get('moisture', 0):.1f}%",
    )

    console.print(soil_table)

    # Section 3: Fertilizer & Agronomic Prescription
    presc_table = Table(
        box=ROUNDED, expand=True, title="🌾 Disease-Aware Fertilizer Recommendation"
    )
    presc_table.add_column("Element", style="cyan", width=26)
    presc_table.add_column("Prescription & Action", style="white")

    presc_table.add_row(
        "Base Fertilizer (ML)",
        f"[bold bright_green]{rec.base_fertilizer_ml}[/bold bright_green] "
        f"(Model Confidence: {rec.base_fertilizer_confidence * 100:.1f}%)",
    )

    adj = rec.disease_adjustments.get("adjustments", {})
    if adj:
        adj_desc = "\n".join(f"• {k.capitalize()}: {v}" for k, v in adj.items())
        presc_table.add_row("Disease Nutrient Adjustments", adj_desc)

    p_entry = rec.prescription_details.get("fertilizer_recommendation", {})
    if p_entry.get("foliar_micronutrients"):
        foliar_str = "\n".join(f"• {m}" for m in p_entry["foliar_micronutrients"])
        presc_table.add_row("Foliar Micronutrients", foliar_str)

    if p_entry.get("organic_bio_amendments"):
        org_str = "\n".join(f"• {o}" for o in p_entry["organic_bio_amendments"])
        presc_table.add_row("Organic / Bio Amendments", org_str)

    if p_entry.get("fertilizers_to_avoid"):
        avoid_str = "\n".join(f"[red]• {a}[/red]" for a in p_entry["fertilizers_to_avoid"])
        presc_table.add_row("Fertilizers to Avoid", avoid_str)

    if rec.prescription_details.get("scientific_rationale"):
        presc_table.add_row(
            "Scientific Rationale", rec.prescription_details["scientific_rationale"]
        )

    dosage_gen = rec.dosage_guide.get("general", {})
    if dosage_gen.get("general_dosage"):
        presc_table.add_row("Recommended Dosage", dosage_gen["general_dosage"])
    if dosage_gen.get("application"):
        presc_table.add_row("Application Timing", dosage_gen["application"])

    console.print(presc_table)

    # Section 4: Disease Management (if infected)
    mgmt = rec.disease_management
    if mgmt and not dis.is_healthy:
        mgmt_table = Table(
            box=ROUNDED, expand=True, title="🛡️ Integrated Pest & Disease Management (IPM)"
        )
        mgmt_table.add_column("Control Type", style="cyan", width=26)
        mgmt_table.add_column("Recommended Measures", style="white")

        if mgmt.get("pathogen"):
            mgmt_table.add_row("Causal Pathogen", mgmt["pathogen"])
        if mgmt.get("organic_treatment"):
            mgmt_table.add_row(
                "Organic Treatment", "\n".join(f"• {t}" for t in mgmt["organic_treatment"])
            )
        if mgmt.get("chemical_treatment"):
            mgmt_table.add_row(
                "Chemical Fungicide / Spray",
                "\n".join(f"• {c}" for c in mgmt["chemical_treatment"]),
            )
        if mgmt.get("prevention"):
            mgmt_table.add_row(
                "Cultural & Preventive", "\n".join(f"• {p}" for p in mgmt["prevention"])
            )

        console.print(mgmt_table)

    # Disclaimer Panel
    console.print(
        Panel(
            Text(rec.disclaimer, style="italic yellow"),
            box=ROUNDED,
            title="⚠️ Statutory Agronomic Advisory",
            border_style="dim yellow",
        )
    )
    console.print()


def list_classes() -> None:
    """Print all 21 supported crop-disease classes organized by crop."""
    cfg = get_config()
    kb_rel = cfg.get("paths", {}).get("knowledge_base_dir", "data/knowledge_base")
    kb_file = resolve_path(kb_rel) / "diseases.json"

    console.print()
    console.print(
        Panel(
            Text("CropCare AI — Supported Crop & Disease Classes (21 Classes)", style="bold green"),
            box=ROUNDED,
        )
    )

    disease_info = load_json(kb_file).get("diseases", {}) if kb_file.exists() else {}

    crops = ["Apple", "Corn_(maize)", "Potato", "Tomato"]
    for crop in crops:
        display_crop = "Corn (Maize)" if "Corn" in crop else crop
        table = Table(title=f"🌾 {display_crop}", box=ROUNDED, expand=True)
        table.add_column("Class ID", style="cyan", width=36)
        table.add_column("Disease Name", style="white", width=28)
        table.add_column("Pathogen / Type", style="magenta")

        for key, details in disease_info.items():
            if details.get("crop", "").lower() in crop.lower() or crop.lower() in key.lower():
                d_name = details.get("disease", key.split("___")[-1].replace("_", " "))
                pathogen = details.get("pathogen", "Healthy" if "healthy" in key else "Pathogen")
                status = "[green]Healthy Plant[/green]" if "healthy" in key else pathogen
                table.add_row(key, d_name, status)

        console.print(table)
        console.print()


def run_demo() -> None:
    """Run prediction on bundled sample images with realistic environmental parameters."""
    cfg = get_config()
    samples_dir = resolve_path(cfg.get("paths", {}).get("samples_dir", "samples"))
    ensure_dir(samples_dir)

    # Find sample images; if empty, copy representative samples from data/subset
    existing_samples = list(samples_dir.glob("*.jpg")) + list(samples_dir.glob("*.JPG"))
    if not existing_samples:
        console.print(
            "[dim]Populating samples/ with representative images from data/subset...[/dim]"
        )
        subset_dir = resolve_path(cfg.get("dataset", {}).get("subset_dir", "data/subset"))
        demo_targets = [
            ("Tomato___Early_blight", "tomato_early_blight.jpg"),
            ("Apple___Apple_scab", "apple_scab.jpg"),
            ("Corn_(maize)___Common_rust_", "corn_common_rust.jpg"),
            ("Potato___healthy", "potato_healthy.jpg"),
        ]
        for class_dir_name, target_name in demo_targets:
            src_folder = subset_dir / class_dir_name
            if src_folder.exists():
                src_imgs = list(src_folder.glob("*.jpg")) + list(src_folder.glob("*.JPG"))
                if src_imgs:
                    shutil.copy2(src_imgs[0], samples_dir / target_name)

        existing_samples = list(samples_dir.glob("*.jpg")) + list(samples_dir.glob("*.JPG"))

    if not existing_samples:
        console.print("[red]No sample images found in samples/ directory to run demo.[/red]")
        return

    console.print(
        Panel(
            Text(
                f"🚀 Running CropCare AI Demo on {len(existing_samples)} Sample Images",
                style="bold green",
            ),
            box=ROUNDED,
        )
    )

    demo_scenarios = [
        {
            "crop": "Tomato",
            "soil": "Loamy",
            "N": 45.0,
            "P": 35.0,
            "K": 25.0,
            "temp": 28.0,
            "humidity": 65.0,
            "moisture": 40.0,
        },
        {
            "crop": "Apple",
            "soil": "Clayey",
            "N": 70.0,
            "P": 40.0,
            "K": 55.0,
            "temp": 21.0,
            "humidity": 55.0,
            "moisture": 35.0,
        },
        {
            "crop": "Corn",
            "soil": "Black",
            "N": 110.0,
            "P": 60.0,
            "K": 45.0,
            "temp": 30.0,
            "humidity": 70.0,
            "moisture": 50.0,
        },
        {
            "crop": "Potato",
            "soil": "Sandy",
            "N": 95.0,
            "P": 50.0,
            "K": 75.0,
            "temp": 24.0,
            "humidity": 60.0,
            "moisture": 42.0,
        },
    ]

    for idx, img_path in enumerate(existing_samples):
        scen = demo_scenarios[idx % len(demo_scenarios)]
        console.print(
            f"[bold cyan]Demo Case {idx + 1}/{len(existing_samples)}: {img_path.name}[/bold cyan]"
        )
        res = predict_crop_disease_and_fertilizer(
            image_path=img_path,
            crop=scen["crop"],
            soil=scen["soil"],
            N=scen["N"],
            P=scen["P"],
            K=scen["K"],
            temp=scen["temp"],
            humidity=scen["humidity"],
            moisture=scen["moisture"],
        )
        format_prediction_terminal(res)


def run_prediction(args: argparse.Namespace) -> None:
    """Entry point invoked from CLI 'predict' subcommand."""
    result = predict_crop_disease_and_fertilizer(
        image_path=args.image,
        crop=args.crop,
        soil=args.soil,
        N=args.N,
        P=args.P,
        K=args.K,
        temp=args.temp,
        humidity=args.humidity,
        moisture=args.moisture,
    )
    format_prediction_terminal(result)
