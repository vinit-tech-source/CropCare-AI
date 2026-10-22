"""
CropCare AI — Fertilizer recommendation with disease-aware rule layer.

Combines ML-predicted fertilizer from soil/climate parameters with
disease-specific nutrient adjustments, crop base requirements, soil-type
considerations, and agronomic extension guidelines from the knowledge base.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import get_config, resolve_path
from src.train_fertilizer import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    load_fertilizer_pipeline,
)
from src.utils import load_json

logger = logging.getLogger(__name__)

# Canonical mapping for aliases
CROP_ALIASES: dict[str, str] = {
    "corn": "Corn",
    "maize": "Corn",
    "apple": "Apple",
    "potato": "Potato",
    "tomato": "Tomato",
}

# Mapping dataset crops to fertilizer model dataset crops if needed
FERTILIZER_CROP_MAP: dict[str, str] = {
    "Corn": "Maize",
    "Maize": "Maize",
    "Apple": "Maize",  # Fallback for crops not explicitly in 11-crop fertilizer tabular dataset
    "Potato": "Pulses",  # Fallback root/tuber profile
    "Tomato": "Tobacco",  # Solanaceous family proxy
}

FERTILIZER_DOSAGE_ALIASES: dict[str, str] = {
    "10-26-26": "NPK 10-26-26",
    "20-20": "NPK 20-20-20",
    "17-17-17": "NPK 20-20-20",
    "28-28": "NPK 20-20-20",
    "14-35-14": "DAP",
}


@dataclass
class FertilizerRecommendation:
    """Complete fertilizer and agronomic recommendation package."""

    crop: str
    disease_class: str
    disease_name: str
    soil_type: str
    soil_parameters: dict[str, float]
    base_fertilizer_ml: str
    base_fertilizer_confidence: float
    top_fertilizers: list[dict[str, Any]] = field(default_factory=list)
    nutrient_status: dict[str, Any] = field(default_factory=dict)
    disease_adjustments: dict[str, Any] = field(default_factory=dict)
    prescription_details: dict[str, Any] = field(default_factory=dict)
    soil_guidance: dict[str, Any] = field(default_factory=dict)
    dosage_guide: dict[str, Any] = field(default_factory=dict)
    disease_management: dict[str, Any] = field(default_factory=dict)
    disclaimer: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert recommendation to a plain JSON-serializable dictionary."""
        return asdict(self)


def _load_knowledge_bases(
    rules_path: Path | str | None = None,
    prescriptions_path: Path | str | None = None,
    diseases_path: Path | str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load knowledge base JSON files."""
    cfg = get_config()
    kb_rel = cfg.get("paths", {}).get("knowledge_base_dir", "data/knowledge_base")
    kb_dir = resolve_path(kb_rel)

    r_path = Path(rules_path).resolve() if rules_path else kb_dir / "fertilizer_rules.json"
    p_path = (
        Path(prescriptions_path).resolve()
        if prescriptions_path
        else kb_dir / "disease_fertilizer_prescriptions.json"
    )
    d_path = Path(diseases_path).resolve() if diseases_path else kb_dir / "diseases.json"

    rules = load_json(r_path) if r_path.exists() else {}
    prescriptions = load_json(p_path) if p_path.exists() else {}
    diseases = load_json(d_path) if d_path.exists() else {}

    return rules, prescriptions, diseases


def _normalize_disease_key(
    disease: str,
    crop: str,
    available_keys: list[str],
) -> str | None:
    """Resolve user- or model-provided disease string to a canonical key."""
    if not disease:
        return None

    # 1. Exact match
    if disease in available_keys:
        return disease

    # 2. Match with crop prefix
    crop_prefixed = f"{crop}___{disease}"
    if crop_prefixed in available_keys:
        return crop_prefixed

    # 3. Normalized case-insensitive comparison
    d_clean = disease.lower().replace(" ", "_").replace("-", "_")
    c_clean = crop.lower()

    for k in available_keys:
        k_clean = k.lower().replace("-", "_")
        if d_clean in k_clean and (c_clean in k_clean or not crop):
            return k

    # 4. Healthy check
    if "healthy" in d_clean:
        for k in available_keys:
            if "healthy" in k.lower() and c_clean in k.lower():
                return k

    return None


def assess_nutrient_levels(
    crop: str,
    n_val: float,
    p_val: float,
    k_val: float,
    rules: dict[str, Any],
) -> dict[str, Any]:
    """Assess farmer soil test N-P-K against crop base requirements."""
    base_recs = rules.get("crop_base_recommendations", {}).get(crop, {})
    if not base_recs:
        return {
            "N_status": "Unknown (no baseline for crop)",
            "P_status": "Unknown (no baseline for crop)",
            "K_status": "Unknown (no baseline for crop)",
            "notes": "Follow regional agricultural university package of practices.",
        }

    n_range = base_recs.get("N_range_kg_per_ha", [50, 150])
    p_range = base_recs.get("P_range_kg_per_ha", [30, 80])
    k_range = base_recs.get("K_range_kg_per_ha", [40, 100])

    def _eval(val: float, rng: list[float]) -> str:
        if val < rng[0]:
            return f"Low (Input: {val:.1f}, Target: {rng[0]}–{rng[1]} kg/ha)"
        if val > rng[1]:
            return f"High (Input: {val:.1f}, Target: {rng[0]}–{rng[1]} kg/ha)"
        return f"Optimal (Input: {val:.1f}, Target: {rng[0]}–{rng[1]} kg/ha)"

    return {
        "N_status": _eval(n_val, n_range),
        "P_status": _eval(p_val, p_range),
        "K_status": _eval(k_val, k_range),
        "optimal_ranges": {
            "N_kg_per_ha": n_range,
            "P_kg_per_ha": p_range,
            "K_kg_per_ha": k_range,
        },
        "agronomic_notes": base_recs.get("notes", ""),
    }


def recommend_fertilizer(
    crop: str,
    disease: str,
    soil: str,
    N: float,
    P: float,
    K: float,
    temp: float,
    humidity: float,
    moisture: float,
    model_path: Path | str | None = None,
    rules_path: Path | str | None = None,
    prescriptions_path: Path | str | None = None,
    diseases_path: Path | str | None = None,
) -> FertilizerRecommendation:
    """Generate disease-aware fertilizer and agronomic recommendations.

    Args:
        crop: Crop name (e.g. 'Tomato', 'Apple', 'Corn', 'Potato').
        disease: Detected disease class or identifier.
        soil: Soil type (e.g. 'Loamy', 'Sandy', 'Clayey', 'Black', 'Red').
        N: Soil nitrogen level.
        P: Soil phosphorus level.
        K: Soil potassium level.
        temp: Temperature in °C.
        humidity: Relative humidity in %.
        moisture: Soil moisture percentage.
        model_path: Optional path to fertilizer model artifact (.joblib).
        rules_path: Optional path to fertilizer_rules.json.
        prescriptions_path: Optional path to disease_fertilizer_prescriptions.json.
        diseases_path: Optional path to diseases.json.

    Returns:
        FertilizerRecommendation: Comprehensive agronomic recommendation.
    """
    rules, prescriptions_kb, diseases_kb = _load_knowledge_bases(
        rules_path=rules_path,
        prescriptions_path=prescriptions_path,
        diseases_path=diseases_path,
    )

    norm_crop = CROP_ALIASES.get(crop.strip().lower(), crop.strip().capitalize())
    fert_crop = FERTILIZER_CROP_MAP.get(norm_crop, norm_crop)
    soil_type = soil.strip().capitalize()

    # 1. Base ML model prediction
    base_fertilizer = "NPK 20-20-20"
    base_confidence = 1.0
    top_candidates: list[dict[str, Any]] = []

    try:
        pipeline, classes = load_fertilizer_pipeline(model_path)
        input_data = pd.DataFrame(
            [
                {
                    "Temparature": float(temp),
                    "Humidity": float(humidity),
                    "Moisture": float(moisture),
                    "Soil Type": soil_type,
                    "Crop Type": fert_crop,
                    "Nitrogen": float(N),
                    "Potassium": float(K),
                    "Phosphorous": float(P),
                }
            ],
            columns=NUMERICAL_FEATURES + CATEGORICAL_FEATURES,
        )

        preds = pipeline.predict(input_data)
        base_fertilizer = str(preds[0])

        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(input_data)[0]
            base_confidence = float(np.max(probs))
            top_indices = np.argsort(probs)[::-1][:3]
            top_candidates = [
                {"fertilizer": str(classes[idx]), "confidence": round(float(probs[idx]), 4)}
                for idx in top_indices
            ]
    except Exception as exc:
        logger.warning(
            "Could not run ML fertilizer model (%s). Using fallback recommendation.",
            exc,
        )
        base_fertilizer = "NPK 20-20-20"
        base_confidence = 0.5
        top_candidates = [{"fertilizer": base_fertilizer, "confidence": 0.5}]

    # 2. Disease identification & prescription matching
    available_prescriptions = list(prescriptions_kb.get("prescriptions", {}).keys())
    matched_key = _normalize_disease_key(disease, norm_crop, available_prescriptions)

    prescription_entry: dict[str, Any] = {}
    disease_management: dict[str, Any] = {}
    display_disease_name = disease

    if matched_key:
        prescription_entry = prescriptions_kb.get("prescriptions", {}).get(matched_key, {})
        display_disease_name = prescription_entry.get("disease_name", matched_key)
        # Load management from diseases.json
        d_info = diseases_kb.get("diseases", {}).get(matched_key, {})
        if d_info:
            disease_management = {
                "pathogen": d_info.get("pathogen", ""),
                "symptoms": d_info.get("symptoms", []),
                "organic_treatment": d_info.get("organic_treatment", []),
                "chemical_treatment": d_info.get("chemical_treatment", []),
                "prevention": d_info.get("prevention", []),
            }

    # 3. Disease-specific nutrient adjustments from rules
    disease_adjustments: dict[str, Any] = {}
    for rule_group, details in rules.get("disease_nutrient_adjustments", {}).items():
        if rule_group == "_description":
            continue
        applies_to = details.get("applies_to", [])
        if matched_key in applies_to:
            disease_adjustments = {
                "rule_group": rule_group,
                "adjustments": details.get("adjustments", {}),
                "reason": details.get("reason", ""),
                "specific_advice": details.get("specific_advice", ""),
            }
            break

    # If no group matched but prescription has actions, use prescription actions
    if not disease_adjustments and prescription_entry:
        fert_rec = prescription_entry.get("fertilizer_recommendation", {})
        disease_adjustments = {
            "rule_group": "specific_prescription",
            "adjustments": {
                "nitrogen": fert_rec.get("action_on_nitrogen", "Maintain"),
                "potassium": fert_rec.get("action_on_potassium", "Maintain"),
                "phosphorus": fert_rec.get("action_on_phosphorus", "Maintain"),
            },
            "reason": prescription_entry.get("scientific_rationale", ""),
            "specific_advice": fert_rec.get("primary_soil_fertilizer", ""),
        }

    # 4. Crop base nutrient assessment
    nutrient_status = assess_nutrient_levels(norm_crop, N, P, K, rules)

    # 5. Soil type considerations
    soil_guidance = rules.get("soil_type_considerations", {}).get(soil_type, {})

    # 6. Dosage guidelines for the predicted fertilizer
    dosage_lookup_key = FERTILIZER_DOSAGE_ALIASES.get(base_fertilizer, base_fertilizer)
    general_dosage = rules.get("general_dosage_guidelines", {}).get(dosage_lookup_key, {})
    if not general_dosage and base_fertilizer in rules.get("general_dosage_guidelines", {}):
        general_dosage = rules.get("general_dosage_guidelines", {})[base_fertilizer]

    # Combine with prescription dosage if present
    dosage_guide = {
        "general": general_dosage,
        "disease_specific": prescription_entry.get("dosage_and_application", {}),
    }

    disclaimer = rules.get(
        "safety_disclaimer",
        "⚠️ DISCLAIMER: Recommendations are for educational and advisory reference only. "
        "Consult local agricultural extension specialists before applying treatments.",
    )

    return FertilizerRecommendation(
        crop=norm_crop,
        disease_class=matched_key or disease,
        disease_name=display_disease_name,
        soil_type=soil_type,
        soil_parameters={
            "N": float(N),
            "P": float(P),
            "K": float(K),
            "temperature": float(temp),
            "humidity": float(humidity),
            "moisture": float(moisture),
        },
        base_fertilizer_ml=base_fertilizer,
        base_fertilizer_confidence=round(base_confidence, 4),
        top_fertilizers=top_candidates,
        nutrient_status=nutrient_status,
        disease_adjustments=disease_adjustments,
        prescription_details=prescription_entry,
        soil_guidance=soil_guidance,
        dosage_guide=dosage_guide,
        disease_management=disease_management,
        disclaimer=disclaimer,
    )
