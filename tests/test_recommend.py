"""
CropCare AI — Unit tests for fertilizer recommendation engine and rule layer.
"""

from __future__ import annotations

from src.recommend import (
    FertilizerRecommendation,
    _normalize_disease_key,
    assess_nutrient_levels,
    recommend_fertilizer,
)


class TestDiseaseKeyNormalization:
    """Tests for disease key matching helper."""

    def test_exact_match(self) -> None:
        keys = ["Apple___Apple_scab", "Tomato___Early_blight"]
        assert _normalize_disease_key("Apple___Apple_scab", "Apple", keys) == "Apple___Apple_scab"

    def test_prefix_match(self) -> None:
        keys = ["Apple___Apple_scab", "Tomato___Early_blight"]
        assert _normalize_disease_key("Early_blight", "Tomato", keys) == "Tomato___Early_blight"

    def test_healthy_match(self) -> None:
        keys = ["Potato___healthy", "Tomato___healthy"]
        assert _normalize_disease_key("healthy", "Potato", keys) == "Potato___healthy"

    def test_no_match_returns_none(self) -> None:
        keys = ["Apple___Apple_scab"]
        assert _normalize_disease_key("unknown_disease", "Wheat", keys) is None


class TestAssessNutrientLevels:
    """Tests for farmer NPK soil test evaluation."""

    def test_nutrient_assessment(self) -> None:
        rules = {
            "crop_base_recommendations": {
                "Tomato": {
                    "N_range_kg_per_ha": [100, 150],
                    "P_range_kg_per_ha": [50, 80],
                    "K_range_kg_per_ha": [80, 120],
                    "notes": "Tomato notes",
                }
            }
        }
        res = assess_nutrient_levels("Tomato", 80.0, 60.0, 140.0, rules)
        assert "Low" in res["N_status"]
        assert "Optimal" in res["P_status"]
        assert "High" in res["K_status"]

    def test_unknown_crop_nutrient_assessment(self) -> None:
        res = assess_nutrient_levels("Dragonfruit", 50.0, 50.0, 50.0, {})
        assert "Unknown" in res["N_status"]


class TestRecommendFertilizer:
    """Tests for end-to-end recommendation function."""

    def test_recommendation_structure(self) -> None:
        rec = recommend_fertilizer(
            crop="Tomato",
            disease="Tomato___Early_blight",
            soil="Loamy",
            N=40.0,
            P=30.0,
            K=20.0,
            temp=28.0,
            humidity=60.0,
            moisture=35.0,
        )

        assert isinstance(rec, FertilizerRecommendation)
        assert rec.crop == "Tomato"
        assert rec.soil_type == "Loamy"
        assert rec.base_fertilizer_ml != ""
        assert rec.base_fertilizer_confidence > 0.0
        assert isinstance(rec.to_dict(), dict)
        assert len(rec.disclaimer) > 0

    def test_disease_adjustments_for_blight(self) -> None:
        rec = recommend_fertilizer(
            crop="Tomato",
            disease="Tomato___Early_blight",
            soil="Red",
            N=120.0,
            P=60.0,
            K=80.0,
            temp=25.0,
            humidity=75.0,
            moisture=45.0,
        )

        assert rec.disease_adjustments is not None
        adj = rec.disease_adjustments.get("adjustments", {})
        # Blight rule prescribes reducing nitrogen or increasing potassium
        assert "nitrogen" in adj or "action_on_nitrogen" in str(adj)

    def test_healthy_crop_recommendation(self) -> None:
        rec = recommend_fertilizer(
            crop="Apple",
            disease="Apple___healthy",
            soil="Loamy",
            N=90.0,
            P=45.0,
            K=60.0,
            temp=22.0,
            humidity=55.0,
            moisture=40.0,
        )

        assert rec.crop == "Apple"
        assert "healthy" in rec.disease_class.lower()

    def test_soil_guidance_present(self) -> None:
        rec = recommend_fertilizer(
            crop="Potato",
            disease="Potato___Late_blight",
            soil="Sandy",
            N=50.0,
            P=40.0,
            K=60.0,
            temp=20.0,
            humidity=80.0,
            moisture=50.0,
        )

        assert rec.soil_type == "Sandy"
        assert "nutrient_retention" in rec.soil_guidance or len(rec.soil_guidance) > 0
