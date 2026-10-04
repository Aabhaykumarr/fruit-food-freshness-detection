"""Unit tests for LLM response schema validation and rule fallback."""

from services.llm_service import LLMFreshnessService, FreshnessAnalysis, RemainingFreshness


def test_rule_based_fallback():
    service = LLMFreshnessService(api_key="")  # Offline mode

    evidence = {
        "visual_evidence": {"item_name": "apple", "confidence": 0.92},
        "sensor_evidence": {"temperature_c": 22.0, "humidity_percent": 60.0, "sensor_status": "valid"},
        "knowledge_guidelines": {"typical_shelf_life_days_room_temp": "3-7"},
    }

    result = service.analyze(evidence)

    assert isinstance(result, FreshnessAnalysis)
    assert result.item_name == "apple"
    assert result.freshness_status in ["FRESH", "MODERATELY_FRESH", "QUESTIONABLE", "NOT_FRESH", "UNKNOWN"]
    assert isinstance(result.estimated_remaining_freshness, RemainingFreshness)
    assert len(result.uncertainty_factors) > 0


def test_schema_serialization():
    analysis = FreshnessAnalysis(
        item_name="mango",
        detection_confidence=0.91,
        freshness_status="FRESH",
        estimated_remaining_freshness=RemainingFreshness(
            value="3-5",
            unit="days",
            range_description="3 to 5 days at room temperature",
            confidence="medium",
            basis=["Visual appearance", "Ambient temperature"],
        ),
        visual_observations=["No mold detected"],
        environmental_context="Temperature 23C is normal",
        reasoning_summary="Fresh mango in good environmental condition",
        uncertainty_factors=["Sensors measure ambient room only"],
        recommendations=["Eat or refrigerate within 4 days"],
    )

    json_str = analysis.model_dump_json()
    assert "mango" in json_str
    assert "FRESH" in json_str
