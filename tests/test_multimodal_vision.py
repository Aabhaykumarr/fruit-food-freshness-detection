"""
Tests for Multimodal Vision Pipeline (OpenAI Vision structured parsing, multi-item enumeration,
condition assessment, non-food rejection, uncertainty calibration, and fallback integration).
"""

import json
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image
import numpy as np

from services.vision_service import (
    VisionService,
    MultimodalVisionService,
    SigLIPVisionService,
    ItemVisionResult,
    ImageVisionResult,
)
from services.evidence_aggregator import EvidenceAggregator
from services.llm_service import LLMFreshnessService, FreshnessAnalysis


# =============================================================================
# 1. TEST PYDANTIC SCHEMAS & NORMALIZATION
# =============================================================================

def test_item_vision_result_schema():
    """Verify ItemVisionResult validates all required fields."""
    item = ItemVisionResult(
        item_id=1,
        name="bread",
        category="bakery",
        confidence=96.5,
        condition="moldy",
        condition_confidence=94.0,
        freshness_status="VISIBLE_SPOILAGE",
        visible_signs=["green fuzzy mold colonies on top crust", "white mycelium patches"],
        visual_description="White sliced bread with visible green and white mold colonies across the surface.",
        is_spoiled=True,
        reason="Extensive fungal growth makes the bread unsafe for consumption.",
        alternatives=[],
    )
    assert item.item_id == 1
    assert item.name == "bread"
    assert item.category == "bakery"
    assert item.is_spoiled is True
    assert item.freshness_status == "VISIBLE_SPOILAGE"
    assert len(item.visible_signs) == 2


def test_image_vision_result_multi_item_schema():
    """Verify ImageVisionResult handles multiple items correctly."""
    item1 = ItemVisionResult(
        item_id=1,
        name="bread",
        category="bakery",
        confidence=98.0,
        condition="moldy",
        condition_confidence=95.0,
        freshness_status="VISIBLE_SPOILAGE",
        visible_signs=["green mold spots"],
        visual_description="Slice of bread with mold.",
        is_spoiled=True,
        reason="Fungal mold growth.",
        alternatives=[],
    )
    item2 = ItemVisionResult(
        item_id=2,
        name="tomato",
        category="vegetable",
        confidence=94.0,
        condition="fresh/ripe",
        condition_confidence=90.0,
        freshness_status="FRESH",
        visible_signs=["firm smooth red skin", "green calyx"],
        visual_description="Fresh red ripe whole tomato.",
        is_spoiled=False,
        reason="No defects or bruising visible.",
        alternatives=[],
    )
    img_res = ImageVisionResult(
        is_food=True,
        overall_status="food",
        items=[item1, item2],
        alternatives=[],
        overall_visual_summary="A multi-item food scene containing moldy bread and a fresh ripe tomato.",
        model_notes="Clear indoor lighting.",
    )

    assert img_res.is_food is True
    assert img_res.overall_status == "food"
    assert len(img_res.items) == 2
    assert img_res.items[0].name == "bread"
    assert img_res.items[0].is_spoiled is True
    assert img_res.items[1].name == "tomato"
    assert img_res.items[1].is_spoiled is False


def test_non_food_rejection_schema():
    """Verify ImageVisionResult schema for non-food rejection."""
    non_food_res = ImageVisionResult(
        is_food=False,
        overall_status="not_food",
        items=[],
        alternatives=["smartphone", "electronic device"],
        overall_visual_summary="An electronic smartphone resting on a wooden table.",
        model_notes=None,
    )
    assert non_food_res.is_food is False
    assert non_food_res.overall_status == "not_food"
    assert len(non_food_res.items) == 0


# =============================================================================
# 2. TEST MULTIMODAL VISION SERVICE MOCKED INFERENCE
# =============================================================================

def test_multimodal_vision_multi_item_enumeration():
    """
    CRITICAL REGRESSION TEST:
    Verifies that a grouped scene with moldy bread, tomato, and onion is NOT called a sandwich.
    Enumerate 3 items with independent conditions.
    """
    mock_client = MagicMock()
    mock_parsed_result = ImageVisionResult(
        is_food=True,
        overall_status="food",
        items=[
            ItemVisionResult(
                item_id=1,
                name="bread",
                category="bakery",
                confidence=95.0,
                condition="moldy",
                condition_confidence=93.0,
                freshness_status="VISIBLE_SPOILAGE",
                visible_signs=["green and white fungal colonies across surface"],
                visual_description="White sliced loaf bread showing widespread fuzzy mold colonies.",
                is_spoiled=True,
                reason="Visible fungal decay and mycelium network.",
                alternatives=[],
            ),
            ItemVisionResult(
                item_id=2,
                name="tomato",
                category="vegetable",
                confidence=92.0,
                condition="fresh/ripe",
                condition_confidence=88.0,
                freshness_status="FRESH",
                visible_signs=["smooth taut red skin", "green stem intact"],
                visual_description="Whole ripe red tomato without wrinkles or bruises.",
                is_spoiled=False,
                reason="Firm, glossy skin and normal coloration.",
                alternatives=[],
            ),
            ItemVisionResult(
                item_id=3,
                name="onion",
                category="vegetable",
                confidence=90.0,
                condition="fresh",
                condition_confidence=85.0,
                freshness_status="FRESH",
                visible_signs=["dry papery outer skin"],
                visual_description="Whole raw red onion with dry intact outer layers.",
                is_spoiled=False,
                reason="Firm dry bulb with no sprouting or softness.",
                alternatives=[],
            ),
        ],
        alternatives=[],
        overall_visual_summary="Multi-item produce and bakery scene containing moldy bread, fresh tomato, and fresh onion.",
        model_notes="Individual food items placed side-by-side.",
    )

    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(parsed=mock_parsed_result))]
    mock_client.beta.chat.completions.parse.return_value = mock_completion

    service = MultimodalVisionService(api_key="mock-test-key")
    service.client = mock_client

    img = Image.new("RGB", (300, 300), color="white")
    result = service.analyze_image(img)

    # Assertions
    assert result["status"] == "recognized"
    assert result["source"] == "multimodal_openai"
    assert len(result["items"]) == 3
    # Check that it is NOT classified as sandwich
    assert result["name"] != "sandwich"
    assert result["items"][0]["name"] == "bread"
    assert result["items"][0]["is_spoiled"] is True
    assert result["items"][0]["freshness_status"] == "VISIBLE_SPOILAGE"
    assert result["items"][1]["name"] == "tomato"
    assert result["items"][1]["is_spoiled"] is False
    assert result["items"][2]["name"] == "onion"
    assert result["items"][2]["is_spoiled"] is False


def test_multimodal_vision_non_food_rejection():
    """Verify non-food objects return status='not_food' with items=[]."""
    mock_client = MagicMock()
    mock_parsed_result = ImageVisionResult(
        is_food=False,
        overall_status="not_food",
        items=[],
        alternatives=["laptop", "keyboard"],
        overall_visual_summary="A silver laptop keyboard and display screen.",
        model_notes=None,
    )
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(parsed=mock_parsed_result))]
    mock_client.beta.chat.completions.parse.return_value = mock_completion

    service = MultimodalVisionService(api_key="mock-test-key")
    service.client = mock_client

    img = Image.new("RGB", (200, 200), color="black")
    result = service.analyze_image(img)

    assert result["status"] == "not_food"
    assert result["name"] is None
    assert result["is_food"] is False
    assert len(result["items"]) == 0
    assert "laptop" in result["overall_visual_summary"].lower()


def test_multimodal_vision_uncertain_food():
    """Verify unfamiliar / out-of-distribution food returns status='uncertain'."""
    mock_client = MagicMock()
    mock_parsed_result = ImageVisionResult(
        is_food=True,
        overall_status="uncertain",
        items=[
            ItemVisionResult(
                item_id=1,
                name="uncertain",
                category="fruit",
                confidence=60.0,
                condition="fresh",
                condition_confidence=75.0,
                freshness_status="CANNOT_DETERMINE",
                visible_signs=["red hairy rind with soft spines"],
                visual_description="Exotic tropical fruit with reddish-orange spiky rind resembling rambutan or lychee.",
                is_spoiled=False,
                reason="Uncommon regional fruit variety.",
                alternatives=["rambutan", "lychee", "mamoncillo"],
            )
        ],
        alternatives=["rambutan", "lychee"],
        overall_visual_summary="An exotic tropical fruit with spiky red rind.",
        model_notes="Uncommon produce item.",
    )
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(parsed=mock_parsed_result))]
    mock_client.beta.chat.completions.parse.return_value = mock_completion

    service = MultimodalVisionService(api_key="mock-test-key")
    service.client = mock_client

    img = Image.new("RGB", (200, 200), color="red")
    result = service.analyze_image(img)

    assert result["status"] == "uncertain"
    assert result["name"] is None
    assert len(result["items"]) == 1
    assert "rambutan" in result["items"][0]["alternatives"]


# =============================================================================
# 3. TEST EVIDENCE AGGREGATION WITH MULTI-ITEM VISION
# =============================================================================

def test_evidence_aggregator_multi_item_fusion():
    """Verify EvidenceAggregator enriches multi-item vision results with per-item rules & telemetry."""
    aggregator = EvidenceAggregator()
    vision_res = {
        "status": "recognized",
        "name": "bread",
        "category": "bakery",
        "confidence": 0.95,
        "visual_description": "Moldy bread next to fresh apple.",
        "overall_visual_summary": "Moldy bread next to fresh apple.",
        "source": "multimodal_openai",
        "is_food": True,
        "items": [
            {
                "item_id": 1,
                "name": "bread",
                "category": "bakery",
                "confidence": 95.0,
                "condition": "moldy",
                "condition_confidence": 92.0,
                "freshness_status": "VISIBLE_SPOILAGE",
                "visible_signs": ["green mold"],
                "visual_description": "Moldy white bread",
                "is_spoiled": True,
                "reason": "Mold growth",
                "alternatives": [],
            },
            {
                "item_id": 2,
                "name": "apple",
                "category": "fruit",
                "confidence": 93.0,
                "condition": "fresh/ripe",
                "condition_confidence": 90.0,
                "freshness_status": "FRESH",
                "visible_signs": ["crisp red skin"],
                "visual_description": "Fresh whole red apple",
                "is_spoiled": False,
                "reason": "No blemishes",
                "alternatives": [],
            },
        ],
    }

    sensor_reading = {
        "temperature_c": 24.0,
        "humidity_percent": 60.0,
        "gas_value": 250,
        "sensor_status": "valid",
        "sensor_scope": "environment",
        "source": "arduino_serial",
    }

    packet = aggregator.aggregate(vision_res, sensor_reading)

    assert packet["status"] == "recognized"
    assert packet["is_food"] is True
    assert len(packet["items_evidence"]) == 2
    assert packet["items_evidence"][0]["name"] == "bread"
    assert packet["items_evidence"][0]["is_spoiled"] is True
    assert packet["items_evidence"][1]["name"] == "apple"
    assert packet["items_evidence"][1]["is_spoiled"] is False
    assert packet["items_evidence"][1]["storage_guidelines"] is not None
    assert "typical_shelf_life_days_room_temp" in packet["items_evidence"][1]["storage_guidelines"]
    assert packet["sensor_evidence"]["temperature_c"] == 24.0


# =============================================================================
# 4. TEST LLM REASONING MULTI-ITEM SYNTHESIS & RULE-BASED FALLBACK
# =============================================================================

def test_llm_rule_based_fallback_with_multi_item_evidence():
    """Verify rule-based fallback accurately identifies spoiled items in a multi-item packet."""
    llm_service = LLMFreshnessService(api_key="")  # Offline mode

    evidence_packet = {
        "status": "recognized",
        "is_food": True,
        "visual_evidence": {
            "status": "recognized",
            "item_name": "bread",
            "confidence": 0.95,
            "visual_description": "Moldy bread with fresh vegetables.",
        },
        "sensor_evidence": {
            "temperature_c": 26.5,
            "humidity_percent": 75.0,
            "sensor_status": "valid",
        },
        "knowledge_guidelines": {
            "typical_shelf_life_days_room_temp": "2-4",
        },
        "items_evidence": [
            {
                "item_id": 1,
                "name": "bread",
                "category": "bakery",
                "condition": "moldy",
                "is_spoiled": True,
                "visual_description": "Bread with green mold",
            },
            {
                "item_id": 2,
                "name": "tomato",
                "category": "vegetable",
                "condition": "fresh/ripe",
                "is_spoiled": False,
                "visual_description": "Fresh ripe red tomato",
                "storage_guidelines": {"typical_shelf_life_days_room_temp": "3-7"},
            },
        ],
    }

    analysis = llm_service.analyze(evidence_packet)

    assert analysis.freshness_status == "NOT_FRESH"
    assert len(analysis.item_summaries) == 2
    assert analysis.item_summaries[0].item_name == "bread"
    assert analysis.item_summaries[0].is_spoiled is True
    assert analysis.item_summaries[1].item_name == "tomato"
    assert analysis.item_summaries[1].is_spoiled is False


# =============================================================================
# 5. TEST UNIFIED VISION SERVICE ROUTER & FALLBACK
# =============================================================================

def test_unified_vision_service_routes_to_siglip_when_no_api_key():
    """Verify VisionService uses SigLIP fallback when no GEMINI_API_KEY or OPENAI_API_KEY is configured."""
    VisionService._instance = None
    with patch("services.vision_service.GEMINI_API_KEY", ""), \
         patch("services.vision_service.OPENAI_API_KEY", ""):
        service = VisionService()
        service.gemini_service.client = None
        if service.openai_service:
            service.openai_service.client = None
        img = Image.new("RGB", (224, 224), color="green")
        res = service.analyze_image(img)
        assert res["source"] == "local_siglip_fallback"
        assert "status" in res
        assert "items" in res
    VisionService._instance = None
