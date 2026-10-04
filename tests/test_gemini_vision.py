"""
Comprehensive Mocked Test Suite for Google Gemini Multimodal Vision Integration.
Tests cover:
1. GeminiVisionService initialization and configuration
2. Single item recognition (apple, banana, bread)
3. Spoiled & moldy food detection with condition annotations
4. Multi-item scene enumeration without single-dish collapsing
5. Non-food rejection with items=[]
6. Uncertain / out-of-distribution food handling
7. Malformed JSON response retry & recovery
8. VisionService coordinator routing & priority (Gemini -> OpenAI -> SigLIP)
9. Seamless fallback to SigLIP when Gemini fails (quota, network, exceptions)
10. LLMFreshnessService reasoning & chat with Gemini backend
"""

import json
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image
import numpy as np

from services.vision_service import (
    VisionService,
    GeminiVisionService,
    MultimodalVisionService,
    SigLIPVisionService,
    ItemVisionResult,
    ImageVisionResult,
)
from services.llm_service import LLMFreshnessService, FreshnessAnalysis


# =============================================================================
# 1. TEST GEMINI VISION SERVICE INITIALIZATION & CONFIG
# =============================================================================

def test_gemini_vision_service_init_default():
    """Verify GeminiVisionService initializes with correct defaults."""
    with patch("google.genai.Client") as mock_client_cls:
        service = GeminiVisionService(api_key="test-gemini-key", model="gemini-3.8-flash")
        assert service.api_key == "test-gemini-key"
        assert service.model == "gemini-3.8-flash"
        assert service.client is not None
        mock_client_cls.assert_called_once_with(api_key="test-gemini-key")


def test_gemini_vision_service_no_key():
    """Verify GeminiVisionService handles missing API key without crashing."""
    service = GeminiVisionService(api_key="")
    assert service.api_key == ""
    assert service.client is None
    with pytest.raises((RuntimeError, ValueError), match=r"(missing GEMINI_API_KEY|GEMINI_API_KEY is required|client not initialized)"):
        img = Image.new("RGB", (100, 100), color="red")
        service.analyze_image(img)


# =============================================================================
# 2. TEST SINGLE ITEM RECOGNITION (APPLE / BANANA / BREAD)
# =============================================================================

def test_gemini_vision_single_fresh_apple():
    """Verify single fresh apple is accurately parsed from Gemini structured JSON."""
    service = GeminiVisionService(api_key="test-key")
    mock_client = MagicMock()
    service.client = mock_client

    gemini_payload = {
        "is_food": True,
        "overall_status": "food",
        "items": [
            {
                "item_id": 1,
                "name": "apple",
                "category": "fruit",
                "confidence": 98.0,
                "condition": "fresh/ripe",
                "condition_confidence": 95.0,
                "freshness_status": "FRESH",
                "visible_signs": ["smooth taut red skin", "intact green stem"],
                "visual_description": "Crisp, whole red delicious apple with glossy skin.",
                "is_spoiled": False,
                "reason": "Uniform coloration and no soft spots or bruising.",
                "alternatives": [],
            }
        ],
        "alternatives": [],
        "overall_visual_summary": "Single whole fresh red apple.",
        "overall_freshness_summary": "Fresh and optimal for consumption.",
        "model_notes": "High clarity lighting.",
    }

    mock_response = MagicMock()
    mock_response.text = json.dumps(gemini_payload)
    mock_client.models.generate_content.return_value = mock_response

    img = Image.new("RGB", (256, 256), color="red")
    result = service.analyze_image(img)

    assert result["status"] == "recognized"
    assert result["source"] == "gemini"
    assert result["name"] == "apple"
    assert result["category"] == "fruit"
    assert result["confidence"] == 0.98
    assert result["is_food"] is True
    assert len(result["items"]) == 1
    assert result["items"][0]["is_spoiled"] is False
    assert result["items"][0]["freshness_status"] == "FRESH"


# =============================================================================
# 3. TEST SPOILED & MOLDY FOOD DETECTION
# =============================================================================

def test_gemini_vision_moldy_bread():
    """Verify moldy bread triggers is_spoiled=True and VISIBLE_SPOILAGE status."""
    service = GeminiVisionService(api_key="test-key")
    mock_client = MagicMock()
    service.client = mock_client

    gemini_payload = {
        "is_food": True,
        "overall_status": "food",
        "items": [
            {
                "item_id": 1,
                "name": "bread",
                "category": "bakery",
                "confidence": 96.0,
                "condition": "moldy",
                "condition_confidence": 94.0,
                "freshness_status": "VISIBLE_SPOILAGE",
                "visible_signs": ["green fuzzy mold colonies on crust", "white mycelium spreading"],
                "visual_description": "Sliced white sandwich bread showing extensive green fungal colonies.",
                "is_spoiled": True,
                "reason": "Visible fungal sporulation renders the bread unsafe.",
                "alternatives": [],
            }
        ],
        "alternatives": [],
        "overall_visual_summary": "Sliced white bread with visible green mold contamination.",
        "overall_freshness_summary": "Spoiled due to mold; discard immediately.",
        "model_notes": None,
    }

    mock_response = MagicMock()
    mock_response.text = json.dumps(gemini_payload)
    mock_client.models.generate_content.return_value = mock_response

    img = Image.new("RGB", (256, 256), color="white")
    result = service.analyze_image(img)

    assert result["status"] == "recognized"
    assert result["source"] == "gemini"
    assert result["name"] == "bread"
    assert len(result["items"]) == 1
    assert result["items"][0]["is_spoiled"] is True
    assert result["items"][0]["condition"] == "moldy"
    assert result["items"][0]["freshness_status"] == "VISIBLE_SPOILAGE"
    assert "mold" in result["items"][0]["visible_signs"][0].lower()


# =============================================================================
# 4. TEST MULTI-ITEM SCENE ENUMERATION (CRITICAL REGRESSION)
# =============================================================================

def test_gemini_vision_multi_item_enumeration_not_sandwich():
    """
    CRITICAL REGRESSION TEST:
    Verifies that a grouped scene with moldy bread, tomato, and onion is NOT collapsed
    into a sandwich, but enumerated as 3 distinct items with independent conditions.
    """
    service = GeminiVisionService(api_key="test-key")
    mock_client = MagicMock()
    service.client = mock_client

    gemini_payload = {
        "is_food": True,
        "overall_status": "food",
        "items": [
            {
                "item_id": 1,
                "name": "bread",
                "category": "bakery",
                "confidence": 95.0,
                "condition": "moldy",
                "condition_confidence": 92.0,
                "freshness_status": "VISIBLE_SPOILAGE",
                "visible_signs": ["green mold patches"],
                "visual_description": "Slice of white bread with green mold growth.",
                "is_spoiled": True,
                "reason": "Fungal growth present.",
                "alternatives": [],
            },
            {
                "item_id": 2,
                "name": "tomato",
                "category": "vegetable",
                "confidence": 93.0,
                "condition": "fresh/ripe",
                "condition_confidence": 89.0,
                "freshness_status": "FRESH",
                "visible_signs": ["smooth taut skin", "firm texture"],
                "visual_description": "Whole fresh ripe red tomato.",
                "is_spoiled": False,
                "reason": "Clean, unblemished skin.",
                "alternatives": [],
            },
            {
                "item_id": 3,
                "name": "onion",
                "category": "vegetable",
                "confidence": 91.0,
                "condition": "fresh",
                "condition_confidence": 87.0,
                "freshness_status": "FRESH",
                "visible_signs": ["dry papery outer layers"],
                "visual_description": "Whole fresh red onion with intact peel.",
                "is_spoiled": False,
                "reason": "Firm and dry.",
                "alternatives": [],
            },
        ],
        "alternatives": [],
        "overall_visual_summary": "Scene containing moldy bread, fresh tomato, and fresh onion.",
        "overall_freshness_summary": "Mixed freshness: bread is spoiled, vegetables are fresh.",
        "model_notes": "Distinct produce and bakery items placed side by side.",
    }

    mock_response = MagicMock()
    mock_response.text = json.dumps(gemini_payload)
    mock_client.models.generate_content.return_value = mock_response

    img = Image.new("RGB", (300, 300), color="white")
    result = service.analyze_image(img)

    assert result["status"] == "recognized"
    assert result["source"] == "gemini"
    assert result["name"] != "sandwich"
    assert len(result["items"]) == 3
    assert result["items"][0]["name"] == "bread"
    assert result["items"][0]["is_spoiled"] is True
    assert result["items"][1]["name"] == "tomato"
    assert result["items"][1]["is_spoiled"] is False
    assert result["items"][2]["name"] == "onion"
    assert result["items"][2]["is_spoiled"] is False


# =============================================================================
# 5. TEST NON-FOOD REJECTION
# =============================================================================

def test_gemini_vision_non_food_rejection():
    """Verify non-food images return status='not_food', is_food=False, and items=[]."""
    service = GeminiVisionService(api_key="test-key")
    mock_client = MagicMock()
    service.client = mock_client

    gemini_payload = {
        "is_food": False,
        "overall_status": "not_food",
        "items": [],
        "alternatives": ["smartphone", "electronic device"],
        "overall_visual_summary": "A modern touchscreen smartphone resting on a wooden desk.",
        "overall_freshness_summary": None,
        "model_notes": "Non-food electronic object.",
    }

    mock_response = MagicMock()
    mock_response.text = json.dumps(gemini_payload)
    mock_client.models.generate_content.return_value = mock_response

    img = Image.new("RGB", (200, 200), color="black")
    result = service.analyze_image(img)

    assert result["status"] == "not_food"
    assert result["is_food"] is False
    assert result["name"] is None
    assert len(result["items"]) == 0
    assert "smartphone" in result["overall_visual_summary"].lower()


# =============================================================================
# 6. TEST UNCERTAIN / OUT-OF-DISTRIBUTION FOOD
# =============================================================================

def test_gemini_vision_uncertain_food():
    """Verify unfamiliar food returns status='uncertain' with alternatives."""
    service = GeminiVisionService(api_key="test-key")
    mock_client = MagicMock()
    service.client = mock_client

    gemini_payload = {
        "is_food": True,
        "overall_status": "uncertain",
        "items": [
            {
                "item_id": 1,
                "name": "uncertain",
                "category": "fruit",
                "confidence": 55.0,
                "condition": "fresh",
                "condition_confidence": 70.0,
                "freshness_status": "CANNOT_DETERMINE",
                "visible_signs": ["red spiky rind"],
                "visual_description": "Exotic tropical fruit with hairy spines resembling rambutan or lychee.",
                "is_spoiled": False,
                "reason": "Rare botanical specimen.",
                "alternatives": ["rambutan", "lychee"],
            }
        ],
        "alternatives": ["rambutan", "lychee"],
        "overall_visual_summary": "An exotic spiky tropical fruit.",
        "overall_freshness_summary": "Appears intact but exact species is uncertain.",
        "model_notes": None,
    }

    mock_response = MagicMock()
    mock_response.text = json.dumps(gemini_payload)
    mock_client.models.generate_content.return_value = mock_response

    img = Image.new("RGB", (200, 200), color="red")
    result = service.analyze_image(img)

    assert result["status"] == "uncertain"
    assert result["name"] is None
    assert len(result["items"]) == 1
    assert "rambutan" in result["items"][0]["alternatives"]


# =============================================================================
# 7. TEST MALFORMED JSON RETRY HANDLING
# =============================================================================

def test_gemini_vision_malformed_json_retry_success():
    """Verify GeminiVisionService retries once when initial output is malformed JSON."""
    service = GeminiVisionService(api_key="test-key")
    mock_client = MagicMock()
    service.client = mock_client

    valid_payload = {
        "is_food": True,
        "overall_status": "food",
        "items": [
            {
                "item_id": 1,
                "name": "banana",
                "category": "fruit",
                "confidence": 95.0,
                "condition": "ripe",
                "condition_confidence": 92.0,
                "freshness_status": "FRESH",
                "visible_signs": ["yellow peel", "minor sugar spots"],
                "visual_description": "Ripe yellow banana.",
                "is_spoiled": False,
                "reason": "Optimal ripeness.",
                "alternatives": [],
            }
        ],
        "alternatives": [],
        "overall_visual_summary": "A ripe yellow banana.",
        "overall_freshness_summary": "Fresh and ready to eat.",
        "model_notes": None,
    }

    resp1 = MagicMock(text="```json\n{malformed json missing closing brace")
    resp2 = MagicMock(text=json.dumps(valid_payload))
    mock_client.models.generate_content.side_effect = [resp1, resp2]

    img = Image.new("RGB", (200, 200), color="yellow")
    result = service.analyze_image(img)

    assert result["status"] == "recognized"
    assert result["name"] == "banana"
    assert mock_client.models.generate_content.call_count == 2


# =============================================================================
# 8. TEST UNIFIED VISION SERVICE ROUTER & FALLBACK
# =============================================================================

def test_unified_vision_service_prioritizes_gemini():
    """Verify VisionService routes to Gemini when GEMINI_API_KEY is present."""
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-gemini-key", "VISION_PRIMARY_BACKEND": "gemini"}):
        with patch("services.vision_service.GeminiVisionService.analyze_image") as mock_gemini_analyze:
            mock_gemini_analyze.return_value = {
                "status": "recognized",
                "name": "mango",
                "category": "fruit",
                "confidence": 0.94,
                "visual_description": "Ripe golden mango.",
                "source": "gemini",
                "is_food": True,
                "items": [],
            }

            service = VisionService()
            img = Image.new("RGB", (224, 224), color="orange")
            res = service.analyze_image(img)

            assert res["source"] == "gemini"
            assert res["name"] == "mango"
            mock_gemini_analyze.assert_called_once()


def test_unified_vision_service_falls_back_to_siglip_on_gemini_error():
    """Verify VisionService smoothly falls back to local SigLIP when Gemini raises an exception."""
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-gemini-key", "VISION_PRIMARY_BACKEND": "gemini"}):
        with patch("services.vision_service.GeminiVisionService.analyze_image", side_effect=RuntimeError("Quota exceeded (429)")):
            service = VisionService()
            img = Image.new("RGB", (224, 224), color="green")
            res = service.analyze_image(img)

            # Assert fallback to SigLIP succeeded
            assert res["source"] == "local_siglip_fallback"
            assert "status" in res
            assert "items" in res


# =============================================================================
# 9. TEST LLM REASONING & CHAT WITH GEMINI
# =============================================================================

def test_llm_freshness_service_gemini_reasoning():
    """Verify LLMFreshnessService uses Gemini client for multimodal freshness reasoning."""
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-gemini-key"}):
        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            reasoning_payload = {
                "item_name": "apple",
                "detection_confidence": 0.98,
                "freshness_status": "FRESH",
                "reasoning_summary": "The apple shows bright red skin and ambient temperature is safe.",
                "visual_observations": ["Smooth taut red skin", "No bruising"],
                "environmental_context": "Safe ambient pantry conditions.",
                "estimated_remaining_freshness": {
                    "value": "5",
                    "unit": "days",
                    "range_description": "4 to 7 days under refrigeration",
                    "confidence": "high",
                    "basis": ["visual appearance", "safe ambient temp"],
                },
                "recommendations": ["Refrigerate to maximize crispness"],
                "uncertainty_factors": ["Internal quality cannot be evaluated without slicing"],
                "item_summaries": [
                    {
                        "item_name": "apple",
                        "category": "fruit",
                        "condition": "fresh/ripe",
                        "freshness_status": "FRESH",
                        "is_spoiled": False,
                        "shelf_life_estimate": "5 days",
                        "key_observation": "Crisp red skin",
                    }
                ],
            }

            mock_response = MagicMock()
            mock_response.text = json.dumps(reasoning_payload)
            mock_client.models.generate_content.return_value = mock_response

            llm_service = LLMFreshnessService(api_key="test-gemini-key", model="gemini-3.8-flash")

            packet = {
                "status": "recognized",
                "is_food": True,
                "visual_evidence": {
                    "status": "recognized",
                    "item_name": "apple",
                    "confidence": 0.98,
                    "visual_description": "Crisp red apple",
                },
                "sensor_evidence": {"temperature_c": 22.0, "humidity_percent": 55.0, "sensor_status": "valid"},
                "knowledge_guidelines": {"typical_shelf_life_days_room_temp": "5-7"},
                "items_evidence": [],
            }

            analysis = llm_service.analyze(packet)

            assert analysis.freshness_status == "FRESH"
            assert analysis.item_name == "apple"
            assert analysis.estimated_remaining_freshness.value == "5"
            assert len(analysis.item_summaries) == 1


def test_llm_freshness_service_gemini_chat():
    """Verify LLMFreshnessService chat_about_analysis calls Gemini when initialized."""
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-gemini-key"}):
        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            mock_response = MagicMock()
            mock_response.text = "You can keep this apple at room temperature for up to 5 days, or refrigerate it to extend its life."
            mock_client.models.generate_content.return_value = mock_response

            llm_service = LLMFreshnessService(api_key="test-gemini-key", model="gemini-3.8-flash")

            context = {
                "detected_food": "apple",
                "freshness_status": "FRESH",
                "estimated_shelf_life": "5 days",
            }

            reply = llm_service.chat_about_analysis(
                analysis_context=context,
                conversation_history=[],
                user_question="How long will this apple last?",
            )

            assert "apple" in reply.lower()
            assert "refrigerate" in reply.lower()


# =============================================================================
# 10. TEST QUOTA EXHAUSTION & 503 ERROR RECOVERY (ZERO HAMMERING)
# =============================================================================

def test_gemini_vision_429_quota_exhausted_zero_retries():
    """
    CRITICAL TEST: Verify 429 RESOURCE_EXHAUSTED immediately raises GeminiQuotaExhaustedError
    with zero retries (call_count == 1).
    """
    from services.vision_service import GeminiQuotaExhaustedError
    service = GeminiVisionService(api_key="test-key")
    mock_client = MagicMock()
    service.client = mock_client
    mock_client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED: quota metric generate_content_free_tier_requests exceeded. Please retry after 60s.")

    img = Image.new("RGB", (200, 200), color="green")
    with pytest.raises(GeminiQuotaExhaustedError) as exc_info:
        service.analyze_image(img)

    assert "429" in str(exc_info.value)
    # MUST only attempt once — no retrying on quota exhaustion
    assert mock_client.models.generate_content.call_count == 1


def test_gemini_vision_503_service_unavailable_retries_at_most_once():
    """
    Verify HTTP 503 Service Unavailable retries at most once (call_count == 2)
    before raising the exception for coordinator fallback.
    """
    service = GeminiVisionService(api_key="test-key")
    mock_client = MagicMock()
    service.client = mock_client
    mock_client.models.generate_content.side_effect = Exception("503 Service Unavailable")

    img = Image.new("RGB", (200, 200), color="green")
    with pytest.raises(Exception) as exc_info:
        service.analyze_image(img)

    assert "503" in str(exc_info.value)
    # Retries at most once
    assert mock_client.models.generate_content.call_count == 2


def test_vision_service_quota_exhausted_immediate_fallback_with_notice():
    """
    Verify VisionService catches GeminiQuotaExhaustedError, returns local_siglip_fallback,
    does NOT mark source as gemini, and sets clear fallback notice.
    """
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-gemini-key", "VISION_PRIMARY_BACKEND": "gemini"}):
        from services.vision_service import GeminiQuotaExhaustedError
        with patch("services.vision_service.GeminiVisionService.analyze_image") as mock_gemini:
            mock_gemini.side_effect = GeminiQuotaExhaustedError("429 RESOURCE_EXHAUSTED")

            service = VisionService()
            img = Image.new("RGB", (224, 224), color="orange")
            res = service.analyze_image(img)

            # Assert fallback metadata
            assert res["source"] == "local_siglip_fallback"
            assert res.get("quota_exhausted") is True
            assert res.get("fallback_reason") == "gemini_quota_exhausted"
            assert res.get("fallback_notice") == "Gemini quota temporarily exhausted — using Local Fallback Vision."
            assert mock_gemini.call_count == 1


def test_llm_chat_quota_exhausted_immediate_fallback_message():
    """
    Verify chat_about_analysis returns user-friendly quota notice on 429 without repeated retries.
    """
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-gemini-key"}):
        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED: Free tier quota exceeded.")

            llm_service = LLMFreshnessService(api_key="test-gemini-key", model="gemini-3.8-flash")
            reply = llm_service.chat_about_analysis(
                analysis_context={"detected_food": "banana"},
                conversation_history=[],
                user_question="Is this banana still fresh?",
            )

            assert "quota is temporarily exhausted" in reply.lower() or "unavailable" in reply.lower()
            assert mock_client.models.generate_content.call_count == 1

