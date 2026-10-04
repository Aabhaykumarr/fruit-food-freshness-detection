"""
Tests for high-accuracy hierarchical vision, multimodal image encoding, and knowledge rules.
"""

import numpy as np
from PIL import Image
from services.vision_service import VisionService
from services.detection_service import DetectionService
from services.llm_service import LLMFreshnessService
from services.evidence_aggregator import EvidenceAggregator
from services.history_service import HistoryService
from config import FOOD_CLASSES


def test_food_ontology_loaded():
    """Verify key food and fruit classes are present in config."""
    assert "apple" in FOOD_CLASSES
    assert "banana" in FOOD_CLASSES
    assert "mango" in FOOD_CLASSES
    assert "samosa" in FOOD_CLASSES
    assert "biryani" in FOOD_CLASSES
    assert "dosa" in FOOD_CLASSES
    assert "paneer" in FOOD_CLASSES
    assert "dal" in FOOD_CLASSES
    assert len(FOOD_CLASSES) >= 20


def test_vision_service_initialization():
    """Verify VisionService initializes properly."""
    service = VisionService()
    assert service is not None
    assert len(service.gate_prompts) > 0
    assert len(service.specialist_prompts) > 0


def test_image_encoding_support():
    """Verify multimodal service encodes various image formats to base64 JPEG."""
    service = LLMFreshnessService()

    # 1. Numpy array
    np_img = np.zeros((100, 100, 3), dtype=np.uint8)
    b64_np = service._encode_image(np_img)
    assert b64_np is not None
    assert isinstance(b64_np, str)

    # 2. PIL Image
    pil_img = Image.new("RGB", (50, 50), color="red")
    b64_pil = service._encode_image(pil_img)
    assert b64_pil is not None
    assert isinstance(b64_pil, str)

    # 3. None handling
    assert service._encode_image(None) is None


def test_expanded_knowledge_rules():
    """Verify knowledge lookup for diverse food items."""
    aggregator = EvidenceAggregator()
    for item in ["mango", "papaya", "guava", "watermelon", "grapes", "dal", "paneer", "roti", "apple", "banana"]:
        rule = aggregator.get_knowledge_for_item(item)
        assert rule is not None, f"Expected knowledge rule for {item}"
        assert "typical_shelf_life_days_room_temp" in rule


def test_hierarchical_vision_result_structure():
    """Verify analyze_image returns required dictionary keys."""
    service = VisionService()
    img = Image.new("RGB", (224, 224), color="white")
    result = service.analyze_image(img)

    assert "status" in result
    assert result["status"] in ["recognized", "uncertain", "not_food"]
    assert "category" in result
    assert "name" in result
    assert "confidence" in result
    assert "visual_description" in result
    assert "latency_ms" in result


def test_render_yaml_configuration():
    """Verify render.yaml structure, syntax, and Gemini environment configuration."""
    from pathlib import Path
    import re

    render_path = Path("render.yaml")
    assert render_path.exists(), "render.yaml must exist in repository root"
    content = render_path.read_text()

    assert "type: web" in content
    assert "env: python" in content
    assert "buildCommand:" in content
    assert "startCommand:" in content
    assert "streamlit run web_app.py" in content
    assert "GEMINI_API_KEY" in content
    assert "gemini-3.5-flash-lite" in content
    assert "OPENAI_API_KEY" not in content

