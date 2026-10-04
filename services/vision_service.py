"""
Multimodal Vision Pipeline & Hierarchical Vision Service for Food & Fruit Recognition.

Architecture:
1. PRIMARY: Multimodal Vision Model (OpenAI Vision API via OPENAI_VISION_MODEL, e.g. gpt-4o-mini).
   - Full scene inspection without downscaling or aggressive cropping.
   - Multi-item enumeration (up to 8 items) with independent identities and condition assessment.
   - Food-specific condition spectra (underripe, ripe, overripe, bruised, wilted, stale, moldy, rotten).
   - Clear distinction between food identity, ripeness, cosmetic damage, and actual spoilage.
   - Non-food negative rejection (electronics, faces, furniture, vehicles, household items).
   - Calibrated uncertainty for ambiguous or out-of-distribution items.
2. FALLBACK: SigLIP Zero-Shot Vision-Language Classifier (google/siglip-base-patch16-224).
   - 3-stage local pipeline: non-food gate -> ontology classification -> observable attributes.
   - Used when offline, API key missing, or on API failure, clearly labeled as 'local_siglip_fallback'.
"""

import os
import io
import time
import base64
import json
import logging
from typing import Dict, Any, Optional, List, Tuple, Union, Literal
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
import torch

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

try:
    from config import (
        CONFIDENCE_THRESHOLD,
        GATE_NON_FOOD_THRESHOLD,
        UNCERTAIN_PROB_THRESHOLD,
        UNCERTAIN_MARGIN_THRESHOLD,
        VISION_PRIMARY_BACKEND,
        GEMINI_API_KEY,
        GEMINI_VISION_MODEL,
        OPENAI_VISION_MODEL,
        OPENAI_API_KEY,
    )
except ImportError:
    CONFIDENCE_THRESHOLD = 0.40
    GATE_NON_FOOD_THRESHOLD = 0.55
    UNCERTAIN_PROB_THRESHOLD = 0.12
    UNCERTAIN_MARGIN_THRESHOLD = 0.04
    VISION_PRIMARY_BACKEND = "gemini"
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
    GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-3.8-flash")
    OPENAI_VISION_MODEL = "gpt-4o-mini"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

logger = logging.getLogger(__name__)


class GeminiQuotaExhaustedError(Exception):
    """Raised when Google Gemini returns 429 RESOURCE_EXHAUSTED or quota limit reached."""
    pass


# =============================================================================
# PYDANTIC STRUCTURED OUTPUT SCHEMAS FOR MULTIMODAL VISION
# =============================================================================

class ItemVisionResult(BaseModel):
    item_id: int = Field(description="Sequential 1-based index of the detected food item")
    name: str = Field(description="Normalized canonical food name (e.g. 'bread', 'apple', 'banana', 'tomato', 'onion', 'biryani', 'dal', 'pizza') or 'uncertain' if ambiguous")
    category: Literal["fruit", "vegetable", "bakery", "dish", "staple", "dairy", "meat_protein", "beverage", "other"] = Field(
        description="Broad culinary category"
    )
    confidence: float = Field(description="Detection & identification confidence percentage between 0.0 and 100.0")
    condition: str = Field(
        description="Food-specific visual condition: e.g. 'fresh', 'unripe', 'ripe', 'overripe', 'bruised', 'damaged', 'wilted', 'shrivelled', 'dried', 'stale', 'discolored', 'burnt', 'moldy', 'rotten', 'decomposing', 'leaking', 'questionable', 'uncertain', 'fresh-looking'"
    )
    condition_confidence: float = Field(description="Confidence percentage in condition assessment (0.0 to 100.0)")
    freshness_status: Literal["FRESH", "GOOD_TO_EAT_VISUALLY", "QUESTIONABLE", "POSSIBLE_SPOILAGE", "VISIBLE_SPOILAGE", "CANNOT_DETERMINE"] = Field(
        description="Calibrated high-level freshness status"
    )
    visible_signs: List[str] = Field(description="Specific visual signs observed (e.g. 'green mold colony on crust', 'wrinkling skin', 'dark brown bruising', 'crisp vibrant leaf')")
    visual_description: str = Field(description="Detailed physical description of this item's appearance, texture, and color")
    is_spoiled: bool = Field(description="True if visible decomposition, rot, extensive mold, or advanced spoilage is present")
    reason: str = Field(description="Reasoning explaining the condition and freshness status")
    alternatives: List[str] = Field(default_factory=list, description="Plausible alternative identities if uncertain")


class ImageVisionResult(BaseModel):
    is_food: bool = Field(description="True if the image contains edible food, fruit, vegetable, or prepared dish. False for non-food objects (electronics, humans, furniture, tools, etc.)")
    overall_status: Literal["food", "not_food", "uncertain"] = Field(
        description="'food' if recognizable food items are present; 'not_food' if the image contains no food; 'uncertain' if food-like but cannot be reliably determined"
    )
    items: List[ItemVisionResult] = Field(
        default_factory=list,
        description="List of distinct food items detected in the scene (up to 8 items). Empty if not_food."
    )
    alternatives: List[str] = Field(default_factory=list, description="Alternative plausible scene interpretations if ambiguous")
    overall_visual_summary: str = Field(description="Concise summary describing the entire scene, all detected items, and overall visual condition")
    overall_freshness_summary: Optional[str] = Field(default=None, description="High-level synthesis of freshness across all detected items")
    model_notes: Optional[str] = Field(default=None, description="Optional notes on lighting, occlusions, or image clarity")


# =============================================================================
# MULTIMODAL VISION SYSTEM PROMPT
# =============================================================================

MULTIMODAL_VISION_SYSTEM_PROMPT = """You are a world-class Food & Produce Computer Vision and Food Freshness Inspection System.

Your task is to analyze the entire presented image in full detail and return a structured JSON assessment.

CRITICAL INSPECTION RULES:
1. FULL SCENE MULTI-ITEM ENUMERATION:
   - Identify EVERY distinct edible food, fruit, vegetable, bakery item, staple, or prepared dish that is visibly present in the image.
   - Do NOT collapse unrelated objects into one dish.
   - A single slice of bread is bread, not automatically a sandwich.
   - Only identify a sandwich when the visual structure actually indicates a sandwich.
   - Maximum 8 distinct items.
   - If multiple food items are present (e.g. bread, onions, tomatoes, apples), enumerate EACH distinct item independently.
     * Example: Sliced bread with mold next to fresh tomatoes and onions:
       - Item 1: bread (category: bakery, condition: moldy, freshness_status: VISIBLE_SPOILAGE, is_spoiled: true)
       - Item 2: tomato (category: vegetable, condition: ripe, freshness_status: FRESH, is_spoiled: false)
       - Item 3: onion (category: vegetable, condition: fresh, freshness_status: FRESH, is_spoiled: false)

2. SINGLE-ITEM PRECISION:
   - If only one food item is present, identify it accurately (e.g. apple, banana, mango, tomato, bread, pizza, dosa, biryani, roti, paneer).

3. NON-FOOD REJECTION:
   - If the image contains non-food objects (smartphones, laptops, electronics, human faces/portraits, furniture, cars, tools, books, clothing, pets):
     * Set `is_food = false`
     * Set `overall_status = "not_food"`
     * Set `items = []`
     * Describe the non-food object in `overall_visual_summary`.

4. UNCERTAINTY & OUT-OF-DISTRIBUTION ITEMS:
   - If the image contains an ambiguous or unfamiliar dish/food:
     * Set `name = "uncertain"` and/or `overall_status = "uncertain"`
     * Provide plausible candidates in `alternatives`
     * Accurately describe the visible physical characteristics (e.g. grain base, sauce color, visible vegetables/proteins).

5. SEPARATE IDENTITY FROM CONDITION:
   - What the food IS (e.g. "bread", "banana") must be separated from its CONDITION (e.g. "moldy", "overripe").
   - Assess food-specific condition spectrum:
     fresh, unripe, ripe, overripe, bruised, damaged, wilted, shrivelled, dried, stale, discolored, burnt, moldy, rotten, decomposing, leaking, questionable, uncertain, fresh-looking.

6. FRESHNESS STATUS SPECTRUM (NOT BINARY):
   - Freshness is NOT simply binary fresh/spoiled. Cover the full spectrum:
     * FRESH: vibrant, prime condition, crisp, no defects.
     * GOOD_TO_EAT_VISUALLY: minor cosmetic variation or normal ripeness, completely safe to eat.
     * QUESTIONABLE: bruised, wilted, dried, or overripe with early deterioration.
     * POSSIBLE_SPOILAGE: strong signs of decay, off-color patches, severe softening.
     * VISIBLE_SPOILAGE: visible mold colonies, fungal growth, slime, rot, or decomposition (`is_spoiled: true`).
     * CANNOT_DETERMINE: image too blurry, occluded, or ambiguous.

7. DISTINGUISH RIPENESS & COSMETIC DAMAGE FROM SPOILAGE:
   - A banana with natural brown sugar spots is ripe/overripe, NOT spoiled unless moldy or collapsing.
   - Green banana -> unripe / underripe.
   - Yellow banana -> ripe.
   - Bread with normal appearance -> fresh-looking / fresh. Dry/hardened bread -> stale / dried. Bread with mold -> visible spoilage.
   - Leafy greens: fresh/crisp -> fresh; drooping -> wilted; yellowing -> questionable / deteriorating; mold -> visible spoilage.
   - A slightly bruised apple has mechanical damage (condition: "bruised", freshness_status: "QUESTIONABLE"), not necessarily rot.
   - Visible mold, slime, rot, or decomposition MUST be flagged as `is_spoiled: true` and `freshness_status: "VISIBLE_SPOILAGE"`.

8. VISIBLE FOOD SAFETY CAVEAT:
   - You evaluate VISIBLE freshness indicators. You do not claim invisible microbiological laboratory safety.
"""


# =============================================================================
# PRIMARY: GOOGLE GEMINI MULTIMODAL VISION SERVICE
# =============================================================================

class GeminiVisionService:
    """
    Primary vision service utilizing Google Gemini Multimodal Vision API (e.g., gemini-3.8-flash)
    with Pydantic structured output enforcement for multi-item food scene understanding.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, timeout: float = 30.0):
        self.api_key = api_key if api_key is not None else (GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")))
        self.model = model if model is not None else (GEMINI_VISION_MODEL or os.getenv("GEMINI_VISION_MODEL", "gemini-3.8-flash"))
        self.timeout = timeout
        self.client = None
        self._init_client()

    def _init_client(self):
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                logger.info("GeminiVisionService (Primary) initialized with model: %s", self.model)
            except Exception as e:
                logger.error("Failed to initialize Gemini client for GeminiVisionService: %s", e)
                self.client = None
        else:
            self.client = None

    def is_available(self) -> bool:
        return self.client is not None

    def _prepare_pil_image(self, image: Any) -> Optional[Image.Image]:
        """Convert various image formats (PIL, numpy, bytes, filepath, base64) to a PIL Image (RGB)."""
        if image is None:
            return None

        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, str):
            if image.startswith("data:image"):
                raw_b64 = image.split(",", 1)[1]
                img_bytes = base64.b64decode(raw_b64)
                return Image.open(io.BytesIO(img_bytes)).convert("RGB")
            if len(image) > 500 and not image.endswith((".jpg", ".png", ".jpeg")):
                img_bytes = base64.b64decode(image)
                return Image.open(io.BytesIO(img_bytes)).convert("RGB")
            if os.path.exists(image):
                return Image.open(image).convert("RGB")

        if isinstance(image, bytes):
            return Image.open(io.BytesIO(image)).convert("RGB")

        if isinstance(image, np.ndarray):
            try:
                import cv2
                if len(image.shape) == 3:
                    if image.shape[2] == 3:
                        # Assume BGR from OpenCV and convert to RGB
                        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        return Image.fromarray(rgb)
                    elif image.shape[2] == 4:
                        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                        return Image.fromarray(rgb)
                elif len(image.shape) == 2:
                    return Image.fromarray(image).convert("RGB")
            except Exception as e:
                logger.error("Failed to convert numpy array to PIL Image: %s", e)
                return None

        return None

    def analyze_image(self, image: Any) -> Dict[str, Any]:
        """
        Executes Google Gemini multimodal vision inspection on the image.
        Returns standardized vision result dictionary.
        """
        t0 = time.perf_counter()
        pil_img = self._prepare_pil_image(image)

        if pil_img is None:
            return {
                "status": "not_food",
                "category": None,
                "name": None,
                "confidence": 0.0,
                "visual_description": "No valid image provided.",
                "items": [],
                "overall_visual_summary": "No valid image provided.",
                "overall_freshness_summary": None,
                "model_notes": None,
                "source": "gemini",
                "latency_ms": 0.0,
                "raw_predictions": [],
                "is_food": False,
            }

        if self.client is None:
            raise RuntimeError("Gemini client not initialized (missing GEMINI_API_KEY).")

        from google.genai import types

        prompt_text = (
            "Analyze this image in detail. "
            "Identify EVERY distinct edible food, fruit, vegetable, bakery item, staple, or prepared dish that is visibly present in the image. "
            "Do not collapse unrelated objects into one dish. "
            "A single slice of bread is bread, not automatically a sandwich. "
            "Only identify a sandwich when the visual structure actually indicates a sandwich. "
            "Maximum 8 distinct items. "
            "Assess each item's condition, ripeness, visual freshness, visible spoilage signs, and confidence. "
            "If non-food objects or out-of-distribution foods are present, classify them strictly according to instructions."
        )

        config = types.GenerateContentConfig(
            system_instruction=MULTIMODAL_VISION_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=ImageVisionResult,
            temperature=0.2,
        )

        last_error = None
        # Try once, with one retry on parsing failure only.
        # Quota exhaustion (429 RESOURCE_EXHAUSTED) is never retried.
        # HTTP 503 is retried at most once with bounded backoff.
        for attempt in range(2):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[pil_img, prompt_text],
                    config=config,
                )

                latency = (time.perf_counter() - t0) * 1000

                # Parse and validate response
                result: Optional[ImageVisionResult] = None
                if hasattr(response, "parsed") and isinstance(response.parsed, ImageVisionResult):
                    result = response.parsed
                elif response.text:
                    result = ImageVisionResult.model_validate_json(response.text)
                else:
                    raise ValueError("Empty response received from Gemini API.")

                # Convert to standardized dictionary format
                items_list = [
                    item.model_dump() if hasattr(item, "model_dump") else item.dict()
                    for item in result.items
                ]

                # Determine top-level legacy fields for backward compatibility
                if not result.is_food or result.overall_status == "not_food" or len(result.items) == 0:
                    top_status = "not_food"
                    top_name = None
                    top_category = None
                    top_conf = 0.0
                    top_desc = result.overall_visual_summary or "Non-food object detected."
                elif result.overall_status == "uncertain" or (len(result.items) > 0 and result.items[0].name.lower() == "uncertain"):
                    top_status = "uncertain"
                    top_name = None
                    top_category = result.items[0].category if result.items else "other"
                    top_conf = round(result.items[0].confidence / 100.0, 3) if result.items else 0.5
                    top_desc = result.overall_visual_summary or (result.items[0].visual_description if result.items else "")
                else:
                    top_status = "recognized"
                    primary_item = result.items[0]
                    top_name = primary_item.name.lower().strip().replace(" ", "_")
                    top_category = primary_item.category
                    top_conf = round(primary_item.confidence / 100.0, 3)
                    top_desc = result.overall_visual_summary or primary_item.visual_description

                raw_preds = [
                    {
                        "class": it.name,
                        "category": it.category,
                        "confidence": round(it.confidence / 100.0, 3),
                        "condition": it.condition,
                        "freshness_status": it.freshness_status,
                    }
                    for it in result.items
                ]

                return {
                    "status": top_status,
                    "category": top_category,
                    "name": top_name,
                    "confidence": top_conf,
                    "visual_description": top_desc,
                    "items": items_list,
                    "overall_visual_summary": result.overall_visual_summary,
                    "overall_freshness_summary": result.overall_freshness_summary,
                    "alternatives": result.alternatives,
                    "model_notes": result.model_notes,
                    "source": "gemini",
                    "latency_ms": round(latency, 1),
                    "raw_predictions": raw_preds,
                    "is_food": result.is_food,
                }

            except Exception as e:
                last_error = e
                err_str = str(e).lower()

                # --- Quota exhaustion: 429 RESOURCE_EXHAUSTED ---
                # Do NOT retry. Raise immediately so the coordinator can fall back.
                if "429" in str(e) or "resource_exhausted" in err_str or "quota" in err_str:
                    logger.warning(
                        "Gemini quota exhausted (429 RESOURCE_EXHAUSTED). "
                        "No retry — falling back immediately."
                    )
                    raise GeminiQuotaExhaustedError(str(e)) from e

                # --- HTTP 503 Service Unavailable ---
                # Allow at most ONE retry with bounded backoff.
                if "503" in str(e) or "service unavailable" in err_str:
                    if attempt == 0:
                        logger.warning("Gemini 503 Service Unavailable — retrying once after 1s backoff.")
                        time.sleep(1.0)
                        continue
                    else:
                        logger.error("Gemini 503 persisted after 1 retry. Falling back.")
                        raise

                # Other errors (parse failures, etc.): retry once
                logger.warning("GeminiVisionService attempt %d failed: %s", attempt + 1, e)
                if attempt == 0:
                    time.sleep(0.5)

        logger.error("GeminiVisionService analysis failed after retries: %s", last_error)
        raise last_error


# =============================================================================
# PRIMARY: MULTIMODAL VISION SERVICE (OPENAI VISION API)
# =============================================================================

class MultimodalVisionService:
    """
    Primary vision service utilizing OpenAI Multimodal Vision models (e.g., gpt-4o-mini)
    with Pydantic structured output enforcement for multi-item food scene understanding.
    """

    def __init__(self, api_key: str = "", model: str = "gpt-4o-mini", timeout: float = 35.0):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model or os.getenv("OPENAI_VISION_MODEL", os.getenv("LLM_MODEL", "gpt-4o-mini"))
        self.timeout = timeout
        self.client = None
        self._init_client()

    def _init_client(self):
        if self.api_key and self.api_key.strip():
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key, timeout=self.timeout)
                logger.info("MultimodalVisionService (Optional Secondary OpenAI) initialized with model: %s", self.model)
            except Exception as e:
                logger.error("Failed to initialize OpenAI client for MultimodalVisionService: %s", e)
                self.client = None
        else:
            self.client = None

    def is_available(self) -> bool:
        return self.client is not None

    def _encode_image(self, image: Any) -> Optional[str]:
        """Convert various image formats (PIL, numpy, bytes, filepath) to base64 JPEG without heavy resizing."""
        if image is None:
            return None

        if isinstance(image, str):
            if image.startswith("data:image"):
                return image.split(",", 1)[1]
            if len(image) > 500 and not image.endswith((".jpg", ".png", ".jpeg")):
                return image
            if os.path.exists(image):
                try:
                    with open(image, "rb") as f:
                        return base64.b64encode(f.read()).decode("utf-8")
                except Exception as e:
                    logger.error("Failed to read image path %s: %s", image, e)
                    return None

        if isinstance(image, bytes):
            return base64.b64encode(image).decode("utf-8")

        if isinstance(image, Image.Image):
            try:
                buf = io.BytesIO()
                image.convert("RGB").save(buf, format="JPEG", quality=95)
                return base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception as e:
                logger.error("Failed to encode PIL Image: %s", e)
                return None

        if isinstance(image, np.ndarray):
            try:
                import cv2
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # Check if BGR or RGB
                    success, buffer = cv2.imencode(".jpg", image)
                    if success:
                        return base64.b64encode(buffer).decode("utf-8")
                elif len(image.shape) == 2:
                    success, buffer = cv2.imencode(".jpg", image)
                    if success:
                        return base64.b64encode(buffer).decode("utf-8")
            except Exception as e:
                logger.error("Failed to encode numpy array: %s", e)
                return None

        return None

    def analyze_image(self, image: Any) -> Dict[str, Any]:
        """
        Executes multimodal vision inspection on the image.
        Returns standardized vision result dictionary.
        """
        t0 = time.perf_counter()
        b64_img = self._encode_image(image)

        if not b64_img:
            return {
                "status": "not_food",
                "category": None,
                "name": None,
                "confidence": 0.0,
                "visual_description": "No valid image provided.",
                "items": [],
                "overall_visual_summary": "No valid image provided.",
                "model_notes": None,
                "source": "multimodal_openai",
                "latency_ms": 0.0,
                "raw_predictions": [],
                "is_food": False,
            }

        if self.client is None:
            raise RuntimeError("OpenAI client not initialized (missing API key).")

        try:
            prompt_content = [
                {
                    "type": "text",
                    "text": (
                        "Analyze this image in detail. Identify every distinct food, produce, or dish item present. "
                        "Assess each item's condition, ripeness, visual freshness, visible spoilage signs, and confidence. "
                        "If non-food objects or out-of-distribution foods are present, classify them strictly according to instructions."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_img}",
                        "detail": "high",
                    },
                },
            ]

            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": MULTIMODAL_VISION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_content},
                ],
                response_format=ImageVisionResult,
            )

            latency = (time.perf_counter() - t0) * 1000
            result: ImageVisionResult = completion.choices[0].message.parsed

            if result is None:
                raise ValueError("Received null parsed result from OpenAI Vision API.")

            # Convert to standardized dictionary format
            items_list = [
                item.model_dump() if hasattr(item, "model_dump") else item.dict()
                for item in result.items
            ]

            # Determine top-level legacy fields for backward compatibility
            if not result.is_food or result.overall_status == "not_food" or len(result.items) == 0:
                top_status = "not_food"
                top_name = None
                top_category = None
                top_conf = 0.0
                top_desc = result.overall_visual_summary or "Non-food object detected."
            elif result.overall_status == "uncertain" or (len(result.items) > 0 and result.items[0].name.lower() == "uncertain"):
                top_status = "uncertain"
                top_name = None
                top_category = result.items[0].category if result.items else "other"
                top_conf = round(result.items[0].confidence / 100.0, 3) if result.items else 0.5
                top_desc = result.overall_visual_summary or result.items[0].visual_description
            else:
                top_status = "recognized"
                primary_item = result.items[0]
                top_name = primary_item.name.lower().strip().replace(" ", "_")
                top_category = primary_item.category
                top_conf = round(primary_item.confidence / 100.0, 3)
                top_desc = result.overall_visual_summary or primary_item.visual_description

            raw_preds = [
                {
                    "class": it.name,
                    "category": it.category,
                    "confidence": round(it.confidence / 100.0, 3),
                    "condition": it.condition,
                    "freshness_status": it.freshness_status,
                }
                for it in result.items
            ]

            return {
                "status": top_status,
                "category": top_category,
                "name": top_name,
                "confidence": top_conf,
                "visual_description": top_desc,
                "items": items_list,
                "overall_visual_summary": result.overall_visual_summary,
                "alternatives": result.alternatives,
                "model_notes": result.model_notes,
                "source": "multimodal_openai",
                "latency_ms": round(latency, 1),
                "raw_predictions": raw_preds,
                "is_food": result.is_food,
            }

        except Exception as e:
            logger.error("MultimodalVisionService analysis failed: %s", e)
            raise e


# =============================================================================
# FALLBACK: SIGLIP VISION SERVICE (LOCAL OFFLINE VISION)
# =============================================================================

GATE_PROMPTS = [
    "a photo of fresh raw edible fruit, whole raw vegetable, or farm produce",
    "a photo of a cooked meal, prepared food dish, curry, rice, flatbread, soup, or plated food",
    "a photo of an electronic device, computer, laptop, smartphone, circuit board, charger, cable, mouse, headphones, or screen",
    "a photo of a human person, portrait, face, hand, or human body",
    "a photo of indoor furniture, chair, desk, table, couch, room interior, or building architecture",
    "a photo of a car, automobile, sports car, motor vehicle, road, traffic, or outdoor street",
    "a photo of non-food household objects, metal keys, wristwatch, pen, shoe, clothing, paper book, bottle, container, or pet animal",
]

SYNONYM_MAP = {
    "chapati": "roti",
    "phulka": "roti",
    "rotli": "roti",
    "roti": "roti",
    "paratha": "paratha",
    "aloo_paratha": "paratha",
    "stuffed_paratha": "paratha",
    "naan": "naan",
    "garlic_naan": "naan",
    "butter_naan": "naan",
    "tandoori_roti": "naan",
    "kulcha": "naan",
    "plain_rice": "rice",
    "steamed_rice": "rice",
    "boiled_rice": "rice",
    "white_rice": "rice",
    "chawal": "rice",
    "rice": "rice",
    "fried_rice": "fried_rice",
    "pulao": "fried_rice",
    "jeera_rice": "fried_rice",
    "veg_pulao": "fried_rice",
    "dal": "dal",
    "yellow_dal": "dal",
    "dal_tadka": "dal",
    "dal_fry": "dal",
    "toor_dal": "dal",
    "moong_dal": "dal",
    "rajma": "rajma",
    "kidney_beans": "rajma",
    "rajma_curry": "rajma",
    "chole": "chole",
    "chana": "chole",
    "chana_masala": "chole",
    "chickpea_curry": "chole",
    "dal_makhani": "dal_makhani",
    "sambar": "sambar",
    "rajma_chawal": "rajma_chawal",
    "rice_dal_combo": "rice_dal_combo",
    "dal_chawal": "rice_dal_combo",
    "thali_mixed": "thali_mixed",
    "indian_thali": "thali_mixed",
    "thali": "thali_mixed",
    "chole_bhature": "chole_bhature",
    "pav_bhaji": "pav_bhaji",
    "khichdi": "khichdi",
    "dal_khichdi": "khichdi",
    "poha": "poha",
    "upma": "upma",
    "biryani": "biryani",
    "chicken_biryani": "biryani",
    "mutton_biryani": "biryani",
    "dum_biryani": "biryani",
    "paneer": "paneer",
    "paneer_butter_masala": "paneer",
    "shahi_paneer": "paneer",
    "matar_paneer": "paneer",
    "kadai_paneer": "paneer",
    "palak_paneer": "palak_paneer",
    "saag_paneer": "palak_paneer",
    "aloo_gobi": "aloo_gobi",
    "aloo_gobhi": "aloo_gobi",
    "samosa": "samosa",
    "dosa": "dosa",
    "masala_dosa": "dosa",
    "plain_dosa": "dosa",
    "idli": "idli",
    "vada": "vada",
    "medu_vada": "vada",
    "pizza": "pizza",
    "burger": "burger",
    "hamburger": "burger",
    "pasta": "pasta",
    "spaghetti": "pasta",
    "sandwich": "sandwich",
    "soup": "soup",
    "salad": "salad",
    "apple": "apple",
    "banana": "banana",
    "orange": "orange",
    "mango": "mango",
    "strawberry": "strawberry",
    "watermelon": "watermelon",
    "grapes": "grapes",
    "pomegranate": "pomegranate",
    "pineapple": "pineapple",
    "lemon": "lemon",
    "lime": "lemon",
    "papaya": "papaya",
    "guava": "guava",
    "tomato": "tomato",
    "potato": "potato",
    "aloo": "potato",
    "batata": "potato",
    "onion": "onion",
    "pyaz": "onion",
    "carrot": "carrot",
    "gajar": "carrot",
    "cucumber": "cucumber",
    "kheera": "cucumber",
    "spinach": "spinach",
    "palak": "spinach",
    "broccoli": "broccoli",
    "mushroom": "mushroom",
    "okra": "okra",
    "bhindi": "okra",
    "ladyfinger": "okra",
    "eggplant": "eggplant",
    "baingan": "eggplant",
    "brinjal": "eggplant",
    "aubergine": "eggplant",
    "cauliflower": "cauliflower",
    "gobi": "cauliflower",
    "gobhi": "cauliflower",
    "bell_pepper": "bell_pepper",
    "capsicum": "bell_pepper",
    "ginger": "ginger",
    "adrak": "ginger",
    "garlic": "garlic",
    "lahsun": "garlic",
    "green_chili": "green_chili",
    "hari_mirch": "green_chili",
}

ONTOLOGY_SPECS = [
    {"prompt": "fresh whole red apple fruit with stem", "raw_class": "apple", "category": "fruit", "desc": "Fresh whole apple with crisp skin"},
    {"prompt": "bunch of ripe yellow bananas with peel", "raw_class": "banana", "category": "fruit", "desc": "Ripe banana with yellow peel"},
    {"prompt": "fresh round citrus orange fruit with textured rind", "raw_class": "orange", "category": "fruit", "desc": "Citrus orange fruit with textured rind"},
    {"prompt": "ripe golden yellow sweet mango fruit", "raw_class": "mango", "category": "fruit", "desc": "Sweet ripe mango with smooth skin"},
    {"prompt": "fresh ripe red strawberries with seeds", "raw_class": "strawberry", "category": "fruit", "desc": "Fresh strawberries with red pulp and seeds"},
    {"prompt": "sliced fresh watermelon with red flesh and dark rind", "raw_class": "watermelon", "category": "fruit", "desc": "Juicy watermelon slice with red flesh and dark rind"},
    {"prompt": "cluster of fresh green or purple grapes", "raw_class": "grapes", "category": "fruit", "desc": "Cluster of fresh juicy grapes"},
    {"prompt": "fresh red pomegranate fruit with seeds", "raw_class": "pomegranate", "category": "fruit", "desc": "Pomegranate fruit with crimson seed arils"},
    {"prompt": "tropical pineapple fruit with spiky rind and green crown", "raw_class": "pineapple", "category": "fruit", "desc": "Tropical pineapple with spiky textured rind"},
    {"prompt": "fresh bright yellow lemon or green lime", "raw_class": "lemon", "category": "fruit", "desc": "Yellow citrus lemon"},
    {"prompt": "tropical papaya fruit with orange flesh", "raw_class": "papaya", "category": "fruit", "desc": "Tropical papaya with orange flesh"},
    {"prompt": "fresh green round guava fruit", "raw_class": "guava", "category": "fruit", "desc": "Green guava fruit"},
    {"prompt": "fresh ripe red tomato with smooth skin", "raw_class": "tomato", "category": "vegetable", "desc": "Fresh ripe red tomato"},
    {"prompt": "raw whole brown potato tuber with skin", "raw_class": "potato", "category": "vegetable", "desc": "Raw whole potato tuber"},
    {"prompt": "fresh raw whole onion bulb with papery skin", "raw_class": "onion", "category": "vegetable", "desc": "Fresh whole onion bulb"},
    {"prompt": "fresh orange carrot taproot with pointed tip", "raw_class": "carrot", "category": "vegetable", "desc": "Crunchy orange carrot taproot"},
    {"prompt": "fresh green slicing cucumber with dark green skin", "raw_class": "cucumber", "category": "vegetable", "desc": "Crisp green slicing cucumber"},
    {"prompt": "fresh raw leafy green spinach bunch", "raw_class": "spinach", "category": "vegetable", "desc": "Leafy green fresh spinach"},
    {"prompt": "fresh green broccoli crown with compact florets", "raw_class": "broccoli", "category": "vegetable", "desc": "Green broccoli crown florets"},
    {"prompt": "fresh edible white button or cremini mushrooms", "raw_class": "mushroom", "category": "vegetable", "desc": "Fresh edible culinary mushrooms"},
    {"prompt": "fresh green pointed okra or bhindi pods", "raw_class": "okra", "category": "vegetable", "desc": "Fresh green okra / bhindi pods"},
    {"prompt": "glossy purple eggplant or brinjal", "raw_class": "eggplant", "category": "vegetable", "desc": "Glossy purple eggplant / baingan"},
    {"prompt": "creamy white cauliflower gobi head", "raw_class": "cauliflower", "category": "vegetable", "desc": "Creamy white cauliflower / gobi head"},
    {"prompt": "fresh crisp green or red bell pepper capsicum", "raw_class": "bell_pepper", "category": "vegetable", "desc": "Crisp bell pepper / capsicum"},
    {"prompt": "fresh knobby raw ginger root rhizome", "raw_class": "ginger", "category": "vegetable", "desc": "Aromatic ginger root rhizome"},
    {"prompt": "fresh whole garlic bulb with cloves", "raw_class": "garlic", "category": "vegetable", "desc": "Pungent garlic bulb and cloves"},
    {"prompt": "fresh slender green chili peppers", "raw_class": "green_chili", "category": "vegetable", "desc": "Pungent green chili peppers"},
    {"prompt": "plain steamed white rice, cooked basmati rice, or boiled white chawal in a bowl or on a plate", "raw_class": "plain_rice", "category": "staple", "desc": "Steamed white plain basmati rice grains"},
    {"prompt": "vegetable fried rice, chinese wok fried rice, seasoned pulao, or jeera rice with mixed vegetables", "raw_class": "fried_rice", "category": "dish", "desc": "Seasoned fried rice / pulao with visible grains and vegetables"},
    {"prompt": "whole wheat flatbread, chapati, phulka, or roti with light brown spots", "raw_class": "roti", "category": "staple", "desc": "Whole-wheat unleavened Indian roti / chapati flatbread"},
    {"prompt": "pan-fried shallow golden layered indian paratha flatbread or aloo paratha", "raw_class": "paratha", "category": "staple", "desc": "Golden shallow-fried layered Indian paratha flatbread"},
    {"prompt": "tandoori garlic butter naan bread or charred tandoor flatbread with blisters", "raw_class": "naan", "category": "staple", "desc": "Tandoori baked naan flatbread with blistered crust"},
    {"prompt": "a bowl of yellow dal tadka, moong dal, yellow lentil soup, or dal fry with tempering and herbs", "raw_class": "dal", "category": "dish", "desc": "Tempered yellow lentil dal soup with spices"},
    {"prompt": "red kidney bean curry (rajma masala) in dark spiced gravy in a bowl", "raw_class": "rajma", "category": "dish", "desc": "Red kidney bean (rajma) curry in thick spiced tomato-onion gravy"},
    {"prompt": "spiced chickpea curry (chole chana masala) with garbanzo beans in a bowl", "raw_class": "chole", "category": "dish", "desc": "Spiced chickpea (chole / chana masala) curry"},
    {"prompt": "creamy dark black lentil dal makhani with butter and cream", "raw_class": "dal_makhani", "category": "dish", "desc": "Creamy slow-cooked black lentil dal makhani"},
    {"prompt": "south indian tangy vegetable and lentil sambar stew with drumsticks in a bowl", "raw_class": "sambar", "category": "dish", "desc": "South Indian tangy vegetable and lentil sambar stew"},
    {"prompt": "a meal of white rice and red kidney bean rajma curry served together on a plate", "raw_class": "rajma_chawal", "category": "dish", "desc": "Rajma Chawal combination plate of steamed rice with red kidney bean curry"},
    {"prompt": "a meal plate of white rice with yellow dal lentil curry or dal chawal with yellow lentil gravy and papad", "raw_class": "rice_dal_combo", "category": "dish", "desc": "Rice and Dal combination plate with yellow lentil curry over white rice"},
    {"prompt": "traditional indian thali platter with multiple small metal katoris or bowls containing several curries, dal, roti and rice", "raw_class": "thali_mixed", "category": "dish", "desc": "Traditional Indian thali platter with multiple regional dishes, curries, dal, roti and rice"},
    {"prompt": "spicy chickpea chole curry served with large puffy deep-fried bhatura bread", "raw_class": "chole_bhature", "category": "dish", "desc": "Chole Bhature plate with fried puffy bhature bread and spicy chickpea curry"},
    {"prompt": "spiced mashed vegetable curry (bhaji) served with buttered toasted soft bread buns (pav)", "raw_class": "pav_bhaji", "category": "dish", "desc": "Pav Bhaji spiced mashed vegetable curry served with toasted buttered pav bread"},
    {"prompt": "a bowl of soft savory rice and yellow lentil khichdi porridge", "raw_class": "khichdi", "category": "dish", "desc": "Comforting soft savory rice and yellow lentil khichdi"},
    {"prompt": "seasoned yellow flattened rice flakes (poha) cooked with turmeric, mustard seeds, and peanuts", "raw_class": "poha", "category": "dish", "desc": "Spiced yellow flattened rice flakes (poha) with turmeric, mustard seeds, and peanuts"},
    {"prompt": "savory south indian rava upma semolina porridge with mustard seeds and curry leaves in a bowl", "raw_class": "upma", "category": "dish", "desc": "Savory roasted semolina porridge (upma) seasoned with curry leaves and mustard"},
    {"prompt": "unsupported foreign dish, chinese dim sum dumplings, spanish seafood saffron paella with rice and prawns, tropical rambutan fruit, or uncommon international recipe", "raw_class": "unknown_food", "category": "dish", "desc": "Unidentified or unsupported foreign dish"},
    {"prompt": "spiced layered chicken or mutton biryani with seasoned basmati rice and aromatic spices", "raw_class": "biryani", "category": "dish", "desc": "Fragrant seasoned basmati rice biryani cooked with rich spices and aromatics"},
    {"prompt": "cubed paneer cottage cheese pieces in rich creamy orange tomato gravy (paneer butter masala)", "raw_class": "paneer", "category": "dish", "desc": "Cubed paneer cottage cheese in rich creamy tomato-butter gravy"},
    {"prompt": "cubed paneer cottage cheese pieces in vibrant green pureed spinach gravy (palak paneer or saag paneer)", "raw_class": "palak_paneer", "category": "dish", "desc": "Paneer cottage cheese cubes in vibrant spiced green spinach puree (palak paneer)"},
    {"prompt": "dry spiced curry of potato cubes and cauliflower florets (aloo gobi)", "raw_class": "aloo_gobi", "category": "dish", "desc": "Spiced stir-fried potato cubes and cauliflower florets (aloo gobi)"},
    {"prompt": "golden fried triangular pastry samosa with spiced potato filling", "raw_class": "samosa", "category": "dish", "desc": "Golden fried triangular pastry filled with spiced potato and pea stuffing"},
    {"prompt": "crispy thin golden brown rolled south indian dosa crepe served with chutney and sambar", "raw_class": "dosa", "category": "dish", "desc": "Golden fermented rice-and-lentil crepe (dosa)"},
    {"prompt": "steamed round fluffy white savory rice cakes (idli)", "raw_class": "idli", "category": "dish", "desc": "Soft steamed savory fermented white rice cakes (idli)"},
    {"prompt": "crispy golden fried savory lentil donut (medu vada)", "raw_class": "vada", "category": "dish", "desc": "Crispy golden fried savory lentil donut (medu vada)"},
    {"prompt": "baked round pizza crust topped with melted cheese, tomato sauce, and toppings", "raw_class": "pizza", "category": "dish", "desc": "Baked crust pizza with melted cheese and savory toppings"},
    {"prompt": "hamburger or cheeseburger with bun, cooked patty, lettuce, and cheese", "raw_class": "burger", "category": "dish", "desc": "Burger patty in sandwich bun with fresh toppings and condiments"},
    {"prompt": "cooked italian pasta noodles with sauce (spaghetti, penne, or macaroni)", "raw_class": "pasta", "category": "dish", "desc": "Cooked pasta noodles with seasoned sauce"},
    {"prompt": "toasted sandwich with sliced bread and fillings", "raw_class": "sandwich", "category": "bakery", "desc": "Layered sandwich with toasted sliced bread and fillings"},
    {"prompt": "liquid soup broth in a bowl with vegetables or meat", "raw_class": "soup", "category": "dish", "desc": "Savory liquid soup broth with cooked ingredients"},
    {"prompt": "fresh bowl of green leafy salad with raw vegetables", "raw_class": "salad", "category": "dish", "desc": "Fresh mixed salad greens and raw vegetable slices"},
]

ATTRIBUTE_PROMPTS_GRAIN = [
    "steamed white rice grains",
    "spiced basmati rice with mixed seasonings",
    "flatbread, roti, naan, or bread roll",
    "pasta or noodles",
    "flattened rice flakes or semolina grains",
    "no visible rice, grain, or bread base",
]

ATTRIBUTE_PROMPTS_GRAVY = [
    "yellow tempered lentil dal gravy",
    "red-orange creamy tomato-onion spiced curry gravy",
    "dark brown spicy masala gravy",
    "vibrant green pureed spinach saag gravy",
    "dry stir-fried preparation with no gravy",
    "clear or blended liquid soup broth",
]

ATTRIBUTE_PROMPTS_LEGUME_VEG = [
    "whole red kidney beans or chickpeas",
    "cubed paneer cottage cheese pieces",
    "steamed or stir-fried mixed vegetables and potatoes",
    "golden fried pastry or deep-fried dough",
    "steamed white fermented cakes or crepe",
    "fresh sliced fruits or raw salad",
]

ATTRIBUTE_PROMPTS_PLATING = [
    "a single ceramic bowl or plate containing one main food item",
    "a combination plate with rice and gravy poured together",
    "a multi-compartment metal thali platter with multiple separate small bowls",
    "a platter of bread served with dipping bowls",
]


class SigLIPVisionService:
    """
    Local offline SigLIP vision-language classifier.
    Preserved as a reliable local offline fallback.
    """

    def __init__(self, model_name: str = "google/siglip-base-patch16-224"):
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gate_prompts = GATE_PROMPTS
        self.specialist_prompts = [item["prompt"] for item in ONTOLOGY_SPECS]
        self.ontology_map = {item["prompt"]: item for item in ONTOLOGY_SPECS}
        self.attr_grain_prompts = ATTRIBUTE_PROMPTS_GRAIN
        self.attr_gravy_prompts = ATTRIBUTE_PROMPTS_GRAVY
        self.attr_legume_prompts = ATTRIBUTE_PROMPTS_LEGUME_VEG
        self.attr_plating_prompts = ATTRIBUTE_PROMPTS_PLATING
        self._load_model()

    def _load_model(self):
        try:
            from transformers import AutoProcessor, AutoModel
            logger.info("Initializing SigLIP fallback model: %s on %s...", self.model_name, self.device)
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

            self.gate_text_embeds = self._precompute_text_features(self.gate_prompts)
            self.spec_text_embeds = self._precompute_text_features(self.specialist_prompts)
            self.attr_grain_embeds = self._precompute_text_features(self.attr_grain_prompts)
            self.attr_gravy_embeds = self._precompute_text_features(self.attr_gravy_prompts)
            self.attr_legume_embeds = self._precompute_text_features(self.attr_legume_prompts)
            self.attr_plating_embeds = self._precompute_text_features(self.attr_plating_prompts)
            self.logit_scale = self.model.logit_scale.exp()
            self.logit_bias = self.model.logit_bias
            logger.info("SigLIP fallback model loaded successfully.")
        except Exception as e:
            logger.warning("SigLIP model loading deferred or failed: %s", e)
            self.model = None
            self.processor = None

    def _precompute_text_features(self, prompts: List[str]) -> torch.Tensor:
        inputs = self.processor(text=prompts, padding="max_length", return_tensors="pt").to(self.device)
        with torch.no_grad():
            tf = self.model.get_text_features(**inputs).pooler_output
            return tf / tf.norm(dim=-1, keepdim=True)

    def _prepare_pil_image(self, image: Any) -> Optional[Image.Image]:
        if image is None:
            return None
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, str) and os.path.exists(image):
            try:
                return Image.open(image).convert("RGB")
            except Exception:
                return None
        if isinstance(image, bytes):
            try:
                return Image.open(io.BytesIO(image)).convert("RGB")
            except Exception:
                return None
        if isinstance(image, np.ndarray):
            try:
                if len(image.shape) == 2:
                    return Image.fromarray(image).convert("RGB")
                if len(image.shape) == 3 and image.shape[2] == 3:
                    import cv2
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    return Image.fromarray(rgb)
            except Exception:
                return None
        return None

    def _extract_observable_attributes_from_embed(self, img_embed: torch.Tensor) -> str:
        try:
            with torch.no_grad():
                g_logits = (img_embed @ self.attr_grain_embeds.T) * self.logit_scale + self.logit_bias
                p_grain = torch.softmax(g_logits[0], dim=-1)
                best_grain_idx = int(torch.argmax(p_grain).item())
                best_grain = self.attr_grain_prompts[best_grain_idx]

                gv_logits = (img_embed @ self.attr_gravy_embeds.T) * self.logit_scale + self.logit_bias
                p_gravy = torch.softmax(gv_logits[0], dim=-1)
                best_gravy_idx = int(torch.argmax(p_gravy).item())
                best_gravy = self.attr_gravy_prompts[best_gravy_idx]

                l_logits = (img_embed @ self.attr_legume_embeds.T) * self.logit_scale + self.logit_bias
                p_legume = torch.softmax(l_logits[0], dim=-1)
                best_legume_idx = int(torch.argmax(p_legume).item())
                best_legume = self.attr_legume_prompts[best_legume_idx]

                pl_logits = (img_embed @ self.attr_plating_embeds.T) * self.logit_scale + self.logit_bias
                p_plating = torch.softmax(pl_logits[0], dim=-1)
                best_plating_idx = int(torch.argmax(p_plating).item())

            components = []
            if best_plating_idx == 2:
                components.append("Composite multi-compartment meal (thali) with multiple bowls")
            elif best_plating_idx == 1:
                components.append("Combination plate with grain base and gravy together")

            n_grain = len(self.attr_grain_prompts)
            if best_grain_idx != (n_grain - 1):
                components.append(f"Base: {best_grain}")
            if best_gravy_idx != 4:
                components.append(f"Sauce/gravy: {best_gravy}")
            if best_legume_idx < 3:
                components.append(f"Observed elements: {best_legume}")

            if not components:
                return "Food preparation with mixed culinary ingredients."
            return ". ".join(components) + "."
        except Exception:
            return "Food item with unconfirmed specific recipe."

    def analyze_image(self, image: Any) -> Dict[str, Any]:
        t0 = time.perf_counter()
        pil_img = self._prepare_pil_image(image)

        if pil_img is None or self.model is None or self.processor is None:
            return {
                "status": "not_food",
                "category": None,
                "name": None,
                "confidence": 0.0,
                "visual_description": "No valid image provided or local vision model unavailable.",
                "items": [],
                "overall_visual_summary": "No valid image provided.",
                "source": "local_siglip_fallback",
                "latency_ms": 0.0,
                "raw_predictions": [],
                "is_food": False,
            }

        try:
            img_inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                img_f = self.model.get_image_features(**img_inputs).pooler_output
                img_f = img_f / img_f.norm(dim=-1, keepdim=True)

            # Stage 1: Gate
            gate_logits = (img_f @ self.gate_text_embeds.T) * self.logit_scale + self.logit_bias
            gate_probs = torch.softmax(gate_logits[0], dim=-1)
            non_food_score = float(torch.sum(gate_probs[2:]).item())
            top_gate_idx = int(torch.argmax(gate_probs).item())
            max_non_food_prob = float(torch.max(gate_probs[2:]).item()) if len(gate_probs) > 2 else 0.0

            is_non_food = (top_gate_idx >= 2 or non_food_score > 0.50 or max_non_food_prob > 0.35)

            if is_non_food:
                latency = (time.perf_counter() - t0) * 1000
                gate_reason = self.gate_prompts[top_gate_idx]
                desc = f"Not a fruit or food. Object visually detected as: {gate_reason.replace('a photo of ', '')}."
                return {
                    "status": "not_food",
                    "category": None,
                    "name": None,
                    "confidence": round(non_food_score, 3),
                    "visual_description": desc,
                    "items": [],
                    "overall_visual_summary": desc,
                    "source": "local_siglip_fallback",
                    "latency_ms": round(latency, 1),
                    "raw_predictions": [],
                    "is_food": False,
                }

            # Stage 2: Specialist
            spec_logits = (img_f @ self.spec_text_embeds.T) * self.logit_scale + self.logit_bias
            spec_probs = torch.softmax(spec_logits[0], dim=-1)
            sorted_probs, sorted_indices = torch.sort(spec_probs, descending=True)
            top_prob = float(sorted_probs[0].item())
            second_prob = float(sorted_probs[1].item()) if len(sorted_probs) > 1 else 0.0

            top_prompt = self.specialist_prompts[sorted_indices[0].item()]
            top_spec = self.ontology_map[top_prompt]
            raw_class = top_spec["raw_class"]
            canonical_class = SYNONYM_MAP.get(raw_class, raw_class)
            category = top_spec["category"]
            base_desc = top_spec["desc"]

            raw_top3 = []
            for i in range(min(3, len(sorted_indices))):
                p_idx = sorted_indices[i].item()
                raw_item = self.ontology_map[self.specialist_prompts[p_idx]]
                c_name = SYNONYM_MAP.get(raw_item["raw_class"], raw_item["raw_class"])
                raw_top3.append({
                    "class": c_name,
                    "category": raw_item["category"],
                    "confidence": round(float(sorted_probs[i].item()), 3),
                    "condition": "fresh/ripe",
                    "freshness_status": "FRESH",
                })

            margin = top_prob - second_prob
            latency = (time.perf_counter() - t0) * 1000
            is_uncertain = (raw_class == "unknown_food" or top_prob < 0.28 or (top_prob < 0.40 and margin < 0.06))

            if is_uncertain:
                observable_desc = self._extract_observable_attributes_from_embed(img_f)
                full_desc = f"The exact dish could not be identified with certainty. Visually observed: {observable_desc}"
                single_item = {
                    "item_id": 1,
                    "name": "uncertain",
                    "category": category if category in ["fruit", "vegetable", "bakery", "dish", "staple", "dairy", "meat_protein", "beverage", "other"] else "other",
                    "confidence": round(top_prob * 100, 1),
                    "condition": "fresh-looking",
                    "condition_confidence": 70.0,
                    "freshness_status": "CANNOT_DETERMINE",
                    "visible_signs": ["Uncertain food structure"],
                    "visual_description": full_desc,
                    "is_spoiled": False,
                    "reason": "Local zero-shot fallback classified item with high uncertainty.",
                    "alternatives": [it["class"] for it in raw_top3],
                }
                return {
                    "status": "uncertain",
                    "category": category,
                    "name": None,
                    "confidence": round(top_prob, 3),
                    "visual_description": full_desc,
                    "items": [single_item],
                    "overall_visual_summary": full_desc,
                    "alternatives": [it["class"] for it in raw_top3],
                    "source": "local_siglip_fallback",
                    "latency_ms": round(latency, 1),
                    "raw_predictions": raw_top3,
                    "is_food": True,
                }
            else:
                single_item = {
                    "item_id": 1,
                    "name": canonical_class,
                    "category": category if category in ["fruit", "vegetable", "bakery", "dish", "staple", "dairy", "meat_protein", "beverage", "other"] else "dish",
                    "confidence": round(top_prob * 100, 1),
                    "condition": "fresh/ripe",
                    "condition_confidence": 85.0,
                    "freshness_status": "FRESH",
                    "visible_signs": ["Typical color and shape for " + canonical_class],
                    "visual_description": base_desc,
                    "is_spoiled": False,
                    "reason": f"Local zero-shot fallback identified {canonical_class}.",
                    "alternatives": [],
                }
                return {
                    "status": "recognized",
                    "category": category,
                    "name": canonical_class,
                    "confidence": round(top_prob, 3),
                    "visual_description": base_desc,
                    "items": [single_item],
                    "overall_visual_summary": base_desc,
                    "alternatives": [],
                    "source": "local_siglip_fallback",
                    "latency_ms": round(latency, 1),
                    "raw_predictions": raw_top3,
                    "is_food": True,
                }

        except Exception as e:
            logger.error("SigLIP analysis error: %s", e)
            latency = (time.perf_counter() - t0) * 1000
            return {
                "status": "uncertain",
                "category": "dish",
                "name": None,
                "confidence": 0.0,
                "visual_description": f"Visual processing error: {e}",
                "items": [],
                "overall_visual_summary": f"Visual processing error: {e}",
                "source": "local_siglip_fallback",
                "latency_ms": round(latency, 1),
                "raw_predictions": [],
                "is_food": True,
            }


# =============================================================================
# OPTIONAL SECONDARY: OPENAI VISION SERVICE
# =============================================================================

OpenAIVisionService = MultimodalVisionService


# =============================================================================
# UNIFIED VISION SERVICE (COORDINATOR ROUTER)
# =============================================================================

class VisionService:
    """
    Unified Vision Service coordinator.
    Provider priority:
      1. Google Gemini Multimodal Vision (Primary, e.g. gemini-3.8-flash)
      2. OpenAI Multimodal Vision (Optional secondary, if explicitly configured)
      3. Google SigLIP Zero-Shot Vision (Local offline fallback)
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VisionService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = "google/siglip-base-patch16-224"):
        if getattr(self, "_initialized", False):
            return

        self.model_name = model_name
        self.gemini_service = GeminiVisionService(
            api_key=GEMINI_API_KEY,
            model=GEMINI_VISION_MODEL,
        )
        # Only initialize OpenAI service if key is provided or configured as primary
        self.openai_service = None
        if VISION_PRIMARY_BACKEND == "multimodal_openai" or (OPENAI_API_KEY and OPENAI_API_KEY.strip()):
            self.openai_service = MultimodalVisionService(
                api_key=OPENAI_API_KEY,
                model=OPENAI_VISION_MODEL,
            )

        # Compatibility alias
        self.multimodal_service = self.gemini_service

        self.gate_prompts = GATE_PROMPTS
        self.specialist_prompts = [item["prompt"] for item in ONTOLOGY_SPECS]
        self._siglip_service = None
        self._initialized = True
        logger.info(
            "VisionService coordinator initialized. Primary backend: %s (model: %s)",
            VISION_PRIMARY_BACKEND,
            self.gemini_service.model if VISION_PRIMARY_BACKEND == "gemini" else (OPENAI_VISION_MODEL if self.openai_service else "none"),
        )

    @property
    def siglip_service(self) -> SigLIPVisionService:
        if self._siglip_service is None:
            self._siglip_service = SigLIPVisionService(model_name=self.model_name)
        return self._siglip_service

    def analyze_image(self, image: Any) -> Dict[str, Any]:
        """
        Main entrypoint for image analysis.
        Attempts Gemini first; if optional OpenAI is configured and selected, uses OpenAI;
        otherwise gracefully falls back to SigLIP local vision.
        Quota exhaustion immediately falls back to SigLIP without retrying.
        """
        # 1. Primary Provider: Google Gemini
        use_gemini = (VISION_PRIMARY_BACKEND == "gemini") and self.gemini_service.is_available()

        if use_gemini:
            try:
                logger.info("Executing Primary Multimodal Vision Pipeline (Google Gemini %s)...", self.gemini_service.model)
                result = self.gemini_service.analyze_image(image)
                return result
            except GeminiQuotaExhaustedError as e:
                logger.warning("Gemini Quota Exhausted (%s). Immediately falling back to local SigLIP classifier without retry.", e)
                fallback_res = self.siglip_service.analyze_image(image)
                fallback_res["source"] = "local_siglip_fallback"
                fallback_res["fallback_reason"] = "gemini_quota_exhausted"
                fallback_res["quota_exhausted"] = True
                fallback_res["fallback_notice"] = "Gemini quota temporarily exhausted — using Local Fallback Vision."
                if hasattr(e, "retry_delay") and e.retry_delay:
                    fallback_res["retry_delay_seconds"] = e.retry_delay
                return fallback_res
            except Exception as e:
                logger.warning("Primary Gemini Multimodal Vision failed (%s). Falling back to local SigLIP classifier.", e)
                fallback_res = self.siglip_service.analyze_image(image)
                fallback_res["source"] = "local_siglip_fallback"
                fallback_res["fallback_reason"] = str(e)
                return fallback_res

        # 2. Optional Secondary Provider: OpenAI Vision
        use_openai = (VISION_PRIMARY_BACKEND == "multimodal_openai") and self.openai_service is not None and self.openai_service.is_available()

        if use_openai:
            try:
                logger.info("Executing Optional Secondary Multimodal Vision Pipeline (OpenAI %s)...", self.openai_service.model)
                result = self.openai_service.analyze_image(image)
                return result
            except Exception as e:
                logger.warning("Secondary OpenAI Vision failed (%s). Falling back to local SigLIP classifier.", e)

        # 3. Fallback to SigLIP Local Vision
        logger.info("Executing Local SigLIP Fallback Vision Pipeline...")
        fallback_res = self.siglip_service.analyze_image(image)
        fallback_res["source"] = "local_siglip_fallback"
        return fallback_res

    def detect(self, frame: Any) -> List[Dict[str, Any]]:
        """Compatibility adapter returning detection bounding box list."""
        res = self.analyze_image(frame)
        if res.get("status") == "not_food" or not res.get("is_food", True):
            return []

        item_name = res.get("name") if res.get("name") else "unknown_food"
        return [
            {
                "class_name": item_name,
                "status": res.get("status"),
                "category": res.get("category"),
                "confidence": res.get("confidence"),
                "visual_description": res.get("visual_description"),
                "bbox": [0, 0, 0, 0],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "source": res.get("source", "multimodal_openai"),
            }
        ]
