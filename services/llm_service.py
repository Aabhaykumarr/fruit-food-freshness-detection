"""
LLM Freshness Reasoning Service.
Uses OpenAI GPT-4o-mini with structured outputs (Pydantic schema enforcement)
and Multimodal Vision (base64 image inspection).

Rules enforced:
  1. Never invent missing sensor values.
  2. If an item is classified as non-food, acknowledge it and do not invent food freshness.
  3. If an item is uncertain, reason about visual traits without hallucinating an exact food class.
  4. Separate measured facts from estimates.
  5. Output estimated remaining freshness as ranges with confidence.
  6. Explicitly distinguish visible freshness from microbiological food safety.
  7. Inspect image visual cues (browning, skin texture, mold, wilting, bruising).
  8. Handle multi-item scenes with per-item synthesis and individual condition differentiation.
  9. Never crash on API errors — graceful degradation to rule-based fallback.
"""

import os
import base64
import io
import json
import time
from typing import List, Optional, Literal, Any, Dict
import logging
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

try:
    from config import (
        GEMINI_VISION_MODEL,
        GEMINI_API_KEY,
        OPENAI_API_KEY,
        OPENAI_VISION_MODEL,
        LLM_MODEL,
    )
except ImportError:
    GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-3.8-flash")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")
    LLM_MODEL = os.getenv("LLM_MODEL", GEMINI_VISION_MODEL)

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

logger = logging.getLogger(__name__)


# =============================================================================
# PYDANTIC OUTPUT SCHEMA
# =============================================================================

class RemainingFreshness(BaseModel):
    value: str = Field(description="Estimated remaining freshness window, e.g. '1-3' or '2-4' or 'N/A' or 'insufficient evidence'")
    unit: str = Field(description="Time unit: 'hours', 'days', 'weeks', or 'N/A'")
    range_description: str = Field(description="Human-readable range, e.g. 'approximately 1–3 days under current conditions' or 'N/A (Not food)'")
    confidence: Literal["high", "medium", "low", "very_low"] = Field(description="Confidence level in this estimate")
    basis: List[str] = Field(description="Key factors informing this estimate (temp, appearance, typical shelf life, etc.)")


class ItemFreshnessSummary(BaseModel):
    item_name: str = Field(description="Name of the specific food item")
    category: str = Field(description="Category of the item: fruit, vegetable, bakery, dish, staple, etc.")
    condition: str = Field(description="Observed condition: fresh, ripe, overripe, bruised, wilted, stale, moldy, rotten, etc.")
    freshness_status: str = Field(description="Freshness status: FRESH, MODERATELY_FRESH, QUESTIONABLE, NOT_FRESH, UNKNOWN")
    is_spoiled: bool = Field(description="True if spoiled or moldy")
    shelf_life_estimate: str = Field(description="Estimated shelf life for this specific item (e.g. '0 days (discard)', '3-5 days')")
    key_observation: str = Field(description="Specific visual observation for this item")


class FreshnessAnalysis(BaseModel):
    item_name: Optional[str] = Field(description="The name of the primary food/fruit item or summary of items analyzed, or 'Not Food' if non-food")
    detection_confidence: float = Field(description="Visual detection / classification confidence (0.0 to 1.0)")
    freshness_status: Literal["FRESH", "MODERATELY_FRESH", "QUESTIONABLE", "NOT_FRESH", "UNKNOWN", "NOT_FOOD"] = Field(
        description="Overall freshness category"
    )
    estimated_remaining_freshness: RemainingFreshness = Field(description="Estimated remaining freshness window")
    visual_observations: List[str] = Field(description="Detailed visual observations from image (color, spots, skin, texture, mold, bruising)")
    environmental_context: str = Field(description="Summary of how current temperature/humidity/gas readings impact this item")
    reasoning_summary: str = Field(description="Concise synthesis explaining why this freshness status was assigned, noting individual item differences if multiple items are present")
    uncertainty_factors: List[str] = Field(description="Explicit unknowns and limitations of this estimate")
    recommendations: List[str] = Field(description="Actionable storage or consumption suggestions")
    item_summaries: Optional[List[ItemFreshnessSummary]] = Field(
        default_factory=list,
        description="Per-item breakdown when multiple food items are present"
    )


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are an expert Fruit & Food Freshness and Food Science Assistant.

You analyze food and fruit items using:
1. High-resolution visual image inspection (examining surface color, wrinkles, browning, bruising, mold colonies, firmness cues, moisture)
2. Multimodal scene understanding (supporting both single-item images and multi-item grouped produce/dishes)
3. Environmental sensor evidence (ambient temperature, humidity, gas/VOC readings from Arduino or mock sensors)
4. Reference storage guidelines for the food type
5. Hierarchical vision gating (handling recognized foods, uncertain/OOD foods, and non-food items)

CRITICAL RULES YOU MUST FOLLOW:
1. NON-FOOD ITEMS: If the visual evidence indicates the item is 'not_food' or the image clearly contains an electronic device, face, furniture, or household object, set freshness_status to 'NOT_FOOD', item_name to 'Not Food', and state clearly that this is not an edible food item.
2. MULTI-ITEM SCENES: If multiple food items are present (e.g., moldy bread next to fresh tomatoes and onions), do NOT treat the entire scene as a single dish. Differentiate each item in `item_summaries`. State overall risk and provide itemized advice (e.g. discard the moldy bread; wash and consume the fresh produce).
3. UNCERTAIN / OUT-OF-DISTRIBUTION FOODS: If the visual status is 'uncertain', do NOT hallucinate an exact food name. Describe visible food characteristics (e.g., 'Cooked rice dish with vegetables', 'Mixed regional curry', 'Exotic tropical fruit') and provide prudent freshness advice.
4. REASON STRICTLY FROM EVIDENCE: Base your assessment only on the provided image, sensor evidence, and established food science.
5. NEVER INVENT SENSOR VALUES: If a sensor value is null/unavailable, acknowledge it is missing. Do not guess it.
6. ENVIRONMENTAL VS OBJECT DISTINCTION: Remember that sensors measure the surrounding environment, NOT the internal temperature of the food.
7. ESTIMATE WITH RANGES: Never provide an exact single-number expiration time. Always provide a reasonable range (e.g. '1-3 days') with clear uncertainty.
8. FRESHNESS != FOOD SAFETY: Visible freshness and environmental conditions do not guarantee microbiological food safety. Explicitly state this uncertainty.
9. STRUCTURED OUTPUT ONLY: Return your analysis strictly matching the required schema.
"""


# =============================================================================
# SERVICE IMPLEMENTATION
# =============================================================================

class LLMFreshnessService:
    """
    Calls Google Gemini (Primary) or OpenAI (Optional) with structured outputs & Multimodal Vision
    to reason over multimodal evidence.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, timeout: float = 30.0):
        if api_key is not None:
            self.gemini_api_key = api_key
            self.openai_api_key = ""  # Explicit override takes precedence
        else:
            self.gemini_api_key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
            self.openai_api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
        self.gemini_model = model if model is not None else (GEMINI_VISION_MODEL or os.getenv("GEMINI_VISION_MODEL", "gemini-3.8-flash"))
        self.openai_model = OPENAI_VISION_MODEL or os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")
        self.timeout = timeout
        self.gemini_client = None
        self.openai_client = None
        self._init_clients()

    def _init_clients(self):
        """Initialize Gemini (Primary) or OpenAI (Optional) clients if API keys are provided."""
        if self.gemini_api_key:
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=self.gemini_api_key)
                logger.info("Gemini LLM client initialized with model: %s", self.gemini_model)
            except Exception as e:
                logger.error("Failed to initialize Gemini LLM client: %s", e)
                self.gemini_client = None

        if self.openai_api_key and self.openai_api_key.strip():
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=self.openai_api_key, timeout=self.timeout)
                logger.info("OpenAI LLM client initialized with model: %s", self.openai_model)
            except Exception as e:
                logger.error("Failed to initialize OpenAI LLM client: %s", e)
                self.openai_client = None

        if not self.gemini_client and not self.openai_client:
            logger.warning("No GEMINI_API_KEY or OPENAI_API_KEY provided. LLM service running in offline fallback mode.")

    # Property for test backward compatibility
    @property
    def client(self):
        return self.gemini_client or self.openai_client

    @client.setter
    def client(self, value):
        self.gemini_client = value

    def _prepare_pil_image(self, image: Any) -> Optional[Image.Image]:
        """Convert various image formats to PIL Image."""
        if image is None:
            return None
        from PIL import Image
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, str):
            if image.startswith("data:image"):
                raw_b64 = image.split(",", 1)[1]
                return Image.open(io.BytesIO(base64.b64decode(raw_b64))).convert("RGB")
            if len(image) > 500 and not image.endswith((".jpg", ".png", ".jpeg")):
                return Image.open(io.BytesIO(base64.b64decode(image))).convert("RGB")
            if os.path.exists(image):
                return Image.open(image).convert("RGB")
        if isinstance(image, bytes):
            return Image.open(io.BytesIO(image)).convert("RGB")
        if isinstance(image, np.ndarray):
            try:
                import cv2
                if len(image.shape) == 3:
                    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                elif len(image.shape) == 2:
                    return Image.fromarray(image).convert("RGB")
            except Exception:
                pass
        return None

    def _encode_image(self, image: Any) -> Optional[str]:
        """Convert various image formats (numpy array, PIL Image, bytes, base64 string) to base64 JPEG."""
        if image is None:
            return None

        if isinstance(image, str):
            if image.startswith("data:image"):
                return image.split(",", 1)[1]
            if len(image) > 500 and not image.endswith((".jpg", ".png", ".jpeg")):
                return image  # Raw base64 string
            try:
                with open(image, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
            except Exception:
                return None

        if isinstance(image, bytes):
            return base64.b64encode(image).decode("utf-8")

        try:
            # OpenCV numpy ndarray
            import cv2
            if hasattr(image, "shape"):
                success, buffer = cv2.imencode(".jpg", image)
                if success:
                    return base64.b64encode(buffer).decode("utf-8")
        except Exception:
            pass

        try:
            # PIL Image
            from PIL import Image
            if isinstance(image, Image.Image):
                buf = io.BytesIO()
                image.convert("RGB").save(buf, format="JPEG")
                return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            pass

        return None

    def analyze(self, evidence_packet: dict, image: Optional[Any] = None) -> FreshnessAnalysis:
        """
        Analyze an evidence packet (and optional image) and return a structured FreshnessAnalysis.
        Uses Gemini (Primary), OpenAI (Optional), or falls back to a deterministic rule-based assessment.
        """
        # Non-food immediate handling if indicated in evidence
        status = evidence_packet.get("status") or evidence_packet.get("visual_evidence", {}).get("status")
        is_food = evidence_packet.get("is_food", status != "not_food")
        if status == "not_food" or not is_food:
            visual_desc = evidence_packet.get("visual_evidence", {}).get("visual_description", "Non-food object detected.")
            return FreshnessAnalysis(
                item_name="Not Food",
                detection_confidence=evidence_packet.get("visual_evidence", {}).get("confidence", 1.0),
                freshness_status="NOT_FOOD",
                estimated_remaining_freshness=RemainingFreshness(
                    value="N/A",
                    unit="N/A",
                    range_description="Not applicable for non-food objects.",
                    confidence="high",
                    basis=["Stage 1 Zero-Shot Non-Food Gate rejected this item."],
                ),
                visual_observations=[visual_desc],
                environmental_context="Environmental telemetry ignored for non-food objects.",
                reasoning_summary=f"The presented image was identified as a non-food object ({visual_desc}). No freshness analysis or consumption advice applies.",
                uncertainty_factors=["Object is outside food domain."],
                recommendations=["Please present a valid fruit, vegetable, or prepared dish for freshness analysis."],
                item_summaries=[],
            )

        user_text = f"Analyze this food item evidence packet and provide a comprehensive freshness assessment:\n\n{json.dumps(evidence_packet, indent=2)}"

        # 1. Primary: Google Gemini Structured Reasoning
        if self.gemini_client is not None:
            for attempt in range(2):
                try:
                    from google.genai import types
                    config = types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        response_schema=FreshnessAnalysis,
                        temperature=0.2,
                    )
                    pil_img = self._prepare_pil_image(image)
                    contents = [pil_img, user_text] if pil_img is not None else [user_text]

                    response = self.gemini_client.models.generate_content(
                        model=self.gemini_model,
                        contents=contents,
                        config=config,
                    )
                    if hasattr(response, "parsed") and isinstance(response.parsed, FreshnessAnalysis):
                        return response.parsed
                    elif response.text:
                        return FreshnessAnalysis.model_validate_json(response.text)
                except Exception as e:
                    err_str = str(e).lower()
                    # Quota exhaustion: 429 RESOURCE_EXHAUSTED -> immediate fallback, no retry
                    if "429" in str(e) or "resource_exhausted" in err_str or "quota" in err_str:
                        logger.warning("Gemini quota exhausted during LLM analysis (429 RESOURCE_EXHAUSTED). Immediate fallback to rules.")
                        break

                    # HTTP 503: retry at most once with 1s backoff
                    if "503" in str(e) or "service unavailable" in err_str:
                        if attempt == 0:
                            logger.warning("Gemini LLM 503 Service Unavailable — retrying once after 1s backoff.")
                            time.sleep(1.0)
                            continue
                        else:
                            logger.warning("Gemini LLM 503 persisted after retry. Falling back.")
                            break

                    logger.warning("Gemini LLM analysis attempt %d failed (%s).", attempt + 1, e)
                    break

        # 2. Optional Secondary: OpenAI Structured Reasoning
        if self.openai_client is not None:
            try:
                b64_image = self._encode_image(image)
                if b64_image:
                    content = [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                else:
                    content = user_text

                completion = self.openai_client.beta.chat.completions.parse(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": content},
                    ],
                    response_format=FreshnessAnalysis,
                )

                parsed = completion.choices[0].message.parsed
                if parsed is not None:
                    return parsed
            except Exception as e:
                logger.error("OpenAI API call failed: %s. Using rule-based fallback.", e)

        # 3. Fallback rule-based analysis when LLM is unavailable
        return self._rule_based_fallback(evidence_packet)

    def _rule_based_fallback(self, evidence: dict) -> FreshnessAnalysis:
        """
        Deterministic rule-based fallback when OpenAI API is unavailable or offline.
        Uses baseline knowledge rules and sensor thresholds.
        """
        visual = evidence.get("visual_evidence", {})
        sensors = evidence.get("sensor_evidence", {})
        rules = evidence.get("knowledge_guidelines", {})
        items_ev = evidence.get("items_evidence", [])

        status = visual.get("status", "recognized")
        item_name = visual.get("item_name")
        conf = visual.get("confidence", 0.0)
        visual_desc = visual.get("visual_description", "")
        temp = sensors.get("temperature_c")
        hum = sensors.get("humidity_percent")
        sensor_stat = sensors.get("sensor_status", "unavailable")

        if status == "not_food":
            return FreshnessAnalysis(
                item_name="Not Food",
                detection_confidence=conf,
                freshness_status="NOT_FOOD",
                estimated_remaining_freshness=RemainingFreshness(
                    value="N/A",
                    unit="N/A",
                    range_description="Not applicable for non-food objects.",
                    confidence="high",
                    basis=["Zero-shot gate rejected object."],
                ),
                visual_observations=[visual_desc or "Non-food object detected."],
                environmental_context="Sensors active but skipped for non-food.",
                reasoning_summary="Visual gate determined this image is not food or fruit.",
                uncertainty_factors=["Non-food item."],
                recommendations=["Capture a food item or fruit."],
                item_summaries=[],
            )

        # Check if any item has mold or spoilage
        has_spoiled_item = any(it.get("is_spoiled", False) for it in items_ev)
        item_summaries = []

        for it in items_ev:
            it_name = it.get("name", "food")
            it_cond = it.get("condition", "fresh")
            it_spoil = it.get("is_spoiled", False)
            it_stat = "NOT_FRESH" if it_spoil else ("FRESH" if "fresh" in it_cond.lower() else "MODERATELY_FRESH")
            it_shelf = "0 days (discard)" if it_spoil else str(it.get("storage_guidelines", {}).get("typical_shelf_life_days_room_temp", "2-4") + " days")
            item_summaries.append(ItemFreshnessSummary(
                item_name=it_name,
                category=it.get("category", "other"),
                condition=it_cond,
                freshness_status=it_stat,
                is_spoiled=it_spoil,
                shelf_life_estimate=it_shelf,
                key_observation=it.get("visual_description") or f"{it_name} observed in {it_cond} condition",
            ))

        if status == "uncertain":
            display_name = "Unidentified Food / Dish"
            freshness_status = "UNKNOWN"
            remaining_days = "1-2"
            obs_desc = visual_desc or "Visual features resemble food/produce, but specific category is uncertain."
        elif has_spoiled_item:
            display_name = item_name if item_name else "mixed_food"
            freshness_status = "NOT_FRESH"
            remaining_days = "0"
            obs_desc = visual_desc or "Visible spoilage or mold observed on one or more items."
        else:
            display_name = item_name if item_name else "recognized_item"
            freshness_status = "FRESH"
            remaining_days = rules.get("typical_shelf_life_days_room_temp", "2-5")
            obs_desc = visual_desc or f"Detected '{display_name}' with {conf*100:.1f}% confidence."

        env_notes = []
        if temp is not None:
            env_notes.append(f"Ambient temperature is {temp}°C.")
            if temp > 30.0:
                if freshness_status == "FRESH":
                    freshness_status = "MODERATELY_FRESH"
                env_notes.append("High ambient temperature accelerates spoilage.")
            elif temp < 4.0:
                env_notes.append("Cold storage slows degradation.")
        else:
            env_notes.append("Temperature reading unavailable.")

        if hum is not None:
            env_notes.append(f"Ambient humidity is {hum}%.")
            if hum > 85.0:
                env_notes.append("High humidity increases mold risk.")
        else:
            env_notes.append("Humidity reading unavailable.")

        return FreshnessAnalysis(
            item_name=display_name,
            detection_confidence=conf,
            freshness_status=freshness_status,
            estimated_remaining_freshness=RemainingFreshness(
                value=str(remaining_days),
                unit="days",
                range_description=f"Approximately {remaining_days} days under room temperature guidelines (rule-based fallback)",
                confidence="low",
                basis=["Standard shelf-life reference table", f"Sensor status: {sensor_stat}"],
            ),
            visual_observations=[obs_desc],
            environmental_context=" ".join(env_notes),
            reasoning_summary=f"Rule-based baseline estimate for '{display_name}'. Connect OPENAI_API_KEY for deep multimodal reasoning.",
            uncertainty_factors=[
                "Generated via offline rule-based fallback (LLM not connected).",
                "Sensors measure ambient environment, not internal food core temperature.",
                "Visual analysis cannot detect invisible bacterial contamination.",
            ],
            recommendations=[
                f"Store '{display_name}' in appropriate temperature and humidity.",
                "Always check smell and appearance before consumption.",
            ],
            item_summaries=item_summaries,
        )

    # =========================================================================
    # CHAT ABOUT ANALYSIS
    # =========================================================================

    CHAT_SYSTEM_PROMPT = """You are a helpful food science assistant embedded in a Fruit & Food Freshness Recognition System.

You answer user questions about a food/fruit item or multi-item scene that was just analyzed by the system.
You have access to the full analysis context provided below.

CRITICAL RULES:
1. ONLY use the provided analysis context to answer. Do NOT invent data.
2. MULTI-ITEM CONTEXT: If multiple items were detected (e.g., bread with mold, onions, tomatoes), you must answer specifically about each requested item using its condition and freshness status.
3. If the detection status is "recognized", you may discuss the detected food by name.
4. If the detection status is "uncertain", you MUST preserve that uncertainty. Do NOT claim a definite identity. Describe visual characteristics only.
5. If the detection status is "not_food", do NOT discuss the object as food. State clearly: "The image was classified as not a fruit or food, so there isn't a food identity to analyze."
6. If sensor data is unavailable or null, say the reading is unavailable. NEVER invent sensor values.
7. Distinguish between: model/vision observations, sensor measurements, freshness-rule conclusions, and general food science explanation.
8. Do NOT claim laboratory-grade food safety. Always caveat that visible assessment differs from microbiological testing.
9. Keep answers concise, helpful, and grounded in the provided context.
"""

    def chat_about_analysis(
        self,
        analysis_context: dict,
        conversation_history: list,
        user_question: str,
    ) -> str:
        """
        Answer a user question about the currently analyzed food item.
        Supports Gemini (Primary) and OpenAI (Optional).
        """
        if not user_question or not user_question.strip():
            return "Please type a question about the analyzed food item."

        if self.gemini_client is None and self.openai_client is None:
            return "Online AI chat is unavailable because GEMINI_API_KEY (or OPENAI_API_KEY) is not configured."

        context_block = json.dumps(analysis_context, indent=2, default=str)

        # 1. Primary: Google Gemini Chat
        if self.gemini_client is not None:
            for attempt in range(2):
                try:
                    from google.genai import types

                    chat_prompt = (
                        f"CURRENT ANALYSIS CONTEXT:\n{context_block}\n\n"
                        "PREVIOUS CONVERSATION:\n"
                    )
                    for turn in conversation_history:
                        chat_prompt += f"{turn['role'].upper()}: {turn['content']}\n"
                    chat_prompt += f"\nUSER QUESTION: {user_question}"

                    config = types.GenerateContentConfig(
                        system_instruction=self.CHAT_SYSTEM_PROMPT,
                        temperature=0.4,
                    )

                    response = self.gemini_client.models.generate_content(
                        model=self.gemini_model,
                        contents=[chat_prompt],
                        config=config,
                    )
                    if response.text:
                        return response.text
                except Exception as e:
                    err_str = str(e).lower()
                    # Quota exhaustion: 429 RESOURCE_EXHAUSTED -> immediate fallback, do not hammer repeatedly
                    if "429" in str(e) or "resource_exhausted" in err_str or "quota" in err_str:
                        logger.warning("Gemini quota exhausted during chat (429 RESOURCE_EXHAUSTED). Immediate fallback.")
                        return (
                            "⚠️ Gemini API quota is temporarily exhausted for today (RPD limit reached). "
                            "Chat responses from the cloud model are currently unavailable. "
                            "Please refer to the rule-based freshness recommendations above."
                        )

                    # HTTP 503: retry at most once with 1s backoff
                    if "503" in str(e) or "service unavailable" in err_str:
                        if attempt == 0:
                            logger.warning("Gemini chat 503 Service Unavailable — retrying once after 1s backoff.")
                            time.sleep(1.0)
                            continue
                        else:
                            logger.warning("Gemini chat 503 persisted after retry. Falling back.")
                            break

                    logger.warning("Gemini chat attempt %d failed (%s).", attempt + 1, e)
                    break

        # 2. Optional Secondary: OpenAI Chat
        if self.openai_client is not None:
            try:
                messages = [
                    {"role": "system", "content": self.CHAT_SYSTEM_PROMPT},
                    {
                        "role": "system",
                        "content": f"CURRENT ANALYSIS CONTEXT:\n{context_block}",
                    },
                ]
                for turn in conversation_history:
                    messages.append({
                        "role": turn["role"],
                        "content": turn["content"],
                    })
                messages.append({"role": "user", "content": user_question})

                completion = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=messages,
                    max_tokens=1024,
                    temperature=0.4,
                )
                return completion.choices[0].message.content or "I couldn't generate a response. Please try again."
            except Exception as e:
                logger.error("OpenAI chat API call failed: %s", e)
                return f"⚠️ Unable to reach the AI service right now. Error: {e}"

        return "⚠️ Unable to reach the AI chat service. Please verify your API key."
