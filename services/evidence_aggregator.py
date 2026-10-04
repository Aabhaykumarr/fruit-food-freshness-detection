"""
Multimodal Evidence Aggregator.
Combines:
  1. Visual evidence (Multi-item structured results or single-item classification)
  2. Sensor evidence (Arduino/mock/web-serial: temp, humidity, gas, scope, status)
  3. Knowledge rules (storage guidelines for all detected food items)
  4. Historical observations (previous readings from SQLite)

Produces a clean, structured EvidencePacket dict ready for LLM reasoning and UI presentation.
"""

import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class EvidenceAggregator:
    """
    Fuses multi-source evidence into a single structured packet with clear data provenance.
    Supports both multi-item scenes and single-item focus.
    """

    def __init__(self, rules_path: str = "knowledge/freshness_rules.json"):
        self.rules_path = rules_path
        self.rules = self._load_rules()

    def _load_rules(self) -> Dict[str, Any]:
        """Load the baseline freshness knowledge base."""
        try:
            with open(self.rules_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("rules", {})
        except Exception as e:
            logger.warning("Could not load freshness rules from %s: %s", self.rules_path, e)
            return {}

    def get_knowledge_for_item(self, item_name: Optional[str]) -> Optional[Dict[str, Any]]:
        """Lookup reference storage rules for a detected food item."""
        if not item_name:
            return None
        clean_name = item_name.lower().strip().replace(" ", "_")
        if clean_name in self.rules:
            return self.rules[clean_name]
        return self.rules.get(item_name.lower().strip())

    def aggregate(
        self,
        vision_result: Dict[str, Any],
        sensor_reading: Dict[str, Any],
        history: Optional[List[Dict[str, Any]]] = None,
        user_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate all available evidence for a single image/observation.

        Accepts:
        - Multimodal vision result dictionary with `items` list and top-level fields
        - Sensor reading dictionary (temp, humidity, gas, status, source)
        - History list
        - User metadata

        Returns a structured evidence packet.
        """
        now = datetime.now().isoformat()
        observation_id = str(uuid.uuid4())

        # Determine visual metadata
        status = vision_result.get("status", "recognized")
        item_name = vision_result.get("name") or vision_result.get("class_name")
        category = vision_result.get("category")
        confidence = vision_result.get("confidence", 0.0)
        visual_desc = vision_result.get("visual_description", "")
        raw_predictions = vision_result.get("raw_predictions", [])
        items_raw = vision_result.get("items", [])
        source = vision_result.get("source", "multimodal_openai")
        is_food = vision_result.get("is_food", status != "not_food")

        if status == "not_food":
            item_name = None

        # Build enhanced per-item evidence list with per-item knowledge lookup
        items_evidence = []
        for it in items_raw:
            it_name = it.get("name", "")
            it_knowledge = self.get_knowledge_for_item(it_name) if (it_name and it_name != "uncertain") else None
            items_evidence.append({
                "item_id": it.get("item_id", len(items_evidence) + 1),
                "name": it_name,
                "category": it.get("category", "other"),
                "confidence": it.get("confidence", 0.0),
                "condition": it.get("condition", "fresh"),
                "condition_confidence": it.get("condition_confidence", 0.0),
                "freshness_status": it.get("freshness_status", "FRESH"),
                "visible_signs": it.get("visible_signs", []),
                "visual_description": it.get("visual_description", ""),
                "is_spoiled": it.get("is_spoiled", False),
                "reason": it.get("reason", ""),
                "alternatives": it.get("alternatives", []),
                "storage_guidelines": it_knowledge or {"note": f"Apply standard storage guidelines for {it_name or 'food'}."},
            })

        visual_evidence = {
            "status": status,
            "item_name": item_name,
            "category": category,
            "confidence": confidence,
            "visual_description": visual_desc,
            "items": items_evidence,
            "overall_visual_summary": vision_result.get("overall_visual_summary", visual_desc),
            "raw_predictions": raw_predictions,
            "detection_source": source,
            "detection_timestamp": vision_result.get("timestamp", now),
            "is_food": is_food,
        }

        # Sensor evidence block (with provenance & scope)
        sensor_evidence = {
            "temperature_c": sensor_reading.get("temperature_c"),
            "humidity_percent": sensor_reading.get("humidity_percent"),
            "gas_value": sensor_reading.get("gas_value"),
            "sensor_status": sensor_reading.get("sensor_status", "unavailable"),
            "sensor_scope": sensor_reading.get("sensor_scope", "environment"),
            "sensor_source": sensor_reading.get("source", "none"),
            "sensor_timestamp": sensor_reading.get("timestamp", now),
        }

        # Knowledge baseline for primary/overall item
        if item_name:
            knowledge = self.get_knowledge_for_item(item_name) or {
                "note": f"No pre-configured knowledge entry for '{item_name}'. Use general food science principles."
            }
        elif status == "uncertain":
            knowledge = {
                "note": "Item could not be identified with certainty. Apply general storage guidelines for cooked food / perishable produce."
            }
        else:
            knowledge = {
                "note": "Non-food item detected. No freshness rules apply."
            }

        hist_list = history or []

        return {
            "observation_id": observation_id,
            "timestamp": now,
            "status": status,
            "is_food": is_food,
            "visual_evidence": visual_evidence,
            "sensor_evidence": sensor_evidence,
            "knowledge_guidelines": knowledge,
            "items_evidence": items_evidence,
            "historical_observations": hist_list,
            "user_metadata": user_metadata or {},
        }
