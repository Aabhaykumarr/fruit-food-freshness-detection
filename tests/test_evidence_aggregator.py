"""Unit tests for Evidence Aggregator."""

import os
from services.evidence_aggregator import EvidenceAggregator


def test_evidence_aggregation():
    aggregator = EvidenceAggregator(rules_path="knowledge/freshness_rules.json")

    detection = {
        "class_name": "apple",
        "confidence": 0.94,
        "bbox": [100, 100, 300, 300],
        "source": "yolo_world",
    }

    sensor = {
        "temperature_c": 24.5,
        "humidity_percent": 65.0,
        "gas_value": 310.0,
        "sensor_status": "valid",
        "sensor_scope": "environment",
        "source": "mock",
    }

    history = [
        {"item_name": "apple", "freshness_status": "FRESH", "timestamp": "2026-10-01T10:00:00"}
    ]

    packet = aggregator.aggregate(detection, sensor, history=history)

    assert "observation_id" in packet
    assert packet["visual_evidence"]["item_name"] == "apple"
    assert packet["visual_evidence"]["confidence"] == 0.94
    assert packet["sensor_evidence"]["temperature_c"] == 24.5
    assert packet["sensor_evidence"]["sensor_scope"] == "environment"
    assert packet["knowledge_guidelines"]["item_type"] == "fruit"
    assert len(packet["historical_observations"]) == 1


def test_aggregation_missing_sensor():
    aggregator = EvidenceAggregator(rules_path="knowledge/freshness_rules.json")

    detection = {"class_name": "banana", "confidence": 0.88, "bbox": [0, 0, 50, 50]}
    sensor = {"temperature_c": None, "humidity_percent": None, "sensor_status": "unavailable", "source": "none"}

    packet = aggregator.aggregate(detection, sensor)

    assert packet["visual_evidence"]["item_name"] == "banana"
    assert packet["sensor_evidence"]["temperature_c"] is None
    assert packet["sensor_evidence"]["sensor_status"] == "unavailable"
