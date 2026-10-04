"""
Unit tests for interactive sequential workflow, LCD messaging, and trigger snapshots.
"""

import numpy as np
import pytest
from services.sensor_service import SensorService
from utils.display import (
    draw_food_boxes,
    draw_searching_hud,
    draw_placement_prompt,
    draw_measuring_hud,
    draw_analysis_card,
    draw_sensor_bar,
)
from services.llm_service import FreshnessAnalysis, RemainingFreshness


def test_lcd_message_formatting():
    service = SensorService(mock_mode=True)
    # Should not raise exception in mock mode
    service.send_lcd_message("Detected: Apple", "Place at sensor")
    service.send_lcd_message("Line with very long text that exceeds sixteen characters limit", "Second long line")


def test_capture_snapshot_marking():
    service = SensorService(mock_mode=True)
    service.start()
    snapshot = service.capture_snapshot()
    service.stop()

    assert "captured_at" in snapshot
    assert snapshot.get("is_triggered_reading") is True
    assert "sensor_status" in snapshot


def test_hardware_button_detection():
    service = SensorService(mock_mode=True)
    assert service.check_button_pressed() is False

    # Simulate button press
    service._button_pressed = True
    assert service.check_button_pressed() is True
    # Flag should auto-clear
    assert service.check_button_pressed() is False


def test_display_overlays_render_without_errors():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. Searching HUD
    f1 = frame.copy()
    draw_searching_hud(f1)
    assert f1.shape == (480, 640, 3)

    # 2. Food Bounding Box
    f2 = frame.copy()
    detections = [{"class_name": "apple", "confidence": 0.95, "bbox": [50, 50, 200, 200]}]
    draw_food_boxes(f2, detections)
    # Check that green pixels were drawn
    assert np.any(f2[:, :, 1] > 0)

    # 3. Placement Prompt
    f3 = frame.copy()
    draw_placement_prompt(f3, "apple", 0.95)
    assert f3.shape == (480, 640, 3)

    # 4. Measuring HUD
    f4 = frame.copy()
    draw_measuring_hud(f4, "apple")
    assert f4.shape == (480, 640, 3)

    # 5. Analysis Card
    f5 = frame.copy()
    analysis = FreshnessAnalysis(
        item_name="apple",
        detection_confidence=0.95,
        freshness_status="FRESH",
        estimated_remaining_freshness=RemainingFreshness(
            value="3-5",
            unit="days",
            range_description="3 to 5 days under ambient room conditions",
            confidence="high",
            basis=["appearance", "ambient temp"],
        ),
        visual_observations=["Firm skin", "No browning"],
        environmental_context="Temperature 24.5C is typical.",
        reasoning_summary="Apple appears in good condition.",
        uncertainty_factors=["Internal core quality unverified."],
        recommendations=["Keep in refrigerator for extended freshness."],
    )
    sensor_data = {"temperature_c": 24.5, "humidity_percent": 65.0, "gas_value": 310, "source": "mock", "sensor_status": "valid"}
    draw_analysis_card(f5, analysis, sensor_data)
    draw_sensor_bar(f5, sensor_data)
    assert f5.shape == (480, 640, 3)
