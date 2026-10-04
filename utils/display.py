"""
Display utility — renders clean OpenCV overlays for fruit/food freshness detection.

Features:
  - Vibrant GREEN bounding boxes for verified food items
  - Clean Searching HUD when no food is present
  - Placement Prompt Card directing user to place food at sensor and press trigger
  - Comprehensive Multimodal Analysis Card with freshness badge, shelf-life, and telemetry
  - Ambient Sensor Telemetry status bar
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional

# Color palette (BGR for OpenCV)
COLOR_FOOD_GREEN = (0, 255, 0)     # Vibrant Green for food bounding boxes
COLOR_FRESH = (0, 220, 0)          # Green
COLOR_MODERATE = (0, 215, 255)      # Yellow
COLOR_QUESTIONABLE = (0, 140, 255)  # Orange
COLOR_NOT_FRESH = (0, 0, 255)       # Red
COLOR_UNKNOWN = (160, 160, 160)     # Gray
COLOR_PANEL_BG = (25, 25, 25)       # Dark overlay background
COLOR_TEXT_WHITE = (255, 255, 255)

FRESHNESS_COLORS = {
    "FRESH": COLOR_FRESH,
    "MODERATELY_FRESH": COLOR_MODERATE,
    "QUESTIONABLE": COLOR_QUESTIONABLE,
    "NOT_FRESH": COLOR_NOT_FRESH,
    "UNKNOWN": COLOR_UNKNOWN,
}


def get_freshness_color(status: Optional[str]):
    return FRESHNESS_COLORS.get(status or "UNKNOWN", COLOR_UNKNOWN)


def draw_food_boxes(frame, detections: List[Dict[str, Any]]):
    """
    Draw clean, vibrant GREEN bounding boxes exclusively for detected food items.
    """
    h, w, _ = frame.shape

    for det in detections:
        bbox = det.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = bbox

        # Clamp coordinates
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        class_name = det.get("class_name", "food").upper()
        conf = det.get("confidence", 0.0)

        # 1. Vibrant Green Bounding Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_FOOD_GREEN, 2)

        # 2. Top-left Label badge
        label = f"FOOD: {class_name} ({int(conf * 100)}%)"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y = max(22, y1 - 6)

        # Label background
        cv2.rectangle(
            frame,
            (x1, label_y - th - 4),
            (x1 + tw + 8, label_y + baseline),
            (10, 10, 10),
            cv2.FILLED,
        )
        # Label text in green
        cv2.putText(
            frame,
            label,
            (x1 + 4, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            COLOR_FOOD_GREEN,
            2,
        )


def draw_searching_hud(frame):
    """
    Draws a clean top guide when no food is currently detected in frame.
    """
    h, w, _ = frame.shape
    text = "Scanning... Place food / fruit in view of camera"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    # Semi-transparent top bar
    bar_h = 32
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(
        frame,
        text,
        (max(15, (w - tw) // 2), 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 230, 255),
        1,
    )


def draw_placement_prompt(frame, food_name: str, confidence: float):
    """
    Renders an interactive HUD instruction card prompting the user to place the detected food
    near the physical sensors and press [SPACE] or the Arduino button.
    """
    h, w, _ = frame.shape

    # Panel dimensions (centered card)
    pw = min(560, w - 40)
    ph = 165
    px1 = (w - pw) // 2
    py1 = (h - ph) // 2 - 20
    px2 = px1 + pw
    py2 = py1 + ph

    # Create semi-transparent overlay card
    overlay = frame.copy()
    cv2.rectangle(overlay, (px1, py1), (px2, py2), (15, 15, 15), cv2.FILLED)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Card border in Vibrant Green
    cv2.rectangle(frame, (px1, py1), (px2, py2), COLOR_FOOD_GREEN, 2)

    # Header: Food Detected
    title = f"FOOD LOCKED: {food_name.upper()} ({int(confidence * 100)}%)"
    cv2.putText(
        frame,
        title,
        (px1 + 20, py1 + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        COLOR_FOOD_GREEN,
        2,
    )

    # Divider line
    cv2.line(frame, (px1 + 20, py1 + 45), (px2 - 20, py1 + 45), (80, 80, 80), 1)

    # Instruction 1: Place at sensor
    step1 = f"1. Place {food_name.upper()} close to the sensor module"
    cv2.putText(
        frame,
        step1,
        (px1 + 20, py1 + 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
    )

    # Instruction 2: Trigger measurement
    step2 = "2. Press [SPACE] or Arduino Button to Measure"
    cv2.putText(
        frame,
        step2,
        (px1 + 20, py1 + 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 220, 255),
        2,
    )

    # Instruction 3: Cancel / Re-scan
    step3 = "Press [R] to Cancel / Re-scan other food"
    cv2.putText(
        frame,
        step3,
        (px1 + 20, py1 + 138),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (160, 160, 160),
        1,
    )


def draw_measuring_hud(frame, food_name: str):
    """
    Renders instantaneous measurement HUD when user triggers the sensor reading.
    """
    h, w, _ = frame.shape
    pw = min(480, w - 40)
    ph = 100
    px1 = (w - pw) // 2
    py1 = (h - ph) // 2
    px2 = px1 + pw
    py2 = py1 + ph

    overlay = frame.copy()
    cv2.rectangle(overlay, (px1, py1), (px2, py2), (15, 15, 15), cv2.FILLED)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 200, 255), 2)

    msg = f"Measuring Sensors for {food_name.upper()}..."
    cv2.putText(
        frame,
        msg,
        (px1 + 20, py1 + 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 220, 255),
        2,
    )
    cv2.putText(
        frame,
        "Fusing multimodal evidence & reasoning freshness...",
        (px1 + 20, py1 + 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
    )


def draw_analysis_card(frame, analysis, sensor_data: Dict[str, Any]):
    """
    Renders comprehensive freshness report card over the frame.
    """
    h, w, _ = frame.shape
    pw = min(580, w - 30)
    ph = min(320, h - 60)
    px1 = (w - pw) // 2
    py1 = 20
    px2 = px1 + pw
    py2 = py1 + ph

    overlay = frame.copy()
    cv2.rectangle(overlay, (px1, py1), (px2, py2), (18, 18, 18), cv2.FILLED)
    cv2.addWeighted(overlay, 0.90, frame, 0.10, 0, frame)

    status = getattr(analysis, "freshness_status", "UNKNOWN")
    status_color = get_freshness_color(status)

    # Border
    cv2.rectangle(frame, (px1, py1), (px2, py2), status_color, 2)

    # 1. Header: Item Name + Status Badge
    item_name = getattr(analysis, "item_name", "Food Item").upper()
    conf = getattr(analysis, "detection_confidence", 0.0)
    header_text = f"{item_name} ({int(conf * 100)}%)"
    cv2.putText(
        frame,
        header_text,
        (px1 + 20, py1 + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        COLOR_TEXT_WHITE,
        2,
    )

    badge_text = f"[{status}]"
    (bw, _), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(
        frame,
        badge_text,
        (px2 - bw - 20, py1 + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2,
    )

    # Divider
    cv2.line(frame, (px1 + 20, py1 + 48), (px2 - 20, py1 + 48), (70, 70, 70), 1)

    # 2. Remaining Freshness Window
    est_rf = getattr(analysis, "estimated_remaining_freshness", None)
    rf_desc = "Standard shelf life applies"
    if est_rf is not None:
        if hasattr(est_rf, "range_description"):
            rf_desc = est_rf.range_description
        elif isinstance(est_rf, dict):
            rf_desc = est_rf.get("range_description", str(est_rf))

    cv2.putText(
        frame,
        f"Shelf-Life: {rf_desc}",
        (px1 + 20, py1 + 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        COLOR_FOOD_GREEN,
        2,
    )

    # 3. Triggered Sensor Telemetry Snapshot
    temp = sensor_data.get("temperature_c")
    hum = sensor_data.get("humidity_percent")
    gas = sensor_data.get("gas_value")
    s_src = sensor_data.get("source", "none").upper()
    t_str = f"{temp:.1f}C" if temp is not None else "N/A"
    h_str = f"{hum:.1f}%" if hum is not None else "N/A"
    g_str = f"{gas:.0f}" if gas is not None else "N/A"

    telemetry_str = f"Sensor Telemetry ({s_src}): Temp: {t_str} | Hum: {h_str} | Gas: {g_str}"
    cv2.putText(
        frame,
        telemetry_str,
        (px1 + 20, py1 + 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 220, 220),
        1,
    )

    # 4. Reasoning Summary (Truncate to fit)
    reasoning = getattr(analysis, "reasoning_summary", "")
    if len(reasoning) > 65:
        reasoning = reasoning[:62] + "..."
    cv2.putText(
        frame,
        f"Reasoning: {reasoning}",
        (px1 + 20, py1 + 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (190, 190, 190),
        1,
    )

    # 5. Recommendation (First one)
    recs = getattr(analysis, "recommendations", [])
    rec_text = recs[0] if recs else "Keep stored in cool, dry conditions."
    if len(rec_text) > 65:
        rec_text = rec_text[:62] + "..."
    cv2.putText(
        frame,
        f"Action: {rec_text}",
        (px1 + 20, py1 + 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 215, 255),
        1,
    )

    # Divider
    cv2.line(frame, (px1 + 20, py1 + 205), (px2 - 20, py1 + 205), (70, 70, 70), 1)

    # 6. Action Footer
    footer = "Press [SPACE] or [R] to Scan Next Item  |  [ESC] Exit"
    cv2.putText(
        frame,
        footer,
        (px1 + 20, py1 + 235),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 255, 0),
        2,
    )


def draw_sensor_bar(frame, sensor_reading: Dict[str, Any]):
    """
    Draw bottom sensor telemetry bar.
    """
    h, w, _ = frame.shape
    bar_height = 32
    cv2.rectangle(frame, (0, h - bar_height), (w, h), (25, 25, 25), cv2.FILLED)

    temp = sensor_reading.get("temperature_c")
    hum = sensor_reading.get("humidity_percent")
    gas = sensor_reading.get("gas_value")
    status = sensor_reading.get("sensor_status", "unavailable")
    source = sensor_reading.get("source", "none")

    temp_str = f"{temp:.1f}C" if temp is not None else "--"
    hum_str = f"{hum:.1f}%" if hum is not None else "--"
    gas_str = f"{gas:.0f}" if gas is not None else "--"

    status_color = (0, 220, 0) if status == "valid" else (0, 165, 255) if status == "stale" else (0, 0, 220)

    bar_text = f"SENSORS [{source.upper()}]: Temp: {temp_str} | Hum: {hum_str} | Gas: {gas_str} | Status: {status.upper()}"
    cv2.putText(
        frame,
        bar_text,
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        status_color,
        1,
    )

    # Top-right disclaimer
    disclaimer = "Visual & Environmental Freshness Assessment"
    cv2.putText(
        frame,
        disclaimer,
        (w - 330, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (160, 160, 160),
        1,
    )


def draw_overlay(frame, tracked_objects: List[Any], sensor_reading: Dict[str, Any]):
    """
    Legacy wrapper for full overlay.
    """
    detections = [
        {"class_name": t.class_name, "confidence": t.confidence, "bbox": t.bbox}
        for t in tracked_objects
    ]
    draw_food_boxes(frame, detections)
    draw_sensor_bar(frame, sensor_reading)
    return frame
