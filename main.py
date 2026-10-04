"""
Fruit & Food Freshness Detection and Multimodal Analysis System.

Main application entry point — fully integrated, one-command execution.

Workflow Architecture:
  1. SCANNING: Live camera streams, YOLO-World detects FOOD items only with vibrant GREEN boxes.
               If no food is present, displays a clean searching HUD.
  2. LOCK & PROMPT: When food is framed, the camera pauses and locks the food item.
                    Displays on-screen banner & sends LCD command to Arduino:
                    "Place [FOOD] near sensor, then press [SPACE] or button to measure"
  3. TRIGGERED MEASURE: User positions food at sensor and triggers reading.
                        Captures instantaneous telemetry snapshot (eliminating empty-air readings).
  4. REASONING & RESULT: Aggregates multimodal evidence, computes freshness & shelf-life,
                         displays structured result card on-screen and on Arduino LCD.
  5. NEXT ITEM / RESET: User presses [SPACE] or [R] to scan the next item.

Usage:
  # Run live webcam pipeline:
  python main.py

  # Run on a static image:
  python main.py --mode image --image-path data/samples/fruits.jpg

  # Preserve pre-existing Arduino code without auto-flashing:
  python main.py --no-auto-flash
"""

import sys
import os
import time
import argparse
import logging
import cv2

import config
from camera.webcam import WebcamSource
from camera.image_loader import ImageSource
from services.detection_service import DetectionService
from services.hardware_manager import HardwareManager
from services.evidence_aggregator import EvidenceAggregator
from services.llm_service import LLMFreshnessService
from services.history_service import HistoryService
from utils.display import (
    draw_food_boxes,
    draw_searching_hud,
    draw_placement_prompt,
    draw_measuring_hud,
    draw_analysis_card,
    draw_sensor_bar,
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format="[%(levelname)s] %(asctime)s - %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Main")


def parse_args():
    parser = argparse.ArgumentParser(description="Fruit & Food Freshness Analysis System")
    parser.add_argument("--mode", choices=["webcam", "image"], default="webcam", help="Input mode: 'webcam' or 'image'")
    parser.add_argument("--image-path", type=str, default="", help="Path to input image (for --mode image)")
    parser.add_argument("--camera", type=int, default=config.CAMERA_SOURCE, help="Camera index")
    parser.add_argument("--no-auto-flash", action="store_true", help="Skip automatic firmware flashing on Arduino")
    return parser.parse_args()


def run_image_mode(args, detection_service, sensor_service, aggregator, llm_service, history_service):
    """Process a single image file with sequential prompt and analysis."""
    image_path = args.image_path
    if not image_path:
        logger.error("Please provide an image path using --image-path <path>")
        return

    loader = ImageSource()
    frame = loader.load(image_path)
    if frame is None:
        return

    logger.info("Running food detection on image: %s", image_path)
    all_detections = detection_service.detect(frame)
    # Filter strictly by confidence threshold
    food_detections = [d for d in all_detections if d.get("confidence", 0.0) >= config.CONFIDENCE_THRESHOLD]

    if not food_detections:
        logger.warning("No food items detected in image %s", image_path)
        display_frame = frame.copy()
        draw_searching_hud(display_frame)
        sensor_reading = sensor_service.get_latest_reading()
        draw_sensor_bar(display_frame, sensor_reading)
        cv2.imshow("Fruit & Food Freshness - Image Mode", display_frame)
        print("\nNo food detected in image. Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    logger.info("Found %d food item(s) in image", len(food_detections))
    top_food = max(food_detections, key=lambda x: x.get("confidence", 0.0))

    # 1. Send LCD notification
    sensor_service.send_lcd_message(f"Detected: {top_food['class_name']}", "Analyzing...")

    # 2. Capture instantaneous sensor snapshot
    sensor_reading = sensor_service.capture_snapshot()

    # 3. Aggregate multimodal evidence
    history = history_service.get_recent_history(item_name=top_food["class_name"], limit=3)
    evidence = aggregator.aggregate(
        detection=top_food,
        sensor_reading=sensor_reading,
        history=history,
        user_metadata={"source_file": image_path},
    )

    # 4. Reason freshness
    logger.info("Analyzing freshness for '%s'...", top_food["class_name"])
    analysis = llm_service.analyze(evidence)
    history_service.save_observation(analysis, evidence)

    # 5. Send results to Arduino LCD
    est_rf = getattr(analysis, "estimated_remaining_freshness", None)
    rf_val = getattr(est_rf, "value", "N/A") if est_rf else "N/A"
    rf_unit = getattr(est_rf, "unit", "") if est_rf else ""
    sensor_service.send_lcd_message(
        f"{top_food['class_name'][:8]}:{analysis.freshness_status[:7]}",
        f"Shelf:{rf_val} {rf_unit}"[:16],
    )

    print("\n" + "=" * 60)
    print(f"ITEM: {analysis.item_name.upper()} (Confidence: {analysis.detection_confidence*100:.1f}%)")
    print(f"FRESHNESS STATUS: {analysis.freshness_status}")
    print(f"ESTIMATED SHELF LIFE: {analysis.estimated_remaining_freshness.range_description}")
    print(f"REASONING: {analysis.reasoning_summary}")
    print(f"RECOMMENDATIONS: {', '.join(analysis.recommendations)}")
    print("=" * 60 + "\n")

    # 6. Render overlay
    display_frame = frame.copy()
    draw_food_boxes(display_frame, [top_food])
    draw_analysis_card(display_frame, analysis, sensor_reading)
    draw_sensor_bar(display_frame, sensor_reading)

    cv2.imshow("Fruit & Food Freshness Analysis - Image Mode", display_frame)
    print("Press any key in the window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_webcam_mode(args, detection_service, sensor_service, aggregator, llm_service, history_service):
    """
    Run interactive sequential state-machine webcam analysis:
      STATE_SCANNING -> STATE_LOCKED -> STATE_MEASURING -> STATE_RESULT -> STATE_SCANNING
    """
    STATE_SCANNING = "SCANNING"
    STATE_LOCKED = "LOCKED"
    STATE_MEASURING = "MEASURING"
    STATE_RESULT = "RESULT"

    current_state = STATE_SCANNING
    locked_food = None
    locked_frame = None
    current_analysis = None
    triggered_sensor_data = None

    webcam = WebcamSource(source=args.camera)
    if not webcam.open():
        logger.error("Could not start webcam (camera index %d). Exiting.", args.camera)
        return

    # Notify LCD on startup
    sensor_service.send_lcd_message("Freshness System", "Ready to scan")

    print("\n" + "=" * 60)
    print("  INTERACTIVE FOOD FRESHNESS PIPELINE ACTIVE")
    print("=" * 60)
    print("  1. Point camera at food/fruit to detect (Green Box).")
    print("  2. When food is in view, press [SPACE] to lock.")
    print("  3. Place food near sensors and press [SPACE] or Arduino Button to measure.")
    print("  4. Press [R] to re-scan / cancel.")
    print("  5. Press [ESC] in camera window to exit and export CSV.")
    print("=" * 60 + "\n")

    try:
        while True:
            # -----------------------------------------------------------------
            # STATE 1: SCANNING (Live Camera Feed & Food Detection)
            # -----------------------------------------------------------------
            if current_state == STATE_SCANNING:
                ret, frame = webcam.read_frame()
                if not ret or frame is None:
                    time.sleep(0.05)
                    continue

                display_frame = frame.copy()
                sensor_reading = sensor_service.get_latest_reading()

                # Run food-only detection
                detections = detection_service.detect(frame)
                food_detections = [
                    d for d in detections
                    if d.get("confidence", 0.0) >= config.CONFIDENCE_THRESHOLD
                ]

                if food_detections:
                    # Draw vibrant green boxes around detected food
                    draw_food_boxes(display_frame, food_detections)
                    top_food = max(food_detections, key=lambda x: x.get("confidence", 0.0))

                    # Auto-lock if enabled, or wait for user SPACE key
                    if config.AUTO_LOCK_FOOD:
                        locked_food = top_food
                        locked_frame = frame.copy()
                        current_state = STATE_LOCKED
                        sensor_service.send_lcd_message(f"Detected: {locked_food['class_name']}", "Place at sensor")
                        logger.info("Auto-locked food: '%s' (conf: %.2f)", locked_food['class_name'], locked_food['confidence'])
                else:
                    # No food in view -> show searching HUD, no bounding boxes
                    draw_searching_hud(display_frame)

                draw_sensor_bar(display_frame, sensor_reading)
                cv2.imshow("Fruit & Food Freshness System", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    logger.info("ESC pressed. Exiting...")
                    break
                elif key in (32, 13) and food_detections:  # SPACE or ENTER
                    top_food = max(food_detections, key=lambda x: x.get("confidence", 0.0))
                    locked_food = top_food
                    locked_frame = frame.copy()
                    current_state = STATE_LOCKED
                    sensor_service.send_lcd_message(f"Detected: {locked_food['class_name']}", "Place at sensor")
                    logger.info("Food locked: '%s' (conf: %.2f). Place at sensor and press SPACE.", locked_food['class_name'], locked_food['confidence'])

            # -----------------------------------------------------------------
            # STATE 2: FOOD LOCKED - WAITING FOR PLACEMENT AT SENSOR
            # -----------------------------------------------------------------
            elif current_state == STATE_LOCKED:
                display_frame = locked_frame.copy() if locked_frame is not None else frame.copy()
                sensor_reading = sensor_service.get_latest_reading()

                # Draw green box on locked item
                if locked_food:
                    draw_food_boxes(display_frame, [locked_food])
                    # Draw placement instruction prompt
                    draw_placement_prompt(display_frame, locked_food["class_name"], locked_food["confidence"])

                draw_sensor_bar(display_frame, sensor_reading)
                cv2.imshow("Fruit & Food Freshness System", display_frame)

                # Check for Arduino hardware button trigger OR keyboard press
                hardware_button = sensor_service.check_button_pressed()
                key = cv2.waitKey(20) & 0xFF

                if key == 27:  # ESC
                    break
                elif key in (ord('r'), ord('R')):  # Reset / Re-scan
                    logger.info("Re-scan requested. Returning to SCANNING state.")
                    locked_food = None
                    locked_frame = None
                    current_state = STATE_SCANNING
                    sensor_service.send_lcd_message("Freshness System", "Ready to scan")
                elif key in (32, 13) or hardware_button:  # Trigger measurement
                    logger.info("Trigger received! Capturing sensor snapshot for '%s'...", locked_food["class_name"])
                    current_state = STATE_MEASURING

            # -----------------------------------------------------------------
            # STATE 3: MEASURING SENSORS & REASONING FRESHNESS
            # -----------------------------------------------------------------
            elif current_state == STATE_MEASURING:
                display_frame = locked_frame.copy() if locked_frame is not None else frame.copy()
                if locked_food:
                    draw_food_boxes(display_frame, [locked_food])
                    draw_measuring_hud(display_frame, locked_food["class_name"])
                cv2.imshow("Fruit & Food Freshness System", display_frame)
                cv2.waitKey(1)

                sensor_service.send_lcd_message("Measuring...", locked_food["class_name"])

                # 1. Capture snapshot of sensors at exact placement moment
                triggered_sensor_data = sensor_service.capture_snapshot()

                # 2. Fetch history & aggregate evidence
                history = history_service.get_recent_history(item_name=locked_food["class_name"], limit=3)
                evidence = aggregator.aggregate(
                    detection=locked_food,
                    sensor_reading=triggered_sensor_data,
                    history=history,
                )

                # 3. Analyze freshness
                logger.info("Executing multimodal freshness reasoning for '%s'...", locked_food["class_name"])
                current_analysis = llm_service.analyze(evidence)
                history_service.save_observation(current_analysis, evidence)

                # 4. Transmit result to Arduino LCD
                est_rf = getattr(current_analysis, "estimated_remaining_freshness", None)
                rf_val = getattr(est_rf, "value", "N/A") if est_rf else "N/A"
                rf_unit = getattr(est_rf, "unit", "") if est_rf else ""
                sensor_service.send_lcd_message(
                    f"{locked_food['class_name'][:8]}:{current_analysis.freshness_status[:7]}",
                    f"Shelf:{rf_val} {rf_unit}"[:16],
                )

                print("\n" + "=" * 60)
                print(f"ANALYSIS COMPLETE: {current_analysis.item_name.upper()}")
                print(f"FRESHNESS STATUS: {current_analysis.freshness_status}")
                print(f"ESTIMATED SHELF LIFE: {current_analysis.estimated_remaining_freshness.range_description}")
                print(f"ENVIRONMENT: Temp={triggered_sensor_data.get('temperature_c')}C, Hum={triggered_sensor_data.get('humidity_percent')}%, Gas={triggered_sensor_data.get('gas_value')}")
                print(f"REASONING: {current_analysis.reasoning_summary}")
                print(f"RECOMMENDATIONS: {', '.join(current_analysis.recommendations)}")
                print("=" * 60 + "\n")

                current_state = STATE_RESULT

            # -----------------------------------------------------------------
            # STATE 4: RESULT DISPLAY (Inspection & Next Item Prompt)
            # -----------------------------------------------------------------
            elif current_state == STATE_RESULT:
                display_frame = locked_frame.copy() if locked_frame is not None else frame.copy()
                if locked_food:
                    draw_food_boxes(display_frame, [locked_food])
                if current_analysis and triggered_sensor_data:
                    draw_analysis_card(display_frame, current_analysis, triggered_sensor_data)

                draw_sensor_bar(display_frame, triggered_sensor_data or {})
                cv2.imshow("Fruit & Food Freshness System", display_frame)

                key = cv2.waitKey(30) & 0xFF
                if key == 27:  # ESC
                    break
                elif key in (32, 13, ord('r'), ord('R')):  # Scan next item
                    logger.info("Scan next item requested. Resetting to SCANNING state.")
                    locked_food = None
                    locked_frame = None
                    current_analysis = None
                    triggered_sensor_data = None
                    current_state = STATE_SCANNING
                    sensor_service.send_lcd_message("Freshness System", "Ready to scan")

    finally:
        webcam.release()
        cv2.destroyAllWindows()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  FRUIT & FOOD FRESHNESS DETECTION AND MULTIMODAL SYSTEM")
    print("=" * 60)
    print(f"Mode:            {args.mode.upper()}")
    print(f"LLM Model:       {config.LLM_MODEL}")
    print(f"Database:        {config.DATABASE_PATH}")
    print("=" * 60 + "\n")

    # 1. Initialize Hardware Manager (auto-detects Arduino, auto-flashes sketch, falls back to Mock)
    auto_flash = not args.no_auto_flash and config.AUTO_FLASH_ARDUINO
    hw_manager = HardwareManager(
        cli_path=config.ARDUINO_CLI_PATH,
        sketch_path=config.ARDUINO_SKETCH_PATH,
        auto_flash=auto_flash,
        default_fqbn=config.DEFAULT_FQBN,
        baud_rate=config.BAUD_RATE,
    )
    sensor_service = hw_manager.initialize()

    # 2. Initialize AI & Data Services
    logger.info("Initializing YOLO-World and AI services...")
    detection_service = DetectionService(
        model_path=config.YOLO_MODEL,
        classes=config.FOOD_CLASSES,
        conf=config.CONFIDENCE_THRESHOLD,
    )
    aggregator = EvidenceAggregator(rules_path=config.FRESHNESS_RULES_PATH)
    llm_service = LLMFreshnessService(
        api_key=config.OPENAI_API_KEY,
        model=config.LLM_MODEL,
        timeout=config.LLM_TIMEOUT,
    )
    history_service = HistoryService(db_path=config.DATABASE_PATH)

    try:
        if args.mode == "image":
            run_image_mode(args, detection_service, sensor_service, aggregator, llm_service, history_service)
        else:
            run_webcam_mode(args, detection_service, sensor_service, aggregator, llm_service, history_service)
    finally:
        logger.info("Shutting down hardware and services...")
        sensor_service.send_lcd_message("Freshness System", "System Offline")
        hw_manager.shutdown()
        history_service.export_csv(config.EXPORT_PATH)
        logger.info("System closed successfully.")


if __name__ == "__main__":
    main()
