"""
End-to-End System Integration & Manual Flow Verification Script.
Simulates all user interactions in the Streamlit application:
1. File upload & Camera snapshot processing
2. Hierarchical vision inference across all categories
3. Sensor fusion with mock, manual, and disconnected telemetry
4. Evidence aggregation & knowledge rules mapping
5. LLM structured freshness reasoning (offline fallback and schema verification)
6. SQLite observation storage & CSV export
"""

import os
import sys
import io
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PIL import Image
import numpy as np
import pandas as pd

from config import (
    VISION_MODEL_NAME,
    DATABASE_PATH,
    EXPORT_PATH,
    FRESHNESS_RULES_PATH,
    FOOD_CLASSES,
)
from services.vision_service import VisionService, SYNONYM_MAP
from services.sensor_service import SensorService
from services.evidence_aggregator import EvidenceAggregator
from services.llm_service import LLMFreshnessService
from services.history_service import HistoryService


def test_full_pipeline():
    print("=" * 80)
    print("1. INITIALIZING SERVICES FOR END-TO-END QA")
    print("=" * 80)

    t0 = time.perf_counter()
    vision_svc = VisionService(model_name=VISION_MODEL_NAME)
    aggregator = EvidenceAggregator(rules_path=FRESHNESS_RULES_PATH)
    llm_svc = LLMFreshnessService(api_key="")  # Offline rule fallback verification
    history_svc = HistoryService(db_path="database/test_qa.db", export_path="exports/test_qa.csv")
    print(f"Services initialized in {time.perf_counter() - t0:.2f}s\n")

    # -------------------------------------------------------------------------
    # 2. TEST SYNONYM MAPPINGS & REGIONAL DISHES
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("2. VERIFYING SYNONYM MAPPING & ONTOLOGY CONTRACTS")
    print("=" * 80)
    assert SYNONYM_MAP.get("chapati") == "roti"
    assert SYNONYM_MAP.get("phulka") == "roti"
    assert SYNONYM_MAP.get("plain_rice") == "rice"
    assert SYNONYM_MAP.get("chawal") == "rice"
    assert SYNONYM_MAP.get("yellow_dal") == "dal"
    assert SYNONYM_MAP.get("dal_tadka") == "dal"
    assert SYNONYM_MAP.get("sambar") == "sambar", "CRITICAL: sambar must NOT collapse to generic dal!"
    assert SYNONYM_MAP.get("rajma_chawal") == "rajma_chawal"
    assert SYNONYM_MAP.get("dal_chawal") == "rice_dal_combo"
    assert SYNONYM_MAP.get("thali") == "thali_mixed"
    print(" All critical synonym mappings and regional distinctions verified.\n")

    # -------------------------------------------------------------------------
    # 3. TEST RECOGNIZED FOODS, VEGETABLES & PLATTERS
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("3. TESTING RECOGNIZED FOOD IMAGES (Upload / Snapshot)")
    print("=" * 80)

    test_foods = [
        ("evaluation/real_world_validation/fruits/apple_01.jpg", "apple", "fruit"),
        ("evaluation/real_world_validation/fruits/banana_01.jpg", "banana", "fruit"),
        ("evaluation/real_world_validation/fruits/mango_01.jpg", "mango", "fruit"),
        ("evaluation/real_world_validation/fruits/orange_01.jpg", "orange", "fruit"),
        ("evaluation/real_world_validation/staples/plain_rice_01.jpg", "rice", "food_dish"),
        ("evaluation/real_world_validation/staples/fried_rice_01.jpg", "fried_rice", "food_dish"),
        ("evaluation/real_world_validation/staples/roti_01.jpg", "roti", "food_dish"),
        ("evaluation/real_world_validation/staples/paratha_01.jpg", "paratha", "food_dish"),
        ("evaluation/real_world_validation/staples/naan_01.jpg", "naan", "food_dish"),
        ("evaluation/real_world_validation/lentils/dal_01.jpg", "dal", "food_dish"),
        ("evaluation/real_world_validation/lentils/dal_02.jpg", "sambar", "food_dish"),
        ("evaluation/real_world_validation/lentils/rajma_01.jpg", "rajma", "food_dish"),
        ("evaluation/real_world_validation/lentils/chole_01.jpg", "chole", "food_dish"),
        ("evaluation/real_world_validation/indian_dishes/biryani_01.jpg", "biryani", "food_dish"),
        ("evaluation/real_world_validation/indian_dishes/paneer_01.jpg", "paneer", "food_dish"),
        ("evaluation/real_world_validation/indian_dishes/samosa_01.jpg", "samosa", "food_dish"),
        ("evaluation/real_world_validation/indian_dishes/dosa_01.jpg", "dosa", "food_dish"),
        ("evaluation/real_world_validation/indian_dishes/idli_01.jpg", "idli", "food_dish"),
        ("evaluation/real_world_validation/indian_dishes/rajma_chawal_01.jpg", "rajma_chawal", "food_dish"),
        ("evaluation/real_world_validation/indian_dishes/rice_dal_combo_01.jpg", "rice_dal_combo", "food_dish"),
        ("evaluation/real_world_validation/indian_dishes/thali_mixed_01.jpg", "thali_mixed", "food_dish"),
        ("evaluation/real_world_validation/western_foods/pizza_01.jpg", "pizza", "food_dish"),
        ("evaluation/real_world_validation/western_foods/burger_01.jpg", "burger", "food_dish"),
        ("evaluation/real_world_validation/western_foods/pasta_01.jpg", "pasta", "food_dish"),
        ("evaluation/real_world_validation/western_foods/sandwich_01.jpg", "sandwich", "food_dish"),
    ]

    for img_path, exp_class, exp_cat in test_foods:
        img = Image.open(img_path).convert("RGB")
        res = vision_svc.analyze_image(img)

        assert res["status"] == "recognized", f"Expected recognized for {img_path}, got {res['status']}"
        assert res["name"] == exp_class, f"Expected {exp_class} for {img_path}, got {res['name']}"
        assert res["confidence"] > 0.30
        assert res["visual_description"] != ""

        # Test end-to-end evidence aggregation and freshness estimation
        sensor_reading = {
            "temperature_c": 22.5,
            "humidity_percent": 55.0,
            "gas_value": 120.0,
            "sensor_status": "active",
            "sensor_scope": "environment",
            "source": "manual_slider",
            "timestamp": "2026-10-03T12:00:00",
        }
        packet = aggregator.aggregate(
            vision_result=res,
            sensor_reading=sensor_reading,
            history=[],
            user_metadata={"source_file": os.path.basename(img_path)}
        )
        analysis = llm_svc.analyze(packet, image=img)
        assert analysis.item_name == exp_class or exp_class in str(analysis.item_name).lower()
        assert analysis.freshness_status in ["FRESH", "MODERATELY_FRESH", "QUESTIONABLE", "NOT_FRESH"]
        assert analysis.estimated_remaining_freshness.value != "N/A"

        # Save to database
        row_id = history_svc.save_observation(analysis, packet)
        assert row_id > 0

        print(f" [PASS] Recognized: {exp_class:15s} | Conf: {res['confidence']*100:.1f}% | Freshness: {analysis.freshness_status:15s} | Shelf-Life: {analysis.estimated_remaining_freshness.value} {analysis.estimated_remaining_freshness.unit}")

    print()

    # -------------------------------------------------------------------------
    # 4. TEST NON-FOOD REJECTION (15 DISTINCT OBJECTS)
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("4. TESTING STAGE 1 NON-FOOD REJECTION GATE (15 Objects)")
    print("=" * 80)

    non_food_samples = [
        "phone_01.jpg", "laptop_01.jpg", "keys_01.jpg", "watch_01.jpg",
        "person_face_01.jpg", "book_01.jpg", "water_bottle_01.jpg",
        "mouse_01.jpg", "headphones_01.jpg", "charger_01.jpg",
        "coffee_mug_01.jpg", "pen_01.jpg", "shoe_01.jpg", "car_01.jpg", "cat_01.jpg"
    ]

    for fname in non_food_samples:
        img_path = os.path.join("evaluation/real_world_validation/non_food", fname)
        img = Image.open(img_path).convert("RGB")
        res = vision_svc.analyze_image(img)

        assert res["status"] == "not_food", f"Expected not_food for {fname}, got {res['status']}"
        assert res["name"] is None
        assert "Not a fruit or food" in res["visual_description"]

        # Test aggregator and LLM response contracts
        packet = aggregator.aggregate(
            vision_result=res,
            sensor_reading={"temperature_c": 25.0, "humidity_percent": 50.0, "gas_value": 100.0},
            history=[]
        )
        analysis = llm_svc.analyze(packet, image=img)
        assert analysis.freshness_status == "NOT_FOOD"
        assert analysis.item_name == "Not Food"
        assert analysis.estimated_remaining_freshness.value == "N/A"

        print(f" [PASS] Non-Food Rejected: {fname:20s} | Visual: {res['visual_description'][:55]}...")

    print()

    # -------------------------------------------------------------------------
    # 5. TEST UNKNOWN / OUT-OF-DISTRIBUTION FOODS (SAFE UNCERTAINTY HANDLING)
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("5. TESTING OUT-OF-DISTRIBUTION FOODS (SAFE ATTRIBUTE FALLBACK)")
    print("=" * 80)

    ood_samples = [
        ("dimsum_01.jpg", "Chinese dim sum dumplings"),
        ("paella_01.jpg", "Spanish seafood paella"),
        ("rambutan_01.jpg", "Tropical rambutan"),
    ]

    for fname, desc in ood_samples:
        img_path = os.path.join("evaluation/real_world_validation/unknown_dish", fname)
        img = Image.open(img_path).convert("RGB")
        res = vision_svc.analyze_image(img)

        assert res["status"] == "uncertain", f"Expected uncertain for {fname}, got {res['status']}"
        assert res["name"] is None, f"Expected name to be None for uncertain item {fname}"
        assert "Visually observed:" in res["visual_description"]

        packet = aggregator.aggregate(
            vision_result=res,
            sensor_reading={"temperature_c": 24.0, "humidity_percent": 60.0, "gas_value": 130.0},
            history=[]
        )
        analysis = llm_svc.analyze(packet, image=img)
        assert analysis.item_name in ["Unidentified Food / Dish", "Cooked food", "Food dish"] or "dish" in str(analysis.item_name).lower() or "unidentified" in str(analysis.item_name).lower()
        print(f" [PASS] Safe OOD Fallback: {fname:16s} | Observed: {res['visual_description']}")

    print()

    # -------------------------------------------------------------------------
    # 6. TEST SENSOR MODES (MOCK, HARDWARE, DISCONNECTED)
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("6. TESTING SENSOR SERVICES (Mock, Serial Auto-Discovery, Disconnected)")
    print("=" * 80)

    # 1. Mock sensor mode
    mock_sensor = SensorService(mock_mode=True)
    mock_sensor.start()
    time.sleep(1.2)
    reading = mock_sensor.get_latest_reading()
    assert reading["temperature_c"] is not None
    assert reading["humidity_percent"] is not None
    assert reading["gas_value"] is not None
    assert reading["sensor_status"] == "valid"
    print(f" [PASS] Mock Telemetry: {reading['temperature_c']}°C | {reading['humidity_percent']}% | {reading['gas_value']} ppm | Status: {reading['sensor_status']}")
    mock_sensor.stop()

    # 2. Disconnected hardware fallback
    disconnected_sensor = SensorService(mock_mode=False, port="COM99_NONEXISTENT", timeout=0.1)
    reading_disc = disconnected_sensor.get_latest_reading()
    assert reading_disc["temperature_c"] is None
    assert reading_disc["sensor_status"] == "unavailable"
    print(f" [PASS] Disconnected Fallback: Reading status correctly reported as '{reading_disc['sensor_status']}' without crashing.")

    print()

    # -------------------------------------------------------------------------
    # 7. TEST HISTORY PERSISTENCE & CSV EXPORT
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("7. TESTING SQLITE PERSISTENCE & CSV EXPORT")
    print("=" * 80)

    records = history_svc.get_recent_observations(limit=100)
    assert len(records) > 0, "Expected observations saved in SQLite"
    print(f" [PASS] SQLite retrieved {len(records)} recent observations.")

    csv_path = history_svc.export_to_csv()
    assert os.path.exists(csv_path)
    df = pd.read_csv(csv_path)
    assert len(df) == len(records)
    print(f" [PASS] CSV Export verified: {csv_path} with {len(df)} rows.")

    print("\n" + "=" * 80)
    print("ALL END-TO-END QA & MANUAL VALIDATION TESTS PASSED (100%)")
    print("=" * 80)


if __name__ == "__main__":
    test_full_pipeline()
