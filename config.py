"""
Configuration for Fruit & Food Freshness Detection and Analysis System.

All configurable paths, thresholds, and settings in one place.
Secrets (API keys, COM ports) are loaded from .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# MULTIMODAL VISION MODEL & PIPELINE
# =============================================================================
# Primary Multimodal Vision Backend: Google Gemini Multimodal Vision (Free-Tier)
VISION_PRIMARY_BACKEND = os.getenv("VISION_PRIMARY_BACKEND", "gemini")  # "gemini", "multimodal_openai", or "siglip"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))
GEMINI_MODEL = GEMINI_VISION_MODEL

# Optional Secondary Provider: OpenAI Vision Model
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Fallback Vision Model: Google SigLIP Zero-Shot Vision-Language Classifier
VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", "google/siglip-base-patch16-224")

# Confidence & Gating Thresholds
GATE_NON_FOOD_THRESHOLD = 0.55   # Score above which non-food classes trigger rejection
GATE_SPECIFIC_THRESHOLD = 0.40   # Specific non-food subcategory threshold (electronics, faces, furniture)
CONFIDENCE_THRESHOLD = 0.40      # Minimum confidence for recognized classification
UNCERTAIN_PROB_THRESHOLD = 0.12  # Below this probability among target classes -> uncertain
UNCERTAIN_MARGIN_THRESHOLD = 0.04 # Top-1 minus Top-2 margin threshold for ambiguity

# Legacy compatibility
YOLO_MODEL = "yolov8s-worldv2.pt"
AUTO_LOCK_FOOD = False
REQUIRE_TRIGGER_MEASUREMENT = True

# Target Food & Fruit Classes
FOOD_CLASSES = [
    # Fruits
    "apple", "banana", "orange", "mango", "strawberry", "watermelon",
    "grapes", "pomegranate", "pineapple", "lemon", "papaya", "guava",

    # Vegetables & Greens
    "tomato", "potato", "onion", "carrot", "cucumber", "spinach",
    "broccoli", "mushroom", "okra", "eggplant", "cauliflower",
    "bell_pepper", "ginger", "garlic", "green_chili",

    # Staples & Breads
    "rice", "fried_rice", "roti", "paratha", "naan",

    # Lentils & Beans
    "dal", "rajma", "chole", "dal_makhani", "sambar",

    # Regional & Platter Dishes
    "biryani", "paneer", "palak_paneer", "aloo_gobi", "samosa",
    "dosa", "idli", "vada", "rajma_chawal", "rice_dal_combo",
    "chole_bhature", "pav_bhaji", "khichdi", "poha", "upma",
    "thali_mixed", "curry",

    # Global & Western Foods
    "pizza", "burger", "pasta", "sandwich", "soup", "salad",
]

# Web App Configuration
WEB_HOST = os.getenv("WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.getenv("PORT", os.getenv("WEB_PORT", "8501")))

# =============================================================================
# CAMERA
# =============================================================================
CAMERA_SOURCE = 0  # 0 = default webcam, or RTSP/file path

# =============================================================================
# ARDUINO AUTO-DETECTION & AUTO-FLASH
# =============================================================================
AUTO_DETECT_SENSORS = True  # Automatically detect real Arduino vs fallback to Mock
AUTO_FLASH_ARDUINO = True   # Auto-compile and flash Arduino sketch when connected
ARDUINO_CLI_PATH = os.getenv("ARDUINO_CLI_PATH", "auto")  # "auto" or path to arduino-cli.exe
ARDUINO_SKETCH_PATH = "arduino/sensor_reader"
DEFAULT_FQBN = os.getenv("DEFAULT_FQBN", "arduino:avr:uno")  # Fallback if board FQBN is unspecified

# =============================================================================
# SENSOR / SERIAL
# =============================================================================
SERIAL_PORT = os.getenv("SERIAL_PORT", "auto")  # "auto", "COM10", "/dev/ttyUSB0"
BAUD_RATE = int(os.getenv("BAUD_RATE", "9600"))
SENSOR_TIMEOUT = 2  # seconds to wait for serial read
SENSOR_STALE_SECONDS = 30  # mark readings older than this as stale
SENSOR_READ_INTERVAL = 1.0  # seconds between sensor reads

# =============================================================================
# LLM & REASONING
# =============================================================================
LLM_MODEL = os.getenv("LLM_MODEL", GEMINI_VISION_MODEL)
LLM_TIMEOUT = 30  # seconds
LLM_ANALYSIS_INTERVAL = 10  # minimum seconds between LLM calls per object

# =============================================================================
# DATABASE
# =============================================================================
DATABASE_PATH = "database/freshness.db"

# =============================================================================
# EXPORT
# =============================================================================
EXPORT_PATH = "exports/analysis.csv"

# =============================================================================
# KNOWLEDGE
# =============================================================================
FRESHNESS_RULES_PATH = "knowledge/freshness_rules.json"

# =============================================================================
# LOGGING
# =============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
