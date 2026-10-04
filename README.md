# Fruit & Food Freshness Detection System

An integrated multimodal freshness assessment system combining computer vision, environmental sensor telemetry, food science reference rules, and conversational reasoning to estimate the condition, freshness status, and shelf life of fruits, vegetables, and food items.

---

## Table of Contents

- [Overview & Problem Statement](#overview--problem-statement)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Multimodal Vision Pipeline](#multimodal-vision-pipeline)
  - [Primary Vision: Google Gemini](#primary-vision-google-gemini)
  - [Secondary Vision: OpenAI Vision](#secondary-vision-openai-vision)
  - [Local Offline Fallback: Google SigLIP](#local-offline-fallback-google-siglip)
- [Condition & Freshness Assessment](#condition--freshness-assessment)
- [Hardware & Environmental Sensors](#hardware--environmental-sensors)
  - [Sensors & Pinout](#sensors--pinout)
  - [Arduino Firmware & Two-Way LCD Protocol](#arduino-firmware--two-way-lcd-protocol)
  - [Hardware Trigger & Telemetry Channels](#hardware-trigger--telemetry-channels)
- [User Interfaces](#user-interfaces)
  - [Streamlit Web Application](#streamlit-web-application)
  - [CLI & OpenCV Mode](#cli--opencv-mode)
- [History & Persistence](#history--persistence)
- [Installation & Setup](#installation--setup)
- [Environment Configuration](#environment-configuration)
- [Usage Guide](#usage-guide)
- [Repository Structure](#repository-structure)
- [Hardware Requirements](#hardware-requirements)
- [Limitations & Food Safety Disclaimer](#limitations--food-safety-disclaimer)
- [Future Enhancements](#future-enhancements)

---

## Overview & Problem Statement

Food spoilage and post-harvest losses represent significant economic waste and household challenges. Determining whether produce or prepared food remains fresh, edible, or spoiled often relies on subjective visual cues and ambient storage conditions that are difficult to track consistently.

This project implements a multi-stage, multi-sensor pipeline designed to:
1. **Identify** food and produce items accurately across diverse categories.
2. **Inspect** visual degradation indicators such as color shifts, browning, wrinkling, bruising, surface moisture, and mold growth.
3. **Capture** ambient environmental conditions (temperature, relative humidity, VOC/gas levels) via physical sensors or simulation.
4. **Synthesize** visual observations, sensor data, historical trends, and reference storage rules to provide calibrated freshness determinations, shelf-life estimates, and actionable storage guidance.

---

## Key Features

- **Hierarchical Vision Pipeline**: Primary cloud multimodal vision (Google Gemini) with automatic failover to secondary cloud vision (OpenAI) and local zero-dependency vision (Google SigLIP).
- **Fast-Fail Rate-Limit & Quota Handling**: Detects HTTP 429 `RESOURCE_EXHAUSTED` errors on turn 1 with 0 retries, extracting retry delays and immediately routing to local offline fallback vision without stalling the user.
- **Multi-Item Scene Analysis**: Capable of detecting and assessing multiple individual items in a single scene (up to 8 items) without collapsing distinct ingredients into generic categories.
- **Non-Food Rejection Gate**: Distinguishes food items from non-food background objects before performing condition analysis.
- **Environmental Sensor Telemetry**: Ingests real-time temperature, humidity, and volatile organic compound (VOC) gas levels from DHT11 and MQ-135 sensors.
- **Two-Way Hardware Synchronization**: Sends real-time status and freshness summaries to 16x2 I2C physical LCD screens and listens for hardware push-button measurement triggers.
- **Interactive Streamlit Web Dashboard**: Drag-and-drop image upload, live in-browser webcam capture, preloaded sample gallery, sensor sliders, HUD overlays, and contextual Q&A chat.
- **Local Persistence & Export**: Tracks historical scans in a local SQLite database (`database/freshness.db`) with one-click CSV export (`exports/analysis.csv`).

---

## System Architecture

```
                                SYSTEM ENTRYPOINTS
             ┌───────────────────────────┴───────────────────────────┐
             ▼                                                       ▼
  [Streamlit Web App: web_app.py]                           [CLI Mode: main.py]
  (Webcam / Image Upload / Gallery)                      (Live OpenCV Camera Stream)
             │                                                       │
             └───────────────────────────┬───────────────────────────┘
                                         │
                                         ▼
                               [HARDWARE MANAGER]
                         [services/hardware_manager.py]
                                         │
                       ┌─────────────────┴─────────────────┐
                       ▼                                   ▼
              [ARDUINO DETECTED]                  [NO ARDUINO FOUND]
              Auto-flash firmware or               Realistic mock telemetry or
              connect serial port                  interactive manual sliders
                       │                                   │
                       └─────────────────┬─────────────────┘
                                         │
                                         ▼
                        [STAGE 1: MULTIMODAL VISION]
                        [services/vision_service.py]
                                         │
                       ┌─────────────────┼─────────────────┐
                       ▼                 ▼                 ▼
                [PRIMARY VISION]  [SECONDARY VISION]  [OFFLINE LOCAL VISION]
                 Google Gemini      OpenAI Vision         Google SigLIP
                (gemini-3.8-flash)  (gpt-4o-mini)      (Zero-shot non-food gate,
                 Structured Output Structured Output   45+ food ontology,
                 0-retry on 429     (Optional API)     condition classifier)
                       └─────────────────┬─────────────────┘
                                         │
                                         ▼
                        [STAGE 2: SENSOR SNAPSHOT]
                        [services/sensor_service.py]
                        Captures DHT11 (Temp/Humidity) and MQ-135 (VOC/Gas)
                        Triggered via Spacebar, Web Button, or Arduino Pin 2
                                         │
                                         ▼
                        [STAGE 3: MULTIMODAL REASONING]
                        [services/evidence_aggregator.py & services/llm_service.py]
                        Fuses Visual + Sensor + Reference Rules + Scan History
                        Gemini / GPT-4o-mini / Deterministic Rule Engine
                                         │
                                         ▼
                        [STAGE 4: RESULT, PERSISTENCE & Q&A]
                        - Freshness badge & estimated shelf life
                        - Storage advice & safety considerations
                        - Arduino LCD 16x2 display output
                        - SQLite database logging & CSV export
                        - Interactive chat assistant with scan context
```

---

## Multimodal Vision Pipeline

The vision subsystem (`services/vision_service.py`) operates in a resilient tiered hierarchy:

### Primary Vision: Google Gemini
- **Model**: `gemini-3.8-flash` (via `google-genai` SDK).
- **Structured Schema**: Uses Pydantic models (`ImageVisionResult`, `ItemVisionResult`) for deterministic JSON parsing.
- **Attributes Analyzed**: Primary food class, confidence score, bounding boxes, freshness condition, visible defects (browning, wrinkling, bruising, mold colonies), surface moisture, and qualitative observations.
- **Quota & Error Handling**:
  - `429 RESOURCE_EXHAUSTED`: Fails fast on turn 1 with 0 retries, parses any `retry-after` header, and switches immediately to SigLIP local fallback with clear UI notification.
  - `503 Service Unavailable`: Bounded to at most 1 retry with exponential backoff before falling back.

### Secondary Vision: OpenAI Vision
- **Model**: `gpt-4o-mini` (via `openai` SDK).
- **Behavior**: Used if `OPENAI_API_KEY` is configured and Gemini is unavailable or unconfigured.

### Local Offline Fallback: Google SigLIP
- **Model**: `google/siglip-base-patch16-224` (via Hugging Face `transformers` & PyTorch).
- **Three-Stage Pipeline**:
  1. *Non-Food Gate*: Zero-shot semantic contrast between food and non-food background objects.
  2. *Ontology Classification*: 45+ class curated food and fruit ontology with synonym normalization.
  3. *Condition Assessment*: Zero-shot condition classification across visible state prompts with margin uncertainty calibration.

---

## Condition & Freshness Assessment

The system categorizes items into a six-level condition spectrum:

| Status | Meaning | Action / Shelf Life |
| :--- | :--- | :--- |
| `FRESH` | Peak quality, no visible degradation, optimal storage. | Full expected shelf life. Store appropriately. |
| `GOOD_TO_EAT_VISUALLY` | Normal ripeness or minor cosmetic blemishes; safe to consume. | Consume within normal window. |
| `QUESTIONABLE` | Early dehydration, minor softening, or elevated storage temperature. | Inspect closely; consume soon. |
| `POSSIBLE_SPOILAGE` | Significant wrinkling, discoloration, or elevated VOC levels. | Check smell/texture; do not consume if off. |
| `VISIBLE_SPOILAGE` | Visible mold colonies, severe browning, rot, or decomposition. | Discard immediately. Do not consume. |
| `CANNOT_DETERMINE` | Occluded, ambiguous visual cues, or insufficient sensor evidence. | Re-scan or perform manual inspection. |

---

## Hardware & Environmental Sensors

### Sensors & Pinout

| Sensor | Function | Arduino Uno Pin | Typical Range |
| :--- | :--- | :--- | :--- |
| **DHT11 / DHT22** | Ambient Temperature & Humidity | Digital Pin 4 | Temp: 0–50°C, Humidity: 20–90% RH |
| **MQ-135** | Air Quality / VOC / Ammonia / Alcohol | Analog Pin A0 | 0–1023 raw ADC (baseline ~50–200) |
| **Push Button** | Trigger measurement snapshot | Digital Pin 2 (Internal Pullup) | Pressed = LOW |
| **16x2 I2C LCD** | Visual status and shelf-life display | SDA (A4), SCL (A5) | 2 lines × 16 characters |

### Arduino Firmware & Two-Way LCD Protocol

The Arduino sketch (`arduino/sensor_reader/sensor_reader.ino`) streams continuous JSON telemetry:
```json
{"temp": 24.5, "humidity": 62.0, "gas": 145, "gas_delta": 2, "quality": "MODERATE", "button": false}
```

The Python host sends sanitized two-line LCD commands:
```text
LCD:Apple: FRESH   |Shelf: 3-5 days
```

### Hardware Trigger & Telemetry Channels

Telemetry can be provided via four distinct channels:
1. **Local USB Serial**: Auto-detected via `pyserial` on Windows (`COMx`) or Linux (`/dev/ttyUSBx`).
2. **Web Serial API**: Browser-based direct USB connection for client-side hardware access over HTTPS.
3. **Realistic Mock Telemetry**: Generates physics-grounded temperature, humidity, and gas fluctuations for development without hardware.
4. **Manual UI Sliders**: Direct control of sensor values in the Streamlit sidebar for scenario testing.

---

## User Interfaces

### Streamlit Web Application

Run with:
```bash
streamlit run web_app.py
```

Features:
- **Input Methods**: Webcam snapshot (`st.camera_input`), drag-and-drop file upload, and preloaded sample testing gallery.
- **HUD Overlay**: Dynamic green bounding box annotations with condition labels on detected items.
- **Freshness Card**: Color-coded status badge, estimated shelf-life range, confidence rating, sensor telemetry gauges, and storage recommendations.
- **Contextual Q&A Assistant**: In-dashboard chat interface to ask questions about the analyzed food item, recipes, or storage methods.
- **Historical Observations**: Searchable, filterable table of past scans with one-click CSV download.

### CLI & OpenCV Mode

Run with:
```bash
python main.py
```

Features:
- Live OpenCV camera window with real-time HUD overlays.
- Keyboard controls: `[SPACE]` to lock/measure, `[R]` to reset, `[Q]` to exit.
- Automatic hardware discovery and optional firmware auto-flashing (`--no-auto-flash` to skip).

---

## History & Persistence

All completed analyses are stored in a local SQLite database:
- **Database Location**: `database/freshness.db`
- **Tables**: `analyses` (timestamp, food name, condition, confidence, shelf life, temperature, humidity, gas reading, reasoning text, source provider).
- **Export**: Saved to `exports/analysis.csv` or exported directly from the Streamlit UI.

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10 to 3.12
- Git
- (Optional) Arduino IDE / `arduino-cli` if using physical hardware
- (Optional) Webcam / USB camera

### 2. Clone Repository
```bash
git clone https://github.com/Aabhaykumarr/fruit-food-freshness-detection.git
cd fruit-food-freshness-detection
```

### 3. Create & Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Environment Configuration

Create a `.env` file in the project root based on `.env.example`:

```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

Edit `.env` with your API configuration:

```env
# Primary Multimodal Vision & Reasoning (Google Gemini)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.8-flash

# Secondary Multimodal Vision (OpenAI - Optional)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_VISION_MODEL=gpt-4o-mini

# Vision Provider Preference ("gemini", "openai", or "local")
PRIMARY_VISION_PROVIDER=gemini

# Hardware Serial Configuration
SERIAL_PORT=COM3
SERIAL_BAUD=115200
MOCK_SENSORS=true

# Database & Storage
DATABASE_PATH=database/freshness.db
EXPORTS_PATH=exports/analysis.csv
```

> **Note**: If no API keys are provided, the system operates completely offline using the local Google SigLIP vision model and deterministic rule-based freshness engine.

---

## Usage Guide

### Starting the Web Interface
```bash
streamlit run web_app.py
```
Open your browser at `http://localhost:8501`.

### Running the CLI Interface
```bash
# With mock sensors (default if no board connected)
python main.py

# Skip Arduino auto-flashing
python main.py --no-auto-flash
```

### Running the Test Suite
```bash
pytest
```

---

## Repository Structure

```
fruit_detection/
├── .env.example                     # Environment template with safe placeholders
├── .gitignore                       # Git ignore rules for secrets, DBs, models, caches
├── ARCHITECTURE.md                  # Comprehensive technical architecture document
├── README.md                        # Project documentation
├── requirements.txt                 # Python package dependencies
├── web_app.py                       # Streamlit web application
├── main.py                          # CLI / OpenCV desktop entry point
├── config.py                        # Central configuration & ontology definitions
│
├── arduino/                         # Arduino C++ firmware
│   └── sensor_reader/
│       └── sensor_reader.ino        # DHT11, MQ-135, button, and I2C LCD driver
│
├── data/                            # Sample and test images
│   ├── custom/                      # User-added custom test images
│   └── samples/                     # Built-in sample gallery images
│
├── database/                        # Local SQLite database storage
│   └── .gitkeep
│
├── exports/                         # CSV and analysis report exports
│   └── .gitkeep
│
├── knowledge/                       # Domain knowledge base
│   └── freshness_rules.json         # Storage rules, shelf-life baselines, spoilage cues
│
├── models/                          # Local model weight storage
│   └── .gitkeep
│
├── services/                        # Core backend service modules
│   ├── arduino_flasher.py           # Auto-flash utility for Arduino boards
│   ├── database_service.py          # SQLite database connection & schema manager
│   ├── detection_service.py         # Open-vocabulary object detector
│   ├── evidence_aggregator.py       # Multimodal evidence packet fusion
│   ├── hardware_manager.py          # Hardware detection & mock failover manager
│   ├── history_service.py           # Historical query & export utilities
│   ├── llm_service.py               # Multimodal LLM reasoning & Q&A assistant
│   ├── rule_engine.py               # Deterministic rule-based freshness evaluator
│   ├── sensor_service.py            # Serial communication, mock telemetry & LCD driver
│   └── vision_service.py            # Hierarchical vision coordinator (Gemini/OpenAI/SigLIP)
│
├── tests/                           # Pytest test suite
│   ├── test_evidence_aggregator.py
│   ├── test_gemini_vision.py
│   ├── test_interactive_workflow.py
│   ├── test_rule_engine.py
│   ├── test_sensor_service.py
│   ├── test_siglip_vision.py
│   └── test_web_and_vision.py
│
└── utils/                           # Shared helper utilities
    ├── display.py                   # OpenCV HUD and bounding box visualizer
    └── logger.py                    # Structured logging utility
```

---

## Hardware Requirements

If deploying with physical hardware:
- **Microcontroller**: Arduino Uno R3 / R4, Nano, or ESP32.
- **Sensors**:
  - DHT11 or DHT22 Temperature & Humidity Sensor.
  - MQ-135 Air Quality / Hazardous Gas Sensor.
- **Display**: 16x2 LCD with I2C Backpack (Address `0x27` or `0x3F`).
- **Inputs**: Momentary Push Button with 10kΩ pull-up resistor (or internal pullup on Pin 2).
- **Camera**: USB Webcam (720p or 1080p recommended) or integrated laptop camera.

*Hardware is completely optional; the application includes full mock sensor simulation and sample image testing for purely software-based evaluation.*

---

## Limitations & Food Safety Disclaimer

> **IMPORTANT FOOD SAFETY NOTICE**:
> 
> This system is designed as an educational, informational, and experimental decision-support tool. **Visual inspection and ambient sensor estimation cannot guarantee food safety.**
> 
> - **Invisible Contaminants**: Pathogenic bacteria (*Salmonella*, *Listeria*, *E. coli*), microbial toxins, and bacterial contamination frequently occur without visible discoloration, odor, or mold.
> - **Internal Spoilage**: Visual models inspect surface characteristics only and cannot evaluate internal rot, core breakdown, or anaerobic decomposition.
> - **Environmental Variability**: Sensor readings reflect ambient or near-surface air and do not measure internal food temperature, pH, or water activity.
> 
> **Always follow official food safety guidelines, observe expiration dates, inspect smell and texture, and discard any food item if there is any doubt regarding its safety.**

---

## Future Enhancements

- **NIR Spectroscopy Integration**: Incorporating near-infrared or hyperspectral mini-sensors for internal brix and sugar/moisture quantification.
- **Edge Deployment**: Porting the local SigLIP model to ONNX Runtime / TensorRT for embedded devices (Raspberry Pi 5 / NVIDIA Jetson).
- **Multi-Camera Enclosures**: Calibrated enclosed illumination chambers for standardized computer vision evaluation independent of ambient lighting.
