# CHANGELOG

## [2.3.0] - 2026-10-02

### Added: High-Accuracy Multimodal Vision, Streamlit Web App & Render Cloud Deployment
- **Universal Multimodal Vision Inspection (`services/llm_service.py`)**:
  - Encodes high-resolution image frames directly as base64 JPEG to inspect visual cues (color shifts, wrinkles, browning, bruising, mold colonies, moisture).
  - Identifies 2,000+ global and regional food/fruit items zero-shot.
- **Expanded Food Class Ontology (`config.py`, `services/detection_service.py`, `knowledge/freshness_rules.json`)**:
  - Expanded prompt vocabulary to 120+ fruit, vegetable, bakery, dairy, and regional cooked items (*mango, papaya, guava, pomegranate, watermelon, grapes, lemon, dal, roti, rice, paneer, curry, samosa, biryani, etc.*).
  - Added full reference shelf-life and spoilage rules for all newly introduced items.
- **Interactive Streamlit Web Application (`web_app.py`)**:
  - **📁 File Upload**: Drag-and-drop support for any food photo (`.jpg`, `.jpeg`, `.png`, `.webp`).
  - **📷 Live Webcam**: Real-time browser camera snapshot via `st.camera_input`.
  - **🖼️ Sample Gallery**: Instant one-click testing of preloaded fruit/food images.
  - **🟢 Green Bounding Box Visualization**: Real-time bounding box annotations on detected foods.
  - **🔌 Flexible Sensor Telemetry**: Connects to live Arduino USB serial, simulates realistic mock telemetry, or uses interactive manual cloud sliders.
  - **📊 Historical Observations & CSV Export**: Searchable scan log with one-click `st.download_button` to download CSV.
- **Render Cloud Deployment Ready (`render.yaml`, `Procfile`)**:
  - Added configuration for one-command deployment to Render web service.
- **New Unit Tests (`tests/test_web_and_vision.py`)**: Added 3 new unit tests (19 total unit tests passing).

---

## [2.2.0] - 2026-10-02

### Added: Ordered Sequential Food Detection, Triggered Sensor Measurement & LCD Display Integration
- **Strict Food-Only Detection & Vibrant Green Boxes**: Updated `utils/display.py` and `config.py` to highlight only verified edible food/fruit/vegetable items with green bounding boxes and suppress empty-frame detections with a clean search HUD.
- **Interactive State Machine (`main.py`)**:
  - `STATE_SCANNING`: Scans live camera feed for food items.
  - `STATE_LOCKED`: Locks food on detection/spacebar, holding the camera frame and rendering placement instructions.
  - `STATE_MEASURING`: Captures sensor telemetry snapshot at the exact moment of trigger (via keyboard `[SPACE]` or Arduino hardware button), eliminating empty ambient readings.
  - `STATE_RESULT`: Displays comprehensive freshness card on screen and on the Arduino LCD.
- **Two-Way Serial LCD Protocol & Button Hardware Trigger (`services/sensor_service.py`, `arduino/sensor_reader/sensor_reader.ino`)**:
  - Supports sending two-line LCD commands (`LCD:Line1|Line2`) to Arduino for 16x2 / I2C displays.
  - Supports hardware push-button on PIN 2 sending `{"button":"pressed"}` trigger events over serial.
- **New Unit Tests (`tests/test_interactive_workflow.py`)**: 4 new tests for LCD message formatting, trigger snapshot marking, hardware button detection, and HUD card rendering (16 tests total).

---

## [2.1.0] - 2026-10-02

### Added: One-Command Autonomous Execution & Arduino Auto-Flash Subsystem
- **`services/arduino_flasher.py`**: Auto-detects `arduino-cli` binary, queries connected USB boards, and compiles + uploads the sensor sketch automatically.
- **`services/hardware_manager.py`**: Seamless hardware orchestrator — detects if an Arduino is connected on launch; if connected, flashes code and connects to serial telemetry; if not connected, automatically initializes realistic Mock Sensors with zero manual config changes.
- **Streamlined `main.py`**: Handles hardware discovery, flashing, camera capture, detection, multimodal fusion, LLM analysis, UI display, and SQLite persistence in a single command.

---

## [2.0.0] - 2026-10-02

### MAJOR RE-ARCHITECTURE: Transformation from Face Attendance to Fruit & Food Freshness System
- Safely separated legacy face attendance project to `legacy_face_project/`.
- Integrated YOLO-World open-vocabulary object detector.
- Implemented Arduino serial sensor ingestion with JSON Lines protocol.
- Built Knowledge Base with storage guidelines (`knowledge/freshness_rules.json`).
- Built Multimodal Evidence Aggregator.
- Built OpenAI GPT-4o-mini structured freshness reasoning with offline fallback.
- Built SQLite persistence and CSV export.
