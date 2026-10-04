# System Architecture: Fruit & Food Freshness Detection System

## 1. High-Level Interactive Workflow

```
                                SYSTEM ENTRYPOINTS
             ┌───────────────────────────┴───────────────────────────┐
             ▼                                                       ▼
  [WEB UI: streamlit run web_app.py]                        [CLI: python main.py]
  (Browser Cam / Photo Upload / Gallery)                 (Live Webcam / OpenCV Display)
             │                                                       │
             └───────────────────────────┬───────────────────────────┘
                                         │
                                         ▼
                               [HARDWARE MANAGER]
                         [services/hardware_manager.py]
                                         │
                       ┌─────────────────┴─────────────────┐
                       ▼                                   ▼
              [ARDUINO CONNECTED]                 [ARDUINO NOT CONNECTED]
                       │                                   │
                       ▼                                   ▼
             [services/arduino_flasher]            [services/sensor_service]
            (Auto-compile & auto-upload             (Realistic Mock Sensors or
            arduino/sensor_reader.ino)               Interactive Cloud Sliders)
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
             (gemini-3.5-flash-lite)(gpt-4o-mini)      (Zero-shot non-food gate,
                 Structured Output Structured Output   45+ food ontology,
                 0-retry on 429     (Optional API)     condition classifier)
                       └─────────────────┬─────────────────┘
                                         │
                                         ▼
                        [STAGE 2: LOCKED & PLACEMENT PROMPT]
                        - Camera freezes or displays selected photo
                        - Display Prompt: "Place food near sensor"
                        - Arduino LCD: "Detected: [FOOD]\nPlace at sensor"
                                         │
                             (Press [SPACE] / Arduino Button / Web Click)
                                         │
                                         ▼
                        [STAGE 3: TRIGGERED SENSOR SNAPSHOT]
                        - Instantaneous measurement at placement
                        - Eliminates false ambient-only data (DHT11 + MQ-135)
                                         │
                                         ▼
                        [STAGE 4: MULTIMODAL REASONING]
                        [services/evidence_aggregator.py & services/llm_service.py]
                        - Visual cues + Triggered Sensors + Reference Rules + History
                        - Gemini / GPT-4o-mini / Deterministic Rule-Based Engine
                                         │
                                         ▼
                        [STAGE 5: RESULT, PERSISTENCE & Q&A]
                        - Full Freshness Analysis Dashboard Card
                        - Interactive Contextual Q&A Chat Assistant
                        - Arduino LCD: "[FOOD]: FRESH\nShelf: 2-5 days"
                        - SQLite Persistence (`database/freshness.db`)
                        - CSV Export (`exports/analysis.csv`)
```

---

## 2. Core Subsystems

### A. Hierarchical Vision Subsystem (`services/vision_service.py`)
- **Primary Cloud Vision**: Google Gemini (`gemini-3.5-flash-lite`) via `google-genai` with Pydantic structured output (`ImageVisionResult`). Evaluates visual freshness cues, color shifts, surface degradation, mold presence, and condition confidence. Implements fast-fail with zero retries on 429 `RESOURCE_EXHAUSTED` (rate-limit/quota handling) to immediately trigger fallback.
- **Secondary Cloud Vision**: OpenAI Vision (`gpt-4o-mini`) structured completion if Gemini is unavailable and `OPENAI_API_KEY` is provided.
- **Local Offline Fallback Vision**: Google SigLIP (`google/siglip-base-patch16-224`) running entirely locally:
  - *Stage 1*: Zero-shot non-food rejection gate.
  - *Stage 2*: Specialist 45+ class food ontology classification with synonym normalization.
  - *Stage 3*: Observable attribute and condition assessment with margin calibration.

### B. Sensor Telemetry & Hardware Integration (`services/sensor_service.py`, `services/hardware_manager.py`)
- **Arduino Firmware (`arduino/sensor_reader/sensor_reader.ino`)**: Ingests DHT11 temperature/humidity data and MQ-135 air quality/VOC sensor telemetry.
- **Two-Way LCD Protocol**: Transmits 2-line sanitized status updates (`LCD:Line1|Line2`) to 16x2 I2C physical LCD screens.
- **Hardware Trigger**: Physical push button on Pin 2 emits serial events (`{"button":"pressed"}`) to trigger synchronized multi-sensor measurement.
- **Multi-Channel Ingestion**: Local Python Serial (`pyserial`), Browser Web Serial API (W3C standard over HTTPS for cloud deployments), realistic mock telemetry generator, or manual dashboard sliders.

### C. Multimodal Evidence Aggregation & LLM Reasoning (`services/evidence_aggregator.py`, `services/llm_service.py`)
- Fuses visual condition assessments, real-time sensor telemetry, reference shelf-life knowledge (`knowledge/freshness_rules.json`), and previous scan history into an `EvidencePacket`.
- Generates structured freshness determinations (`FRESH`, `GOOD_TO_EAT_VISUALLY`, `QUESTIONABLE`, `POSSIBLE_SPOILAGE`, `VISIBLE_SPOILAGE`, `CANNOT_DETERMINE`), estimated shelf-life windows, storage recommendations, and safety considerations.
- Provides interactive conversational chat assistance with context from the latest scan.
- Includes a deterministic offline rule engine fallback when cloud LLM APIs are unavailable.

### D. User Interface & Persistence (`web_app.py`, `main.py`, `services/history_service.py`)
- **Streamlit Web Application (`web_app.py`)**: Multi-tab image ingestion (Webcam, Upload, Preloaded Gallery), sensor telemetry controls, visual HUD cards, chat assistant, and historical observation table.
- **CLI Mode (`main.py`)**: OpenCV-based real-time video stream with keyboard controls and terminal output.
- **Persistence**: SQLite database (`database/freshness.db`) and CSV exporter (`exports/analysis.csv`).
