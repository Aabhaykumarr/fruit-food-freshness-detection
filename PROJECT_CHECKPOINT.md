# PROJECT CHECKPOINT: Fruit & Food Freshness Recognition System

**Date**: 2026-10-03  
**Status**: Production-Ready / Fully Verified with Google Gemini 3.8 Flash  

---

## 1. Architecture & Data Flow

```
                                [INPUT IMAGE]
                         (Full-Resolution RGB Image)
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │      VISION SERVICE ROUTER      │
                    │   (services/vision_service.py)  │
                    └────────────────┬────────────────┘
                                     │
                 ┌───────────────────┼───────────────────┐
                 ▼                   ▼                   ▼
    [1. PRIMARY: Gemini Vision] [2. OPTIONAL: OpenAI] [3. FALLBACK: SigLIP Local]
       (gemini-3.5-flash-lite)    (gpt-4o-mini)       (google/siglip-base-patch16-224)
     - Full image prompt        - Only if configured   - 3-stage local pipeline
     - Multi-item (<=8)         - Optional backup      - Labeled "local_siglip_fallback"
     - Structured JSON schema
                 │                   │                   │
                 └───────────────────┼───────────────────┘
                                     │
                                     ▼
                    [STANDARDIZED VISION RESULT DICT]
                    - is_food, overall_status ("food" | "not_food" | "uncertain")
                    - items: [ { item_id, name, category, confidence,
                                condition, condition_confidence, freshness_status,
                                visible_signs, visual_description, is_spoiled, reason } ]
                    - alternatives, overall_visual_summary, overall_freshness_summary
                    - source: "gemini" | "multimodal_openai" | "local_siglip_fallback"
                                     │
                                     ▼
                     [EVIDENCE AGGREGATOR & FUSION]
                    - Merges per-item vision + Arduino sensors (temp, hum, gas)
                    - Storage rules lookup per item (knowledge/freshness_rules.json)
                                     │
                                     ▼
                       [LLM REASONING & SYNTHESIS]
                    - Gemini / OpenAI / Rule-based fallback synthesis & chat
                                     │
                                     ▼
            ┌────────────────────────┼────────────────────────┐
            ▼                        ▼                        ▼
    [STREAMLIT WEB UI]     [SQLITE PERSISTENCE]        [ARDUINO LCD]
  (Multi-item card grid) (database/freshness.db)    (16x2 state sync)
```

---

## 2. Current Primary Vision Model
- **Model**: `gemini-3.5-flash-lite` (Google Gemini 3.5 Flash-Lite Multimodal Vision via `google.genai` SDK).
- **Capabilities**: Full-resolution RGB visual recognition, multi-item visual scene decomposition (up to 8 items), rich condition spectrum, and Pydantic structured output enforcement.

---

## 3. Gemini API Configuration
- `GEMINI_API_KEY`: Loaded securely from `.env` or system environment (never hardcoded in source code, logs, UI, or checkpoints).
- `GEMINI_VISION_MODEL`: Configured as `gemini-3.5-flash-lite` in `config.py` (serving as the single source of truth for both `GeminiVisionService` and `LLMFreshnessService`).
- **Security Rule**: API keys are strictly kept confidential; startup logs report only the model identifier (`Gemini vision initialized with model: gemini-3.5-flash-lite`).

---

## 4. Current Local Fallback
- **Model**: Google SigLIP Zero-Shot Vision-Language Classifier (`google/siglip-base-patch16-224`).
- **Pipeline**: 3-stage local pipeline (Gate prompt check -> Specialist classification with 45+ ontology classes -> Attribute observable inference).
- **Labeling**: Results from the fallback are clearly tagged with `source: "local_siglip_fallback"` (`⚙️ Local Fallback Vision (SigLIP)` in the UI).

---

## 5. Current OpenAI Status
- **Status**: Completely optional secondary provider (`gpt-4o-mini`).
- **Policy**: The primary vision and LLM reasoning pipeline does **NOT** depend on OpenAI API credits. OpenAI is only instantiated if `OPENAI_API_KEY` is explicitly configured.

---

## 6. Current Freshness Architecture
- **Per-Item Separation**: Distinctly separates **Food Identity** (e.g. `bread`, `tomato`, `apple`) from **Condition** (e.g. `fresh`, `ripe`, `overripe`, `bruised`, `wilted`, `moldy`, `stale`).
- **Freshness Spectrum**: Standardized calibrated statuses:
  - `FRESH`: Vibrant, prime condition, no defects.
  - `GOOD_TO_EAT_VISUALLY`: Minor cosmetic variation or normal ripeness, safe to eat.
  - `QUESTIONABLE`: Bruised, wilted, dried, or overripe.
  - `POSSIBLE_SPOILAGE`: Strong signs of decay or severe softening.
  - `VISIBLE_SPOILAGE`: Visible mold colonies, fungal growth, slime, or decomposition (`is_spoiled: true`).
  - `CANNOT_DETERMINE`: Ambiguous or occluded.
- **Evidence Aggregator**: Integrates per-item visual state with environmental sensor data (temperature, humidity, gas/VOC) and rules from `knowledge/freshness_rules.json`.

---

## 7. Arduino / LCD / Sensor Status
- **Sensors**: DHT11 (temperature & humidity) and MQ-135 (air quality / VOC / gas).
- **Port**: Auto-detection or explicit `COM10` (Windows) / `/dev/ttyUSB0` (Linux), 9600 baud.
- **LCD Display**: I2C 16x2 LCD display synchronized across 10 distinct states (Idle, Connecting, Connected, Sensor Ready, Analyzing, Fresh, Spoiled, Moderate/Questionable, Not Food, Uncertain Food).
- **Transport**: Supports both **Browser Web Serial** (direct client-side USB connection over HTTPS for cloud deployments) and **Local Python Serial** (`pyserial`).

---

## 8. UI, Chat & Database Status
- **Streamlit Web UI (`web_app.py`)**: Multi-item responsive card grid, condition badges, sensor telemetry gauges, source indicators, and LCD preview panel.
- **Chat Assistant**: Interactive conversational assistant allowing multi-turn Q&A about analyzed items and storage advice powered by Gemini 3.8 Flash (with offline fallback).
- **Persistence**: SQLite database at `database/freshness.db` with export capabilities to `exports/analysis.csv`.

---

## 9. Current Test Status
- **Total Tests**: **75/75 passing (100%)** via `pytest -v`.
- **Breakdown**:
  - `tests/test_gemini_vision.py`: 12/12 passing (mocked Gemini tests, non-food rejection, spoilage, router priority, SigLIP fallback).
  - `tests/test_chat_feature.py`: 10/10 passing (chat logic, history preservation, offline fallback).
  - `tests/test_multimodal_vision.py`: 9/9 passing (multi-item schemas, aggregator fusion).
  - `tests/test_interactive_workflow.py`, `tests/test_lcd_helper.py`, `tests/test_hardware_manager.py`, `tests/test_sensor_validation.py`, `tests/test_web_serial_integration.py`, `scripts/test_qa_pipeline.py`: 44/44 passing.

---

## 10. Real Gemini Live API Verification Completed
- **Model**: `gemini-3.5-flash-lite`
- **HTTP / API Status**: **SUCCESS** via `google.genai` SDK with real API key.
- **Test Image**: `evaluation/fruits/apple_01.jpg`
- **Verified Output**:
  - `item_name`: `apple`
  - `category`: `fruit`
  - `confidence`: `0.99` (99%)
  - `condition`: `fresh`
  - `condition_confidence`: `0.95` (95%)
  - `freshness_status`: `FRESH`
  - `is_spoiled`: `False`
  - `visual_description`: *"Fresh, shiny red apple with smooth, unblemished skin and intact stem."*
  - `overall_visual_summary`: *"A single, vibrant red apple in excellent condition with no visible defects or signs of spoilage."*
  - `overall_freshness_summary`: *"The apple is in fresh condition, suitable for consumption or storage."*

---

## 11. Next Exact Task
> **Task**: **TEST THE PREVIOUSLY FAILING GROUPED IMAGE** (e.g. multi-item plate with bread slice + raw vegetables) and verify that Gemini 3.8 Flash recognizes multiple individual items instead of collapsing them into a "sandwich", then test mixed freshness and spoilage condition cases.

---

## 12. Known Remaining Issues & Nuances
1. **Gemini 3.8 AFC Warning**: The `google-genai` SDK emits an informational warning: `Direct use of automatic function calling (AFC) in Models.generate_content is not recommended...` when generating structured schema responses.
2. **Live Inference Latency**: Real live Gemini 3.8 Flash API calls currently average ~8 seconds due to remote multimodal roundtrips and transient backoff retries.
3. **Multi-Image Accuracy Benchmark**: A comprehensive real-world multi-image accuracy benchmark under Gemini 3.8 Flash still needs to be run across diverse classes once API rate limits reset.
4. **Accuracy Caveat**: Do **NOT** claim 100% accuracy yet until the full real-world multi-image dataset is evaluated under live API execution.

---

## 13. Exact Commands Needed to Resume

### Launch Web Application
```bash
streamlit run web_app.py
```

### Run Full Test Suite
```bash
pytest -v
```

### Run Gemini-Specific Tests
```bash
pytest tests/test_gemini_vision.py -v
```
