"""
Fruit & Food Freshness Multimodal Analysis System — Streamlit Web UI.

Workflow:
1. Upload Image OR Capture Single Photo
2. Trigger "Analyze Freshness"
3. Hierarchical Gated Vision Processing:
   - Stage 1: Zero-Shot Food vs. Non-Food Gating
   - Stage 2: Specialized Category Classification (Fruits, Vegetables, Prepared Meals, Indian Cuisine)
   - Stage 3: Uncertainty & Margin-Based Calibration
4. Arduino Environmental Telemetry (Temp, Humidity, Gas/VOC)
5. Knowledge Rules & LLM Multimodal Reasoning
6. SQLite History Persistence & CSV Export
"""

import os
import io
import time
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from config import (
    VISION_MODEL_NAME,
    DATABASE_PATH,
    EXPORT_PATH,
    FRESHNESS_RULES_PATH,
    GEMINI_API_KEY,
    GEMINI_VISION_MODEL,
    OPENAI_API_KEY,
    LLM_MODEL,
    SERIAL_PORT,
    BAUD_RATE,
)
from services.vision_service import VisionService
from services.sensor_service import SensorService
from services.evidence_aggregator import EvidenceAggregator
from services.llm_service import LLMFreshnessService
from services.history_service import HistoryService
from services.web_serial_component import render_web_serial_connector
from utils.lcd_helper import (
    get_lcd_message_for_state,
    build_lcd_command,
    format_lcd_message,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _queue_freshness_analysis(
    temperature_c: Optional[float],
    humidity_percent: Optional[float],
    gas_value: Optional[float],
    sensor_source: str,
    sensor_status: str,
):
    """Latch the sensor values and pause browser telemetry before analysis starts."""
    snapshot = {
        "temperature_c": temperature_c,
        "humidity_percent": humidity_percent,
        "gas_value": gas_value,
        "sensor_status": sensor_status,
        "sensor_scope": "environment",
        "source": sensor_source,
        "timestamp": datetime.now().isoformat(),
        "captured_at": datetime.now().isoformat(),
        "is_triggered_reading": True,
    }

    # Local Python Serial can sample its background reader at the actual click.
    local_service = st.session_state.get("local_sensor_svc")
    if sensor_source == "local_serial" and local_service is not None:
        latest = local_service.capture_snapshot()
        snapshot.update({
            key: latest.get(key, snapshot.get(key))
            for key in ("temperature_c", "humidity_percent", "gas_value", "sensor_status", "timestamp", "captured_at", "is_triggered_reading")
        })
    elif sensor_source == "browser_serial":
        latest = st.session_state.get("browser_web_serial_connector")
        if isinstance(latest, dict) and latest.get("connected"):
            for key in ("temperature_c", "humidity_percent", "gas_value"):
                if key in latest:
                    snapshot[key] = latest[key]
            snapshot["sensor_status"] = latest.get("sensor_status", sensor_status)
            snapshot["captured_at"] = datetime.now().isoformat()

    st.session_state.analysis_sensor_snapshot = snapshot
    st.session_state.analysis_pending = True
    st.session_state.analysis_running = True
    st.session_state.current_lcd_command = build_lcd_command("Analyzing...", "Please wait...")
    if local_service is not None and sensor_source == "local_serial":
        local_service.send_lcd_message("Analyzing...", "Please wait...")

# =============================================================================
# STREAMLIT PAGE CONFIG & STYLES
# =============================================================================

st.set_page_config(
    page_title="Food & Fruit Freshness Multimodal System",
    page_icon="🍏",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #10B981;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .status-card-recognized {
        background-color: #064E3B;
        border: 1px solid #059669;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        color: #ECFDF5;
    }
    .status-card-uncertain {
        background-color: #78350F;
        border: 1px solid #D97706;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        color: #FFFBEB;
    }
    .status-card-notfood {
        background-color: #7F1D1D;
        border: 1px solid #DC2626;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        color: #FEF2F2;
    }
    .telemetry-box {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: bold;
        color: #F8FAFC;
    }
    .status-badge-fresh {
        background-color: #064E3B;
        color: #34D399;
        padding: 0.35rem 0.75rem;
        border-radius: 8px;
        font-weight: bold;
        display: inline-block;
    }
    .status-badge-moderate {
        background-color: #78350F;
        color: #FBBF24;
        padding: 0.35rem 0.75rem;
        border-radius: 8px;
        font-weight: bold;
        display: inline-block;
    }
    .status-badge-questionable {
        background-color: #7C2D12;
        color: #FB923C;
        padding: 0.35rem 0.75rem;
        border-radius: 8px;
        font-weight: bold;
        display: inline-block;
    }
    .status-badge-not-fresh {
        background-color: #7F1D1D;
        color: #F87171;
        padding: 0.35rem 0.75rem;
        border-radius: 8px;
        font-weight: bold;
        display: inline-block;
    }
    .item-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.9rem;
        margin-bottom: 0.75rem;
    }
    .item-card-spoiled {
        background-color: #450A0A;
        border: 1px solid #DC2626;
        border-radius: 8px;
        padding: 0.9rem;
        margin-bottom: 0.75rem;
    }
    .item-card-fresh {
        background-color: #064E3B;
        border: 1px solid #059669;
        border-radius: 8px;
        padding: 0.9rem;
        margin-bottom: 0.75rem;
    }
    .source-tag {
        font-size: 0.75rem;
        padding: 3px 8px;
        border-radius: 6px;
        background: #0F172A;
        color: #38BDF8;
        border: 1px solid #0284C7;
        font-weight: 600;
        display: inline-block;
    }
    .source-tag-fallback {
        font-size: 0.75rem;
        padding: 3px 8px;
        border-radius: 6px;
        background: #1E293B;
        color: #94A3B8;
        border: 1px solid #475569;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CACHED SERVICE INITIALIZATION
# =============================================================================

@st.cache_resource
def get_vision_service():
    """Cache the Hierarchical SigLIP Vision Service across sessions."""
    return VisionService(model_name=VISION_MODEL_NAME)


@st.cache_resource
def get_evidence_aggregator():
    return EvidenceAggregator(rules_path=FRESHNESS_RULES_PATH)


@st.cache_resource
def get_history_service():
    return HistoryService(db_path=DATABASE_PATH, export_path=EXPORT_PATH)


def get_freshness_badge_html(status: str) -> str:
    """Generate styled HTML badge for freshness status."""
    status_upper = (status or "UNKNOWN").upper()
    if status_upper == "FRESH":
        return '<span class="status-badge-fresh">🟢 FRESH</span>'
    elif status_upper == "MODERATELY_FRESH":
        return '<span class="status-badge-moderate">🟡 MODERATELY FRESH</span>'
    elif status_upper == "QUESTIONABLE":
        return '<span class="status-badge-questionable">🟠 QUESTIONABLE</span>'
    elif status_upper == "NOT_FRESH":
        return '<span class="status-badge-not-fresh">🔴 NOT FRESH / SPOILED</span>'
    elif status_upper == "NOT_FOOD":
        return '<span class="status-badge-not-fresh">⛔ NOT FOOD</span>'
    else:
        return f'<span class="status-badge-moderate">⚪ {status_upper}</span>'


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    # Sidebar: Controls & Sensor Telemetry
    st.sidebar.title("⚙️ System Controls")

    # API Key Configuration
    api_key_input = st.sidebar.text_input(
        "Google Gemini API Key",
        value=GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")),
        type="password",
        help="Enables Google Gemini multimodal vision & deep reasoning (free tier). SigLIP fallback used if blank.",
    )

    effective_api_key = api_key_input.strip() if api_key_input else (GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")))

    # Initialize LCD command state if not set
    if "current_lcd_command" not in st.session_state:
        st.session_state.current_lcd_command = build_lcd_command("Put food near", "Upload image")

    # Sensor Telemetry Mode
    st.sidebar.subheader("🔌 Environmental Sensors")
    sensor_mode = st.sidebar.radio(
        "Sensor Source",
        options=[
            "Live Arduino (Browser Web Serial)",
            "Live Arduino (Local Python Serial)",
            "Realistic Mock Sensors",
            "Manual Telemetry Sliders",
        ],
        index=0,
        help="Select Browser Web Serial (for deployed Render HTTPS), Local Python Serial, or simulation modes.",
    )

    temp_c: Optional[float] = None
    hum_pct: Optional[float] = None
    gas_val: Optional[float] = None
    sensor_source_name = "manual_slider"
    sensor_status_str = "active"
    local_hw_svc: Optional[SensorService] = None

    # Stop local Python serial service if switching away from Local Serial mode
    if sensor_mode != "Live Arduino (Local Python Serial)":
        if "local_sensor_svc" in st.session_state and st.session_state.local_sensor_svc is not None:
            try:
                st.session_state.local_sensor_svc.stop()
            except Exception:
                pass
            st.session_state.local_sensor_svc = None

    if sensor_mode == "Live Arduino (Browser Web Serial)":
        st.sidebar.caption("🌐 **Browser Web Serial** connects directly to your USB Arduino from Chrome / Edge over HTTPS.")
        web_serial_data = render_web_serial_connector(
            lcd_command=st.session_state.get("current_lcd_command", ""),
            pause_telemetry=bool(st.session_state.get("analysis_running", False)),
            key="browser_web_serial_connector",
        )
        if web_serial_data and web_serial_data.get("connected"):
            temp_c = float(web_serial_data["temperature_c"]) if web_serial_data.get("temperature_c") is not None else None
            hum_pct = float(web_serial_data["humidity_percent"]) if web_serial_data.get("humidity_percent") is not None else None
            gas_val = float(web_serial_data["gas_value"]) if web_serial_data.get("gas_value") is not None else None
            sensor_source_name = "browser_serial"
            sensor_status_str = "active" if (temp_c is not None or hum_pct is not None or gas_val is not None) else "unavailable"
            temp_disp = f"{temp_c:.1f}°C" if temp_c is not None else "--.-°C"
            hum_disp = f"{hum_pct:.1f}%" if hum_pct is not None else "--.-%"
            gas_disp = f"{int(gas_val)} ppm" if gas_val is not None else "N/A"
            st.sidebar.success(
                f"🟢 Arduino Connected (Browser Serial)\n\n"
                f"🌡️ {temp_disp} | 💧 {hum_disp} | 💨 {gas_disp}"
            )
        elif web_serial_data and web_serial_data.get("error"):
            sensor_source_name = "none"
            sensor_status_str = "unavailable"
            st.sidebar.warning(f"⚠️ {web_serial_data.get('error')}")
        else:
            sensor_source_name = "none"
            sensor_status_str = "unavailable"
            st.sidebar.info("🔴 Arduino Disconnected. Click **[Connect Arduino]** above to stream USB sensor telemetry.")

    elif sensor_mode == "Live Arduino (Local Python Serial)":
        serial_port_input = st.sidebar.text_input("Serial Port", value=SERIAL_PORT or "COM10")
        try:
            if (
                "local_sensor_svc" not in st.session_state
                or st.session_state.local_sensor_svc is None
                or st.session_state.get("local_sensor_port") != serial_port_input
            ):
                if "local_sensor_svc" in st.session_state and st.session_state.local_sensor_svc is not None:
                    st.session_state.local_sensor_svc.stop()
                st.session_state.local_sensor_svc = SensorService(
                    port=serial_port_input,
                    baud_rate=BAUD_RATE,
                    mock_mode=False,
                )
                st.session_state.local_sensor_svc.start()
                st.session_state.local_sensor_port = serial_port_input

            local_hw_svc = st.session_state.local_sensor_svc
            reading = local_hw_svc.get_latest_reading()
            has_reading = (
                reading.get("temperature_c") is not None
                or reading.get("humidity_percent") is not None
                or reading.get("gas_value") is not None
            )
            if has_reading:
                temp_c = float(reading["temperature_c"]) if reading.get("temperature_c") is not None else None
                hum_pct = float(reading["humidity_percent"]) if reading.get("humidity_percent") is not None else None
                gas_val = float(reading["gas_value"]) if reading.get("gas_value") is not None else None
                sensor_source_name = "local_serial"
                sensor_status_str = "active"
                temp_disp = f"{temp_c:.1f}°C" if temp_c is not None else "--.-°C"
                hum_disp = f"{hum_pct:.1f}%" if hum_pct is not None else "--.-%"
                gas_disp = f"{int(gas_val)} ppm" if gas_val is not None else "N/A"
                st.sidebar.success(
                    f"🟢 Connected (Local Serial)\n\n"
                    f"🌡️ {temp_disp} | 💧 {hum_disp} | 💨 {gas_disp}"
                )
            else:
                temp_c = None
                hum_pct = None
                gas_val = None
                sensor_source_name = "local_serial"
                sensor_status_str = "unavailable"
                st.sidebar.warning("Arduino connected on local serial, awaiting telemetry...")
        except Exception as err:
            temp_c = None
            hum_pct = None
            gas_val = None
            sensor_source_name = "none"
            sensor_status_str = "unavailable"
            st.sidebar.error(f"Serial Error: {err}. Using ambient fallback.")

    elif sensor_mode == "Realistic Mock Sensors":
        mock_svc = SensorService(mock_mode=True)
        reading = mock_svc.get_latest_reading()
        temp_c = reading.get("temperature_c", 23.8)
        hum_pct = reading.get("humidity_percent", 56.2)
        gas_val = reading.get("gas_value", 135.0)
        sensor_source_name = "mock"
        sensor_status_str = "active"
        st.sidebar.info(f"Mock Telemetry: {temp_c}°C | {hum_pct}% | {gas_val} ppm")

    elif sensor_mode == "Manual Telemetry Sliders":
        temp_c = st.sidebar.slider("Ambient Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.5)
        hum_pct = st.sidebar.slider("Ambient Humidity (%)", min_value=10.0, max_value=100.0, value=58.0, step=1.0)
        gas_val = st.sidebar.slider("Gas / VOC Sensor (ppm)", min_value=10.0, max_value=600.0, value=120.0, step=5.0)
        sensor_source_name = "manual_slider"
        sensor_status_str = "active"

    st.sidebar.markdown("---")
    st.sidebar.caption("© 2026 Hierarchical Gated Vision & Freshness System")

    # Header
    st.markdown('<div class="main-header">🍏 Fruit & Food Freshness Recognition System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Hierarchical Gated Vision • Arduino Environmental Telemetry • Multimodal LLM Reasoning</div>',
        unsafe_allow_html=True,
    )

    # Initialize Services
    vision_service = get_vision_service()
    aggregator = get_evidence_aggregator()
    llm_service = LLMFreshnessService(api_key=effective_api_key, model=LLM_MODEL)
    history_service = get_history_service()

    # Tabs for Inputs
    tab_upload, tab_camera, tab_samples, tab_history = st.tabs([
        "📁 Upload Photo",
        "📷 Capture Photo",
        "🖼️ Sample Gallery",
        "📊 History & CSV Export"
    ])

    image_input = None
    input_source_name = ""

    # Tab 1: Upload Photo
    with tab_upload:
        uploaded_file = st.file_uploader(
            "Choose a photo of fruit, vegetable, or food dish",
            type=["jpg", "jpeg", "png", "webp"],
            help="High-resolution single photo of any fruit, vegetable, or prepared food.",
        )
        if uploaded_file is not None:
            image_input = Image.open(uploaded_file).convert("RGB")
            input_source_name = uploaded_file.name
            st.session_state.selected_sample_path = None

    # Tab 2: Capture Photo
    with tab_camera:
        camera_img = st.camera_input("Point camera at item and click Take Photo")
        if camera_img is not None:
            image_input = Image.open(camera_img).convert("RGB")
            input_source_name = getattr(camera_img, "name", "camera_capture.jpg") or "camera_capture.jpg"
            st.session_state.selected_sample_path = None

    # Tab 3: Sample Gallery
    with tab_samples:
        st.write("Test with sample benchmark images:")
        col1, col2, col3 = st.columns(3)
        sample_path = None
        with col1:
            if os.path.exists("evaluation/dataset/fruits/apple_01.jpg"):
                st.image("evaluation/dataset/fruits/apple_01.jpg", caption="Apple (Fruit)", use_container_width=True)
                if st.button("🧪 Test: Apple"):
                    sample_path = "evaluation/dataset/fruits/apple_01.jpg"
            elif os.path.exists("data/samples/fruits.jpg"):
                st.image("data/samples/fruits.jpg", caption="Sample Fruits", use_container_width=True)
                if st.button("🧪 Test: Fruits"):
                    sample_path = "data/samples/fruits.jpg"
        with col2:
            if os.path.exists("evaluation/dataset/foods/biryani_01.jpg"):
                st.image("evaluation/dataset/foods/biryani_01.jpg", caption="Biryani (Indian Dish)", use_container_width=True)
                if st.button("🧪 Test: Biryani"):
                    sample_path = "evaluation/dataset/foods/biryani_01.jpg"
        with col3:
            if os.path.exists("evaluation/dataset/non_food/phone_01.jpg"):
                st.image("evaluation/dataset/non_food/phone_01.jpg", caption="Phone (Non-Food Negative)", use_container_width=True)
                if st.button("🧪 Test: Phone (Negative)"):
                    sample_path = "evaluation/dataset/non_food/phone_01.jpg"

        if sample_path:
            st.session_state.selected_sample_path = sample_path
            image_input = Image.open(sample_path).convert("RGB")
            input_source_name = os.path.basename(sample_path)
        elif image_input is None and st.session_state.get("selected_sample_path"):
            saved_sample = st.session_state.selected_sample_path
            if os.path.exists(saved_sample):
                image_input = Image.open(saved_sample).convert("RGB")
                input_source_name = os.path.basename(saved_sample)

    # -------------------------------------------------------------------------
    # ANALYSIS WORKFLOW
    # -------------------------------------------------------------------------
    if image_input is not None:
        st.markdown("---")
        col_left, col_right = st.columns([1.0, 1.2])

        with col_left:
            st.subheader("1️⃣ Input Image")
            st.image(image_input, caption=f"Source: {input_source_name}", use_container_width=True)

            # Keep the result card tied to the sensor values actually used by
            # the latest analysis; otherwise show the current live readings.
            shown_snapshot = None
            if st.session_state.get("analysis_pending"):
                shown_snapshot = st.session_state.get("analysis_sensor_snapshot")
            elif st.session_state.get("analyzed_image_source") == input_source_name:
                shown_snapshot = st.session_state.get("last_analysis_sensor_snapshot")
            shown_snapshot = shown_snapshot or {}
            shown_temp = shown_snapshot.get("temperature_c", temp_c)
            shown_humidity = shown_snapshot.get("humidity_percent", hum_pct)
            shown_gas = shown_snapshot.get("gas_value", gas_val)
            telemetry_heading = "SENSOR SNAPSHOT FOR ANALYSIS" if shown_snapshot else "LIVE ENVIRONMENTAL TELEMETRY"
            temp_card_disp = f"{shown_temp:.1f}°C" if shown_temp is not None else "--.-°C"
            hum_card_disp = f"{shown_humidity:.1f}%" if shown_humidity is not None else "--.-%"
            gas_card_disp = f"{int(shown_gas)} ppm" if shown_gas is not None else "N/A"
            st.markdown(f"""
            <div class="telemetry-box">
                <div style="font-weight: 600; color: #94A3B8; margin-bottom: 6px;">{telemetry_heading}</div>
                <div style="display: flex; justify-content: space-between;">
                    <div>🌡️ <b>Temp:</b> {temp_card_disp}</div>
                    <div>💧 <b>Humidity:</b> {hum_card_disp}</div>
                    <div>💨 <b>Gas/VOC:</b> {gas_card_disp}</div>
                </div>
                <div style="font-size: 0.8rem; color: #64748B; margin-top: 6px;">Sensor Source: {sensor_source_name} (ambient scope)</div>
            </div>
            """, unsafe_allow_html=True)

            # Reset LCD prompt to idle only if a brand new unanalyzed image was selected/uploaded
            if (
                st.session_state.get("analyzed_image_source") != input_source_name
                and not st.session_state.get("analysis_pending", False)
                and not st.session_state.get("analysis_running", False)
            ):
                idle_lcd_cmd = build_lcd_command("Put food near", "Upload image")
                if st.session_state.get("current_lcd_command") != idle_lcd_cmd:
                    st.session_state.current_lcd_command = idle_lcd_cmd
                    if local_hw_svc is not None:
                        local_hw_svc.send_lcd_message("Put food near", "Upload image")
            elif (
                st.session_state.get("analyzed_image_source") == input_source_name
                and st.session_state.get("last_lcd_result_cmd")
                and not st.session_state.get("analysis_pending", False)
                and not st.session_state.get("analysis_running", False)
            ):
                # Keep the result LCD command latched across UI reruns until a new scan begins
                if st.session_state.get("current_lcd_command") != st.session_state.last_lcd_result_cmd:
                    st.session_state.current_lcd_command = st.session_state.last_lcd_result_cmd

            analyze_btn = st.button(
                "🚀 Analyze Freshness & Quality",
                type="primary",
                use_container_width=True,
                disabled=bool(st.session_state.get("analysis_running", False)),
                on_click=_queue_freshness_analysis,
                args=(temp_c, hum_pct, gas_val, sensor_source_name, sensor_status_str),
            )

        # Trigger analysis when button is pressed
        analysis_pending = bool(st.session_state.get("analysis_pending", False))
        browser_snapshot_paused = bool((web_serial_data or {}).get("analysis_paused"))
        waiting_for_browser_pause = (
            analysis_pending
            and sensor_source_name == "browser_serial"
            and bool((web_serial_data or {}).get("connected"))
            and not browser_snapshot_paused
        )

        if waiting_for_browser_pause:
            st.info("Freezing the current sensor readings for this analysis…")

        if analysis_pending and not waiting_for_browser_pause:
            # LCD State 5: ANALYZING
            lcd_analyzing_cmd = build_lcd_command("Analyzing...", "Please wait...")
            st.session_state.current_lcd_command = lcd_analyzing_cmd
            if local_hw_svc is not None:
                local_hw_svc.send_lcd_message("Analyzing...", "Please wait...")

            with st.spinner("Executing Hierarchical Gated Vision & Multimodal Reasoning..."):
                # Step 1: Run Hierarchical Vision Pipeline
                vision_res = vision_service.analyze_image(image_input)

                # Step 2: Build sensor snapshot
                sensor_reading = dict(st.session_state.get("analysis_sensor_snapshot", {}))
                if not sensor_reading:
                    sensor_reading = {
                        "temperature_c": temp_c,
                        "humidity_percent": hum_pct,
                        "gas_value": gas_val,
                        "sensor_status": sensor_status_str,
                        "sensor_scope": "environment",
                        "source": sensor_source_name,
                        "timestamp": datetime.now().isoformat(),
                    }
                st.session_state.last_analysis_sensor_snapshot = dict(sensor_reading)

                # Step 3: Fetch history
                item_name = vision_res.get("name")
                history_items = history_service.get_history_for_item(item_name, limit=3) if item_name else []

                # Step 4: Aggregate evidence packet
                evidence_packet = aggregator.aggregate(
                    vision_result=vision_res,
                    sensor_reading=sensor_reading,
                    history=history_items,
                    user_metadata={"source_file": input_source_name},
                )

                # Step 5: Multi-modal LLM Reasoning
                analysis_result = llm_service.analyze(evidence_packet, image=image_input)

                # Step 6: Save observation in SQLite
                history_service.save_observation(analysis_result, evidence_packet)

                # LCD States 6-10: RESULT update
                l1_res, l2_res = get_lcd_message_for_state(
                    "result",
                    vision_status=vision_res.get("status"),
                    freshness_status=analysis_result.freshness_status,
                )
                lcd_result_cmd = build_lcd_command(l1_res, l2_res)
                st.session_state.current_lcd_command = lcd_result_cmd
                st.session_state.last_lcd_result_cmd = lcd_result_cmd
                if local_hw_svc is not None:
                    local_hw_svc.send_lcd_message(l1_res, l2_res)

                # Step 7: Store persistent analysis state and reset chat for this new analysis
                st.session_state.analyzed_image_source = input_source_name
                st.session_state.last_vision_res = vision_res
                st.session_state.last_analysis_result = analysis_result
                st.session_state.chat_context = {
                    "detected_food": vision_res.get("name"),
                    "category": vision_res.get("category"),
                    "confidence": vision_res.get("confidence"),
                    "status": vision_res.get("status"),
                    "visual_description": vision_res.get("visual_description"),
                    "overall_visual_summary": vision_res.get("overall_visual_summary"),
                    "items": vision_res.get("items", []),
                    "vision_source": vision_res.get("source", "gemini"),
                    "temperature_c": sensor_reading.get("temperature_c"),
                    "humidity_percent": sensor_reading.get("humidity_percent"),
                    "gas_value": sensor_reading.get("gas_value"),
                    "sensor_source": sensor_reading.get("source", sensor_source_name),
                    "freshness_status": analysis_result.freshness_status,
                    "estimated_shelf_life": f"{analysis_result.estimated_remaining_freshness.value} {analysis_result.estimated_remaining_freshness.unit}",
                    "shelf_life_range": analysis_result.estimated_remaining_freshness.range_description,
                    "freshness_reasoning": analysis_result.reasoning_summary,
                    "visual_observations": analysis_result.visual_observations,
                    "environmental_context": analysis_result.environmental_context,
                    "storage_recommendations": analysis_result.recommendations,
                    "uncertainty_factors": analysis_result.uncertainty_factors,
                }
                st.session_state.chat_history = []
                st.session_state.analysis_pending = False
                st.session_state.analysis_running = False
                st.rerun()

        with col_right:
            st.subheader("2️⃣ Analysis & Results")

            has_analysis = (
                st.session_state.get("analyzed_image_source") == input_source_name
                and "last_vision_res" in st.session_state
                and "last_analysis_result" in st.session_state
            )

            if has_analysis:
                vision_res = st.session_state.last_vision_res
                analysis_result = st.session_state.last_analysis_result
                status = vision_res.get("status")
                source_type = vision_res.get("source", "local_siglip_fallback")
                items = vision_res.get("items", [])

                # Quota Exhaustion Alert
                if vision_res.get("quota_exhausted") or vision_res.get("fallback_reason") == "gemini_quota_exhausted":
                    st.warning("⚠️ **Gemini quota temporarily exhausted — using Local Fallback Vision.**")

                # Source Badge
                if source_type == "gemini":
                    source_badge = '<span class="source-tag">✨ Gemini Multimodal Vision</span>'
                elif source_type == "multimodal_openai":
                    source_badge = '<span class="source-tag">✨ OpenAI Vision</span>'
                else:
                    source_badge = '<span class="source-tag-fallback">⚙️ Local Fallback Vision (SigLIP)</span>'

                if status == "not_food":
                    st.markdown(f"""
                    <div class="status-card-notfood">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                            <h3 style="margin:0; color:#FCA5A5;">⛔ Not a Fruit or Food Item</h3>
                            {source_badge}
                        </div>
                        <p style="font-size:1.05rem;">{vision_res.get("visual_description")}</p>
                        <p style="font-size:0.85rem; opacity:0.8;">Confidence: <b>{int(vision_res.get('confidence', 0)*100)}%</b> | Latency: <b>{vision_res.get('latency_ms', 0)} ms</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.info("💡 **Notice**: The vision stage identified this image as a non-food object. No food shelf-life analysis or consumption recommendations are applicable.")

                elif status == "uncertain":
                    st.markdown(f"""
                    <div class="status-card-uncertain">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                            <h3 style="margin:0; color:#FDE68A;">⚠️ Unidentified / Out-of-Distribution Food Item</h3>
                            {source_badge}
                        </div>
                        <p style="font-size:1.05rem;">{vision_res.get("visual_description")}</p>
                        <p style="font-size:0.85rem; opacity:0.8;">Confidence: <b>{int(vision_res.get('confidence', 0)*100)}%</b> | Category: <b>{vision_res.get('category', 'food')}</b> | Latency: <b>{vision_res.get('latency_ms', 0)} ms</b></p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Render Freshness Badge & Synthesis
                    badge_html = get_freshness_badge_html(analysis_result.freshness_status)
                    st.markdown(f"**Freshness Status:** {badge_html}", unsafe_allow_html=True)

                    est = analysis_result.estimated_remaining_freshness
                    st.markdown(f"""
                    <div class="telemetry-box" style="margin-top: 10px;">
                        <div style="font-size: 0.85rem; color: #94A3B8;">ESTIMATED GENERAL SHELF LIFE</div>
                        <div class="metric-value">⏳ {est.value} {est.unit}</div>
                        <div style="font-size: 0.95rem; color: #CBD5E1; margin-top: 4px;">{est.range_description}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"**💡 Synthesis & General Food Advice:**")
                    st.write(analysis_result.reasoning_summary)

                    if analysis_result.recommendations:
                        st.markdown("**📋 Storage Recommendations:**")
                        for rec in analysis_result.recommendations:
                            st.markdown(f"- {rec}")

                else:  # recognized
                    # Check if multi-item or single-item
                    if len(items) > 1:
                        st.markdown(f"""
                        <div class="status-card-recognized">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                                <h3 style="margin:0; color:#6EE7B7;">🍱 Multi-Item Scene: {len(items)} Items Detected</h3>
                                {source_badge}
                            </div>
                            <p style="font-size:1.05rem;">{vision_res.get("overall_visual_summary") or vision_res.get("visual_description")}</p>
                            <p style="font-size:0.85rem; opacity:0.8;">Enumerated Items: <b>{len(items)} distinct food items</b> | Latency: <b>{vision_res.get('latency_ms', 0)} ms</b></p>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown("#### 🔍 Individual Item Breakdown")
                        for it in items:
                            it_name = it.get("name", "item").replace("_", " ").title()
                            it_cat = it.get("category", "food").replace("_", " ").title()
                            it_cond = it.get("condition", "fresh").title()
                            it_spoil = it.get("is_spoiled", False)
                            it_conf = int(it.get("confidence", 0))
                            it_stat = it.get("freshness_status", "FRESH")
                            card_cls = "item-card-spoiled" if it_spoil else "item-card"

                            spoil_badge = "🔴 <b>VISIBLY SPOILED / MOLDY</b>" if it_spoil else "🟢 <b>USABLE</b>"

                            st.markdown(f"""
                            <div class="{card_cls}">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <span style="font-size:1.1rem; font-weight:700; color:#F8FAFC;">{it.get('item_id', 1)}. {it_name}</span>
                                    <span style="font-size:0.8rem; background:#334155; padding:2px 8px; border-radius:4px; color:#94A3B8;">{it_cat}</span>
                                </div>
                                <div style="margin-top:6px; font-size:0.9rem; color:#CBD5E1;">
                                    <b>Condition:</b> {it_cond} &nbsp;|&nbsp; <b>Confidence:</b> {it_conf}% &nbsp;|&nbsp; <b>Status:</b> {spoil_badge}
                                </div>
                                <div style="margin-top:6px; font-size:0.85rem; color:#94A3B8;">
                                    {it.get('visual_description', '')}
                                </div>
                                {('<div style="margin-top:4px; font-size:0.8rem; color:#FCA5A5;">⚠️ ' + it.get('reason', '') + '</div>') if it.get('reason') else ''}
                            </div>
                            """, unsafe_allow_html=True)

                    else:
                        # Single-item hero layout
                        food_display = vision_res.get("name", "food").replace("_", " ").title()
                        cat_display = vision_res.get("category", "produce").replace("_", " ").title()
                        conf_pct = int(vision_res.get("confidence", 0) * 100)
                        single_item = items[0] if items else {}
                        cond_display = single_item.get("condition", "fresh/ripe").title() if single_item else "Fresh/Ripe"

                        st.markdown(f"""
                        <div class="status-card-recognized">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                                <h3 style="margin:0; color:#6EE7B7;">🍏 Recognized: {food_display}</h3>
                                {source_badge}
                            </div>
                            <p style="font-size:1.05rem;">{vision_res.get("visual_description")}</p>
                            <p style="font-size:0.85rem; opacity:0.8;">Category: <b>{cat_display}</b> | Visual Condition: <b>{cond_display}</b> | Visual Confidence: <b>{conf_pct}%</b> | Latency: <b>{vision_res.get('latency_ms', 0)} ms</b></p>
                        </div>
                        """, unsafe_allow_html=True)

                    badge_html = get_freshness_badge_html(analysis_result.freshness_status)
                    st.markdown(f"**Freshness Status:** {badge_html}", unsafe_allow_html=True)

                    est = analysis_result.estimated_remaining_freshness
                    st.markdown(f"""
                    <div class="telemetry-box" style="margin-top: 10px;">
                        <div style="font-size: 0.85rem; color: #94A3B8;">ESTIMATED REMAINING SHELF LIFE</div>
                        <div class="metric-value">⏳ {est.value} {est.unit}</div>
                        <div style="font-size: 0.95rem; color: #CBD5E1; margin-top: 4px;">{est.range_description}</div>
                        <div style="font-size: 0.8rem; color: #64748B; margin-top: 3px;">Confidence: <b>{est.confidence.upper()}</b></div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"**💡 Freshness Synthesis:**")
                    st.write(analysis_result.reasoning_summary)

                    if analysis_result.visual_observations:
                        st.markdown("**👁️ Visual Inspection:**")
                        for obs in analysis_result.visual_observations:
                            st.markdown(f"- {obs}")

                    if analysis_result.recommendations:
                        st.markdown("**📋 Storage Recommendations:**")
                        for rec in analysis_result.recommendations:
                            st.markdown(f"- {rec}")

                    with st.expander("⚠️ Uncertainty Factors & Food Safety Disclaimer"):
                        for unc in analysis_result.uncertainty_factors:
                            st.markdown(f"- {unc}")
                        st.caption("Disclaimer: This tool assesses visible freshness and environmental storage risks. It is not a microbiological laboratory food safety test.")
            else:
                st.info("👈 Click **Analyze Freshness & Quality** to run the multimodal inspection.")

    # -------------------------------------------------------------------------
    # CHAT ABOUT ANALYZED FOOD
    # -------------------------------------------------------------------------
    if (
        st.session_state.get("analyzed_image_source") == input_source_name
        and "chat_context" in st.session_state
        and st.session_state.chat_context is not None
    ):
        st.markdown("---")
        st.subheader("💬 Ask About This Food")

        ctx = st.session_state.chat_context

        # Initialize chat history in session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Starter question suggestions
        if not st.session_state.chat_history:
            st.markdown(
                '<div style="font-size:0.9rem; color:#94A3B8; margin-bottom:0.5rem;">'
                "Try asking:</div>",
                unsafe_allow_html=True,
            )
            starter_cols = st.columns(3)
            starters = [
                "Why is this freshness status given?",
                "How should I store this?",
                "What did the model detect?",
                "What do the sensor readings mean?",
                "Tell me about this food.",
                "How long might it last?",
            ]
            for idx, q in enumerate(starters):
                with starter_cols[idx % 3]:
                    if st.button(q, key=f"starter_{idx}"):
                        st.session_state.pending_chat_question = q

        # Display conversation history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                st.chat_message("assistant").write(msg["content"])

        # Check for pending starter question
        pending_q = st.session_state.pop("pending_chat_question", None)

        # Chat input
        user_question = st.chat_input("Ask a question about the analyzed food...")

        # Use pending starter if no typed question
        if pending_q and not user_question:
            user_question = pending_q

        if user_question:
            st.chat_message("user").write(user_question)

            with st.spinner("Thinking..."):
                answer = llm_service.chat_about_analysis(
                    analysis_context=ctx,
                    conversation_history=st.session_state.chat_history,
                    user_question=user_question,
                )

            st.chat_message("assistant").write(answer)

            st.session_state.chat_history.append({"role": "user", "content": user_question})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

    elif image_input is None:
        pass  # No image loaded yet, nothing to show
    elif "chat_context" not in st.session_state:
        pass  # Analysis not yet run

    # -------------------------------------------------------------------------
    # TAB 4: HISTORY & CSV EXPORT
    # -------------------------------------------------------------------------
    with tab_history:
        st.subheader("📊 Scan History & Observation Database")
        search_query = st.text_input("🔍 Filter by item name", value="")

        records = history_service.get_recent_observations(limit=50)

        if records:
            df = pd.DataFrame(records)
            if search_query:
                df = df[df["item_name"].str.contains(search_query, case=False, na=False)]

            display_cols = [c for c in ["id", "timestamp", "item_name", "freshness_status", "detection_confidence", "temperature_c", "humidity_percent", "gas_value"] if c in df.columns]
            st.dataframe(df[display_cols], use_container_width=True)

            # Export CSV
            csv_path = history_service.export_to_csv()
            if os.path.exists(csv_path):
                with open(csv_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Full History CSV",
                        data=f.read(),
                        file_name="food_freshness_history.csv",
                        mime="text/csv",
                    )
        else:
            st.info("No scan observations recorded yet. Run your first analysis above!")


if __name__ == "__main__":
    main()
