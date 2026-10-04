"""
Unit tests for Web Serial payload parsing, telemetry normalization, and LCD state synchronization.
"""

from utils.lcd_helper import get_lcd_message_for_state, build_lcd_command, format_lcd_message
from services.sensor_service import parse_raw_serial_line
from utils.validation import validate_sensor_reading


def test_web_serial_json_payload_normalization():
    """Verify that browser-parsed Web Serial JSON maps to normalized sensor dict."""
    raw_json = '{"temperature_c": 23.4, "humidity_percent": 61.2, "gas_value": 290}'
    parsed = parse_raw_serial_line(raw_json)
    assert parsed is not None
    assert parsed["temperature_c"] == 23.4
    assert parsed["humidity_percent"] == 61.2
    assert parsed["gas_value"] == 290

    validated = validate_sensor_reading(parsed)
    assert validated["sensor_status"] == "valid"
    assert validated["temperature_c"] == 23.4


def test_web_serial_disconnected_state():
    """Verify that disconnected/null telemetry marks sensors as unavailable."""
    disconnected_payload = {
        "connected": False,
        "temperature_c": None,
        "humidity_percent": None,
        "gas_value": None,
        "sensor_status": "unavailable",
    }
    validated = validate_sensor_reading(disconnected_payload)
    assert validated["sensor_status"] == "unavailable"
    assert validated["temperature_c"] is None


def test_lcd_state_synchronization_sequence():
    """Verify full end-to-end LCD sequence from connect to result."""
    # 1. Connect
    l1, l2 = get_lcd_message_for_state("connected")
    cmd_conn = build_lcd_command(l1, l2)
    assert cmd_conn == "LCD:Arduino: ONLINE|Sensors reading\n"

    # 2. Sensor ready
    l1, l2 = get_lcd_message_for_state("sensor_ready", temp_c=25.0, hum_pct=55.0)
    cmd_ready = build_lcd_command(l1, l2)
    assert cmd_ready == "LCD:Temp: 25.0 C|Hum: 55.0 %\n"

    # 3. Analyze clicked
    l1, l2 = get_lcd_message_for_state("analyzing")
    cmd_analyzing = build_lcd_command(l1, l2)
    assert cmd_analyzing == "LCD:Analyzing...|Please wait...\n"

    # 4. Fresh result
    l1, l2 = get_lcd_message_for_state("result", vision_status="recognized", freshness_status="FRESH")
    cmd_fresh = build_lcd_command(l1, l2)
    assert cmd_fresh == "LCD:Status: FRESH|Check report\n"

    # 5. Spoiled result
    l1, l2 = get_lcd_message_for_state("result", vision_status="recognized", freshness_status="NOT_FRESH")
    cmd_spoiled = build_lcd_command(l1, l2)
    assert cmd_spoiled == "LCD:Status: SPOILED|Do not consume\n"

    # 6. Not Food
    l1, l2 = get_lcd_message_for_state("result", vision_status="not_food")
    cmd_notfood = build_lcd_command(l1, l2)
    assert cmd_notfood == "LCD:Not food/fruit|Try again\n"

    # 7. Uncertain Food
    l1, l2 = get_lcd_message_for_state("result", vision_status="uncertain")
    cmd_uncertain = build_lcd_command(l1, l2)
    assert cmd_uncertain == "LCD:Food uncertain|Check screen\n"
