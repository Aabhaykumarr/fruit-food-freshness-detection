"""
Centralized 16x2 LCD Message Formatter and State Mapper.

Ensures that every line sent to the Arduino 16x2 character LCD is strictly
validated, sanitized, and truncated to at most 16 characters.

States supported:
  1. IDLE: "Put food near" / "Upload image"
  2. ARDUINO CONNECTING: "Connect Arduino" / "Select COM port"
  3. ARDUINO CONNECTED: "Arduino: ONLINE" / "Sensors reading"
  4. SENSOR READY: "Temp: XX.X C" / "Hum: XX.X %"
  5. ANALYZING: "Analyzing..." / "Please wait..."
  6. FRESH RESULT: "Status: FRESH" / "Check report"
  7. SPOILED RESULT: "Status: SPOILED" / "Do not consume"
  8. OTHER FRESHNESS: "Stat: MOD FRESH", "Status: AT RISK", "Status: UNKNOWN"
  9. NOT FOOD: "Not food/fruit" / "Try again"
 10. UNCERTAIN FOOD: "Food uncertain" / "Check screen"
"""

from typing import Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

MAX_LCD_LINE_LENGTH = 16


def sanitize_lcd_line(text: Any, max_len: int = MAX_LCD_LINE_LENGTH) -> str:
    """
    Sanitize and truncate a string so it strictly fits on one line of a 16x2 LCD.

    - Replaces newlines, carriage returns, tabs, and pipes with single spaces.
    - Strips leading/trailing whitespace.
    - Truncates to at most `max_len` characters (default 16).
    - Guarantees len(result) <= 16.
    """
    if text is None:
        return ""

    s = str(text)
    # Replace control characters and protocol delimiter '|'
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ").replace("|", " ")
    # Collapse multiple consecutive spaces
    s = " ".join(s.split())
    # Truncate to maximum characters
    truncated = s[:max_len]

    # Hard invariant assertion for testing & runtime safety
    assert len(truncated) <= max_len, f"LCD line exceeded {max_len} chars: '{truncated}'"
    return truncated


def format_lcd_message(line1: Any, line2: Any = "") -> Tuple[str, str]:
    """
    Format and validate two lines for a 16x2 LCD display.
    Returns (line1, line2) where both are guaranteed to be <= 16 characters.
    """
    l1 = sanitize_lcd_line(line1, MAX_LCD_LINE_LENGTH)
    l2 = sanitize_lcd_line(line2, MAX_LCD_LINE_LENGTH)
    return l1, l2


def build_lcd_command(line1: Any, line2: Any = "") -> str:
    """
    Build the standard serial command payload to transmit to Arduino.
    Protocol: LCD:Line1Text|Line2Text\n
    """
    l1, l2 = format_lcd_message(line1, line2)
    return f"LCD:{l1}|{l2}\n"


def get_lcd_message_for_state(
    state: str,
    temp_c: Optional[float] = None,
    hum_pct: Optional[float] = None,
    vision_status: Optional[str] = None,
    freshness_status: Optional[str] = None,
    item_name: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Centralized state-to-LCD message mapper.
    Every mapped message is guaranteed to have both lines <= 16 characters.

    Args:
        state: One of 'idle', 'connecting', 'connected', 'sensor_ready', 'analyzing', 'result'.
        temp_c: Ambient temperature in Celsius (for 'sensor_ready').
        hum_pct: Ambient humidity percentage (for 'sensor_ready').
        vision_status: Vision gating status ('recognized', 'uncertain', 'not_food').
        freshness_status: Freshness category ('FRESH', 'MODERATELY_FRESH', 'QUESTIONABLE', 'NOT_FRESH', 'SPOILED', 'NOT_FOOD', 'UNKNOWN').
        item_name: Optional food item name.

    Returns:
        Tuple of (line1, line2), each len <= 16.
    """
    state_norm = (state or "idle").lower().strip()

    # 1. State: IDLE (Standby)
    if state_norm == "idle":
        return format_lcd_message("Put food near", "Upload image")

    # 2. State: ARDUINO CONNECTING
    if state_norm == "connecting":
        return format_lcd_message("Connect Arduino", "Select COM port")

    # 3. State: ARDUINO CONNECTED
    if state_norm == "connected":
        return format_lcd_message("Arduino: ONLINE", "Sensors reading")

    # 4. State: SENSOR READY (Live readings available)
    if state_norm == "sensor_ready":
        t_str = f"Temp: {temp_c:.1f} C" if temp_c is not None else "Temp: --.- C"
        h_str = f"Hum: {hum_pct:.1f} %" if hum_pct is not None else "Hum: --.- %"
        return format_lcd_message(t_str, h_str)

    # 5. State: ANALYZING
    if state_norm == "analyzing":
        return format_lcd_message("Analyzing...", "Please wait...")

    # 6-10. State: RESULT (Gated by Vision & Freshness Status)
    if state_norm == "result":
        v_stat = (vision_status or "").lower().strip()
        f_stat = (freshness_status or "").upper().strip()

        # State 9: NOT FOOD
        if v_stat == "not_food" or f_stat == "NOT_FOOD":
            return format_lcd_message("Not food/fruit", "Try again")

        # State 10: UNCERTAIN FOOD
        if v_stat == "uncertain":
            return format_lcd_message("Food uncertain", "Check screen")

        # State 6: FRESH RESULT
        if f_stat == "FRESH":
            return format_lcd_message("Status: FRESH", "Check report")

        # State 7: SPOILED / NOT_FRESH RESULT
        if f_stat in ("NOT_FRESH", "SPOILED"):
            return format_lcd_message("Status: SPOILED", "Do not consume")

        # State 8: MODERATELY FRESH
        if f_stat == "MODERATELY_FRESH":
            return format_lcd_message("Stat: MOD FRESH", "Consume soon")

        # State 8: QUESTIONABLE / AT RISK
        if f_stat in ("QUESTIONABLE", "AT_RISK"):
            return format_lcd_message("Status: AT RISK", "Inspect closely")

        # State 8: UNKNOWN / OTHER
        if f_stat == "UNKNOWN":
            return format_lcd_message("Status: UNKNOWN", "Check report")

        # Fallback with status
        status_disp = f_stat[:10] if f_stat else "DONE"
        return format_lcd_message(f"Status: {status_disp}", "Check report")

    # Default fallback
    return format_lcd_message("Freshness System", "Ready to scan")
