"""
Unit tests for 16x2 LCD helper and state mapping.
Verifies that:
  1. Every single line is <= 16 characters.
  2. All 10 required states format correctly.
  3. Sanitization properly cleans newlines, tabs, and pipes.
  4. Protocol commands are properly generated.
"""

import pytest
from utils.lcd_helper import (
    sanitize_lcd_line,
    format_lcd_message,
    build_lcd_command,
    get_lcd_message_for_state,
    MAX_LCD_LINE_LENGTH,
)


def _assert_line_valid(line: str):
    """Helper to assert that an LCD line is valid and <= 16 characters."""
    assert isinstance(line, str)
    assert len(line) <= MAX_LCD_LINE_LENGTH, f"Line exceeds 16 chars ({len(line)}): '{line}'"
    assert "\n" not in line
    assert "\r" not in line
    assert "|" not in line


# =============================================================================
# 1. Truncation and Sanitization Tests
# =============================================================================

def test_sanitize_short_string():
    res = sanitize_lcd_line("Hello World")
    assert res == "Hello World"
    _assert_line_valid(res)


def test_sanitize_exact_16_chars():
    exact = "1234567890123456"
    assert len(exact) == 16
    res = sanitize_lcd_line(exact)
    assert res == exact
    _assert_line_valid(res)


def test_sanitize_long_string_truncated():
    long_str = "This is a very long string that will definitely exceed sixteen characters"
    res = sanitize_lcd_line(long_str)
    assert len(res) == 16
    assert res == "This is a very l"
    _assert_line_valid(res)


def test_sanitize_newlines_and_pipes():
    dirty = "Line1|Line2\nTest\r\tEnd"
    res = sanitize_lcd_line(dirty)
    _assert_line_valid(res)
    assert "|" not in res
    assert "\n" not in res


def test_sanitize_none_and_empty():
    assert sanitize_lcd_line(None) == ""
    assert sanitize_lcd_line("") == ""
    assert sanitize_lcd_line("   ") == ""


def test_format_lcd_message():
    l1, l2 = format_lcd_message("Status: MOD FRESH", "Consume soon")
    _assert_line_valid(l1)
    _assert_line_valid(l2)
    assert l1 == "Status: MOD FRES" or len(l1) <= 16
    assert l2 == "Consume soon"


def test_build_lcd_command():
    cmd = build_lcd_command("Put food near", "Upload image")
    assert cmd.startswith("LCD:")
    assert cmd.endswith("\n")
    assert "Put food near|Upload image" in cmd


# =============================================================================
# 2. Required 10 LCD States Tests
# =============================================================================

def test_state_1_idle():
    l1, l2 = get_lcd_message_for_state("idle")
    _assert_line_valid(l1)
    _assert_line_valid(l2)
    assert l1 == "Put food near"
    assert l2 == "Upload image"


def test_state_2_connecting():
    l1, l2 = get_lcd_message_for_state("connecting")
    _assert_line_valid(l1)
    _assert_line_valid(l2)
    assert l1 == "Connect Arduino"
    assert l2 == "Select COM port"


def test_state_3_connected():
    l1, l2 = get_lcd_message_for_state("connected")
    _assert_line_valid(l1)
    _assert_line_valid(l2)
    assert l1 == "Arduino: ONLINE"
    assert l2 == "Sensors reading"


def test_state_4_sensor_ready():
    l1, l2 = get_lcd_message_for_state("sensor_ready", temp_c=24.5, hum_pct=58.2)
    _assert_line_valid(l1)
    _assert_line_valid(l2)
    assert l1 == "Temp: 24.5 C"
    assert l2 == "Hum: 58.2 %"


def test_state_4_sensor_ready_none():
    l1, l2 = get_lcd_message_for_state("sensor_ready", temp_c=None, hum_pct=None)
    _assert_line_valid(l1)
    _assert_line_valid(l2)
    assert "Temp:" in l1
    assert "Hum:" in l2


def test_state_5_analyzing():
    l1, l2 = get_lcd_message_for_state("analyzing")
    _assert_line_valid(l1)
    _assert_line_valid(l2)
    assert l1 == "Analyzing..."
    assert l2 == "Please wait..."


def test_state_6_fresh_result():
    l1, l2 = get_lcd_message_for_state("result", freshness_status="FRESH", vision_status="recognized")
    _assert_line_valid(l1)
    _assert_line_valid(l2)
    assert l1 == "Status: FRESH"
    assert l2 == "Check report"


def test_state_7_spoiled_result():
    l1, l2 = get_lcd_message_for_state("result", freshness_status="NOT_FRESH", vision_status="recognized")
    _assert_line_valid(l1)
    _assert_line_valid(l2)
    assert l1 == "Status: SPOILED"
    assert l2 == "Do not consume"

    l1_spoil, l2_spoil = get_lcd_message_for_state("result", freshness_status="SPOILED", vision_status="recognized")
    _assert_line_valid(l1_spoil)
    _assert_line_valid(l2_spoil)
    assert l1_spoil == "Status: SPOILED"
    assert l2_spoil == "Do not consume"


def test_state_8_moderately_fresh():
    l1, l2 = get_lcd_message_for_state("result", freshness_status="MODERATELY_FRESH", vision_status="recognized")
    _assert_line_valid(l1)
    _assert_line_valid(l2)
    assert l1 == "Stat: MOD FRESH"
    assert l2 == "Consume soon"


def test_state_8_questionable_and_unknown():
    l1_q, l2_q = get_lcd_message_for_state("result", freshness_status="QUESTIONABLE", vision_status="recognized")
    _assert_line_valid(l1_q)
    _assert_line_valid(l2_q)
    assert l1_q == "Status: AT RISK"

    l1_u, l2_u = get_lcd_message_for_state("result", freshness_status="UNKNOWN", vision_status="recognized")
    _assert_line_valid(l1_u)
    _assert_line_valid(l2_u)
    assert l1_u == "Status: UNKNOWN"


def test_state_9_not_food():
    l1, l2 = get_lcd_message_for_state("result", vision_status="not_food")
    _assert_line_valid(l1)
    _assert_line_valid(l2)
    assert l1 == "Not food/fruit"
    assert l2 == "Try again"

    # Also if freshness_status is NOT_FOOD
    l1_nf, l2_nf = get_lcd_message_for_state("result", freshness_status="NOT_FOOD")
    _assert_line_valid(l1_nf)
    _assert_line_valid(l2_nf)
    assert l1_nf == "Not food/fruit"
    assert l2_nf == "Try again"


def test_state_10_uncertain_food():
    l1, l2 = get_lcd_message_for_state("result", vision_status="uncertain", freshness_status="UNKNOWN")
    _assert_line_valid(l1)
    _assert_line_valid(l2)
    assert l1 == "Food uncertain"
    assert l2 == "Check screen"
