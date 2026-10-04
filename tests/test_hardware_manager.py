"""Unit tests for Hardware Manager and Arduino Flasher."""

import pytest
from services.arduino_flasher import ArduinoFlasher
from services.hardware_manager import HardwareManager


def test_arduino_flasher_cli_detection():
    flasher = ArduinoFlasher(cli_path="auto")
    # Should locate the bundled arduino-cli if installed
    assert flasher.is_available() is True or flasher.cli_path is None


def test_arduino_flasher_detect_boards():
    flasher = ArduinoFlasher(cli_path="auto")
    # detect_boards should return a list without crashing
    boards = flasher.detect_boards()
    assert isinstance(boards, list)


def test_hardware_manager_mock_fallback():
    # Test initialization when no board is connected
    hw = HardwareManager(auto_flash=False)
    sensor_service = hw.initialize()

    assert sensor_service is not None
    reading = sensor_service.get_latest_reading()
    assert "temperature_c" in reading
    assert "humidity_percent" in reading
    assert "sensor_status" in reading

    hw.shutdown()
