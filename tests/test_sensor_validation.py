"""Unit tests for sensor data validation."""

import pytest
from utils.validation import validate_sensor_reading


def test_valid_reading():
    raw = {
        "temperature_c": 24.5,
        "humidity_percent": 65.2,
        "gas_value": 312,
        "source": "arduino",
    }
    result = validate_sensor_reading(raw)
    assert result["temperature_c"] == 24.5
    assert result["humidity_percent"] == 65.2
    assert result["gas_value"] == 312.0
    assert result["sensor_status"] == "valid"
    assert result["source"] == "arduino"


def test_out_of_range_reading():
    raw = {
        "temperature_c": 150.0,  # Physically impossible for ambient
        "humidity_percent": 120.0,  # Impossible percentage
        "gas_value": 300,
    }
    result = validate_sensor_reading(raw)
    assert result["temperature_c"] is None
    assert result["humidity_percent"] is None
    assert result["gas_value"] == 300.0
    assert result["sensor_status"] == "invalid"


def test_none_reading():
    result = validate_sensor_reading(None)
    assert result["temperature_c"] is None
    assert result["humidity_percent"] is None
    assert result["sensor_status"] == "unavailable"


def test_malformed_types():
    raw = {
        "temperature_c": "not-a-number",
        "humidity_percent": None,
    }
    result = validate_sensor_reading(raw)
    assert result["temperature_c"] is None
    assert result["sensor_status"] == "invalid"
