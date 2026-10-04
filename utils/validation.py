"""
Sensor data validation utility.
Validates physical plausibility of environmental sensor readings.
"""

from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Physically plausible ranges for environmental sensors
VALID_TEMP_RANGE = (-40.0, 80.0)      # Celsius
VALID_HUMIDITY_RANGE = (0.0, 100.0)    # Percentage
VALID_GAS_RANGE = (0.0, 10000.0)       # Raw analog or ppm (sensor-dependent)


def validate_sensor_reading(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate a raw sensor reading dictionary.

    Returns a standardized dictionary:
    {
        "temperature_c": float | None,
        "humidity_percent": float | None,
        "gas_value": float | None,
        "sensor_status": "valid" | "invalid" | "unavailable",
        "sensor_scope": "environment",
        "timestamp": str,
        "source": str
    }
    """
    now = datetime.now().isoformat()

    if data is None:
        return {
            "temperature_c": None,
            "humidity_percent": None,
            "gas_value": None,
            "sensor_status": "unavailable",
            "sensor_scope": "environment",
            "timestamp": now,
            "source": "none",
        }

    source = data.get("source", "unknown")
    temp = data.get("temperature_c")
    hum = data.get("humidity_percent")
    gas = data.get("gas_value")
    timestamp = data.get("timestamp", now)
    sensor_scope = data.get("sensor_scope", "environment")

    # Validate temperature
    valid_temp = None
    if temp is not None:
        try:
            t = float(temp)
            if VALID_TEMP_RANGE[0] <= t <= VALID_TEMP_RANGE[1]:
                valid_temp = round(t, 2)
            else:
                logger.warning("Temperature out of physical range: %s", t)
        except (ValueError, TypeError):
            logger.warning("Invalid temperature value: %s", temp)

    # Validate humidity
    valid_hum = None
    if hum is not None:
        try:
            h = float(hum)
            if VALID_HUMIDITY_RANGE[0] <= h <= VALID_HUMIDITY_RANGE[1]:
                valid_hum = round(h, 2)
            else:
                logger.warning("Humidity out of physical range: %s", h)
        except (ValueError, TypeError):
            logger.warning("Invalid humidity value: %s", hum)

    # Validate gas
    valid_gas = None
    if gas is not None:
        try:
            g = float(gas)
            if VALID_GAS_RANGE[0] <= g <= VALID_GAS_RANGE[1]:
                valid_gas = round(g, 2)
            else:
                logger.warning("Gas value out of range: %s", g)
        except (ValueError, TypeError):
            logger.warning("Invalid gas value: %s", gas)

    # Determine overall status
    if valid_temp is not None or valid_hum is not None:
        status = "valid"
    elif temp is None and hum is None and gas is None:
        status = "unavailable"
    else:
        status = "invalid"

    return {
        "temperature_c": valid_temp,
        "humidity_percent": valid_hum,
        "gas_value": valid_gas,
        "sensor_status": status,
        "sensor_scope": sensor_scope,
        "timestamp": timestamp,
        "source": source,
    }
