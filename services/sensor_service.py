"""
Sensor service — handles Arduino USB serial communication with fallback mock mode.

Supports:
  1. JSON Lines: {"temperature_c": 24.5, "humidity_percent": 65.2, "gas_value": 312}
  2. Comma-separated: 24.5, 65.2, 312
  3. Key-Value text: Temp: 24.5 Hum: 65.2 Gas: 312
  4. Mock Mode for offline development.
"""

import json
import time
import random
import re
import threading
from datetime import datetime
from typing import Dict, Any, Optional
import logging

from utils.validation import validate_sensor_reading
from utils.lcd_helper import build_lcd_command, format_lcd_message

logger = logging.getLogger(__name__)


def parse_raw_serial_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Tolerant parser for Arduino serial output in various formats:
      - JSON: {"temperature_c": 24.5, "humidity_percent": 65.2, "gas_value": 312}
      - Comma-separated: 24.5, 65.2, 312
      - Key-value text: T: 24.5 H: 65.2 G: 312 or Temperature: 24.5 Humidity: 65.2
    """
    clean = line.strip()
    if not clean:
        return None

    # 1. Try JSON
    if clean.startswith("{") and clean.endswith("}"):
        try:
            data = json.loads(clean)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    # 2. Try regex extraction of numbers from key-value pairs
    temp = None
    hum = None
    gas = None

    # Temperature match: temp: 24.5 or t: 24.5 or 24.5 C
    temp_match = re.search(r"(?:temp(?:erature)?|t)\s*[:=]?\s*([+-]?\d+(?:\.\d+)?)", clean, re.IGNORECASE)
    if temp_match:
        try:
            temp = float(temp_match.group(1))
        except ValueError:
            pass

    # Humidity match: hum: 65.2 or h: 65.2 or 65.2 %
    hum_match = re.search(r"(?:hum(?:idity)?|h)\s*[:=]?\s*([+-]?\d+(?:\.\d+)?)", clean, re.IGNORECASE)
    if hum_match:
        try:
            hum = float(hum_match.group(1))
        except ValueError:
            pass

    # Gas match: gas: 312 or g: 312 or voc: 312
    gas_match = re.search(r"(?:gas|voc|g|air)\s*[:=]?\s*([+-]?\d+(?:\.\d+)?)", clean, re.IGNORECASE)
    if gas_match:
        try:
            gas = float(gas_match.group(1))
        except ValueError:
            pass

    if temp is not None or hum is not None or gas is not None:
        return {
            "temperature_c": temp,
            "humidity_percent": hum,
            "gas_value": gas,
        }

    # 3. Try comma-separated floats (e.g. 24.5, 65.2, 312)
    parts = clean.split(",")
    if len(parts) >= 2:
        try:
            nums = [float(p.strip()) for p in parts if p.strip()]
            if len(nums) >= 2:
                return {
                    "temperature_c": nums[0],
                    "humidity_percent": nums[1],
                    "gas_value": nums[2] if len(nums) > 2 else None,
                }
        except ValueError:
            pass

    return None


class SensorService:
    """
    Dedicated sensor acquisition layer.
    Supports both real Arduino over USB serial and mock mode for offline development.
    Runs a background reading thread to keep the latest reading up-to-date without blocking.
    """

    def __init__(
        self,
        mock_mode: bool = True,
        port: str = "auto",
        baud_rate: int = 9600,
        timeout: float = 2.0,
        stale_seconds: float = 30.0,
    ):
        self.mock_mode = mock_mode
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.stale_seconds = stale_seconds

        self._latest_raw: Optional[Dict[str, Any]] = None
        self._last_received_time: float = 0.0
        self._button_pressed: bool = False
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._serial_conn = None

    def start(self):
        """Start the background sensor reading thread."""
        self._running = True
        if self.mock_mode:
            logger.info("SensorService started in MOCK mode")
            self._thread = threading.Thread(target=self._mock_loop, daemon=True)
        else:
            logger.info("SensorService started in ARDUINO mode (port=%s, baud=%d)", self.port, self.baud_rate)
            self._thread = threading.Thread(target=self._serial_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the background reader and close any serial connection."""
        self._running = False
        if self._serial_conn is not None:
            try:
                self._serial_conn.close()
            except Exception:
                pass
            self._serial_conn = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        logger.info("SensorService stopped")

    def get_latest_reading(self) -> Dict[str, Any]:
        """
        Get the latest validated sensor reading.
        Detects stale data and returns 'unavailable' if data is too old.
        """
        if self._last_received_time > 0:
            age = time.time() - self._last_received_time
            if age > self.stale_seconds:
                logger.warning("Sensor reading is stale (age=%.1fs > threshold=%.1fs)", age, self.stale_seconds)
                return {
                    "temperature_c": None,
                    "humidity_percent": None,
                    "gas_value": None,
                    "sensor_status": "stale",
                    "sensor_scope": "environment",
                    "timestamp": datetime.now().isoformat(),
                    "source": "none",
                }

        return validate_sensor_reading(self._latest_raw)

    def capture_snapshot(self) -> Dict[str, Any]:
        """
        Capture an immediate snapshot of sensor telemetry at trigger time.
        Guarantees that telemetry is recorded only when food is placed at the sensor.
        """
        reading = self.get_latest_reading()
        reading["captured_at"] = datetime.now().isoformat()
        reading["is_triggered_reading"] = True
        return reading

    def check_button_pressed(self) -> bool:
        """
        Check if an Arduino hardware push-button event was received.
        Clears the flag and returns True if pressed.
        """
        if self._button_pressed:
            self._button_pressed = False
            return True
        return False

    def send_lcd_message(self, line1: str, line2: str = ""):
        """
        Send a 2-line message to a connected Arduino LCD screen over serial.
        Protocol: LCD:Line1Text|Line2Text\n
        Guarantees both lines are <= 16 characters.
        """
        l1, l2 = format_lcd_message(line1, line2)
        msg = build_lcd_command(l1, l2)

        if self._serial_conn is not None and not self.mock_mode:
            try:
                self._serial_conn.write(msg.encode("utf-8"))
                self._serial_conn.flush()
                logger.debug("Sent LCD message: %s", msg.strip())
            except Exception as e:
                logger.warning("Failed to send LCD message: %s", e)
        else:
            logger.debug("[MOCK LCD] %s | %s", l1, l2)

    def _mock_loop(self):
        """Generate realistic mock sensor readings with slight fluctuations."""
        base_temp = 24.0
        base_hum = 62.0
        base_gas = 310.0

        while self._running:
            temp = round(base_temp + random.uniform(-1.5, 1.5), 1)
            hum = round(base_hum + random.uniform(-3.0, 3.0), 1)
            gas = round(base_gas + random.uniform(-15.0, 15.0), 1)

            self._latest_raw = {
                "temperature_c": temp,
                "humidity_percent": hum,
                "gas_value": gas,
                "timestamp": datetime.now().isoformat(),
                "source": "mock",
                "sensor_scope": "environment",
            }
            self._last_received_time = time.time()
            time.sleep(1.0)

    def _serial_loop(self):
        """Read lines from Arduino USB serial with auto-reconnect."""
        import serial
        import serial.tools.list_ports

        while self._running:
            if self._serial_conn is None:
                port_to_use = self.port
                if port_to_use == "auto":
                    port_to_use = self._auto_discover_port()

                if port_to_use is None or port_to_use == "none":
                    time.sleep(3.0)
                    continue

                try:
                    self._serial_conn = serial.Serial(
                        port=port_to_use,
                        baudrate=self.baud_rate,
                        timeout=self.timeout,
                    )
                    logger.info("Connected to Arduino on %s", port_to_use)
                    time.sleep(2.0)
                except Exception as e:
                    logger.error("Failed to connect to serial port %s: %s", port_to_use, e)
                    self._serial_conn = None
                    self._latest_raw = None
                    time.sleep(3.0)
                    continue

            try:
                line = self._serial_conn.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    # Check for button click event
                    if "button" in line.lower() or line.upper() == "BUTTON_PRESSED" or line.upper() == "TRIGGER":
                        self._button_pressed = True
                        logger.info("Arduino hardware button press detected!")
                        continue

                    parsed = parse_raw_serial_line(line)
                    if parsed is not None:
                        if parsed.get("button") or parsed.get("event") == "button":
                            self._button_pressed = True
                            logger.info("Arduino hardware button press detected in JSON!")
                        parsed["source"] = "arduino"
                        parsed["sensor_scope"] = "environment"
                        if "timestamp" not in parsed:
                            parsed["timestamp"] = datetime.now().isoformat()
                        self._latest_raw = parsed
                        self._last_received_time = time.time()
                    else:
                        logger.debug("Received unparsed line from serial: %s", line)
            except Exception as e:
                logger.error("Serial read error: %s. Reconnecting...", e)
                try:
                    self._serial_conn.close()
                except Exception:
                    pass
                self._serial_conn = None
                self._latest_raw = None
                time.sleep(2.0)

    def _auto_discover_port(self) -> Optional[str]:
        """Attempt to find a connected Arduino / USB serial device."""
        import serial.tools.list_ports
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            desc = (p.description or "").lower()
            hwid = (p.hwid or "").lower()
            if "arduino" in desc or "ch340" in desc or "cp210" in desc or "usb-serial" in desc or "ftdi" in desc or "1a86" in hwid:
                logger.info("Auto-discovered Arduino port: %s (%s)", p.device, p.description)
                return p.device

        if len(ports) == 1:
            return ports[0].device

        return None
