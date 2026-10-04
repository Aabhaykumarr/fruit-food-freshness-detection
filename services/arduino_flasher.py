"""
Arduino Auto-Flasher Service.

Detects installed arduino-cli, discovers connected boards (including CH340/FTDI/CP210 clones),
and automatically compiles and uploads the sensor reader sketch.
"""

import os
import shutil
import subprocess
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Standard known search locations for arduino-cli
CANDIDATE_PATHS = [
    # Arduino IDE 2.x bundled CLI
    r"C:\Program Files\Arduino IDE\resources\app\lib\backend\resources\arduino-cli.exe",
    r"C:\Program Files (x86)\Arduino IDE\resources\app\lib\backend\resources\arduino-cli.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Arduino IDE\resources\app\lib\backend\resources\arduino-cli.exe"),
    # Standalone arduino-cli installations
    r"C:\Program Files\arduino-cli\arduino-cli.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Arduino15\arduino-cli.exe"),
    "/usr/local/bin/arduino-cli",
    "/usr/bin/arduino-cli",
]


class ArduinoFlasher:
    """
    Manages board detection, sketch compilation, and firmware uploading via arduino-cli.
    """

    def __init__(self, cli_path: str = "auto", default_fqbn: str = "arduino:avr:uno"):
        self.default_fqbn = default_fqbn
        self.cli_path = self._locate_cli(cli_path)
        if self.cli_path:
            logger.info("Found arduino-cli at: %s", self.cli_path)
        else:
            logger.warning("arduino-cli not found. Auto-upload will be skipped if Arduino is connected.")

    def _locate_cli(self, configured_path: str) -> Optional[str]:
        """Find the arduino-cli binary."""
        if configured_path and configured_path != "auto" and os.path.isfile(configured_path):
            return configured_path

        # Check system PATH
        path_in_env = shutil.which("arduino-cli")
        if path_in_env:
            return path_in_env

        # Check standard Windows/Linux locations
        for candidate in CANDIDATE_PATHS:
            if os.path.isfile(candidate):
                return candidate

        return None

    def is_available(self) -> bool:
        """Returns True if arduino-cli is executable."""
        return self.cli_path is not None and os.path.isfile(self.cli_path)

    def detect_boards(self) -> List[Dict[str, Any]]:
        """
        Scan USB serial ports and return detected Arduino boards.
        Supports official Arduino boards and popular CH340 / CP210x / FTDI clones.
        """
        detected_boards = []
        seen_ports = set()

        # 1. First attempt: query arduino-cli
        if self.is_available():
            try:
                cmd = [self.cli_path, "board", "list", "--format", "json"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    ports = data.get("detected_ports", []) if isinstance(data, dict) else data

                    for item in ports:
                        port_info = item.get("port", {})
                        address = port_info.get("address", "")
                        boards = item.get("boards", [])
                        protocol = port_info.get("protocol", "serial")

                        if boards:
                            for b in boards:
                                fqbn = b.get("fqbn", self.default_fqbn)
                                name = b.get("name", "Arduino Board")
                                if address and address not in seen_ports:
                                    detected_boards.append({
                                        "port": address,
                                        "fqbn": fqbn,
                                        "name": name,
                                        "protocol": protocol,
                                    })
                                    seen_ports.add(address)
            except Exception as e:
                logger.warning("Error running arduino-cli board list: %s", e)

        # 2. Second attempt: scan serial ports for USB-Serial / CH340 / FTDI devices
        try:
            import serial.tools.list_ports
            for p in serial.tools.list_ports.comports():
                dev = p.device
                desc = (p.description or "").lower()
                hwid = (p.hwid or "").lower()

                # Check if it's a USB serial device not yet in detected_boards
                is_usb = "usb" in desc or "ch340" in desc or "cp210" in desc or "ftdi" in desc or "1a86" in hwid or "vid:pid" in hwid
                if is_usb and dev not in seen_ports and "bluetooth" not in desc:
                    board_name = p.description or f"USB Serial Device ({dev})"
                    detected_boards.append({
                        "port": dev,
                        "fqbn": self.default_fqbn,
                        "name": board_name,
                        "protocol": "serial",
                    })
                    seen_ports.add(dev)
                    logger.info("Discovered USB Serial board on %s (%s)", dev, board_name)
        except Exception as e:
            logger.warning("Error scanning pyserial comports: %s", e)

        return detected_boards

    def compile_sketch(self, sketch_path: str, fqbn: str = "arduino:avr:uno") -> bool:
        """Compile an Arduino sketch for the specified FQBN."""
        if not self.is_available():
            logger.error("Cannot compile: arduino-cli not available")
            return False

        abs_sketch_path = os.path.abspath(sketch_path)
        logger.info("Compiling Arduino sketch '%s' for FQBN '%s'...", abs_sketch_path, fqbn)

        try:
            cmd = [self.cli_path, "compile", "--fqbn", fqbn, abs_sketch_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                logger.info("Arduino sketch compiled successfully.")
                return True
            else:
                logger.error("Arduino compilation failed:\n%s", result.stderr)
                return False

        except Exception as e:
            logger.error("Compilation error: %s", e)
            return False

    def upload_sketch(self, port: str, sketch_path: str, fqbn: str = "arduino:avr:uno") -> bool:
        """Upload a compiled Arduino sketch to the specified port."""
        if not self.is_available():
            logger.error("Cannot upload: arduino-cli not available")
            return False

        abs_sketch_path = os.path.abspath(sketch_path)
        logger.info("Uploading sketch to port %s (FQBN: %s)...", port, fqbn)

        try:
            cmd = [self.cli_path, "upload", "-p", port, "--fqbn", fqbn, abs_sketch_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                logger.info("Arduino sketch uploaded successfully to %s!", port)
                return True
            else:
                logger.error("Arduino upload failed:\n%s", result.stderr)
                return False

        except Exception as e:
            logger.error("Upload error: %s", e)
            return False

    def flash_board(self, port: str, sketch_path: str, fqbn: str = "arduino:avr:uno") -> bool:
        """Convenience method: compile and upload in one call."""
        if self.compile_sketch(sketch_path, fqbn=fqbn):
            return self.upload_sketch(port, sketch_path, fqbn=fqbn)
        return False
