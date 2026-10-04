"""
Hardware Manager Service.

Orchestrates the entire hardware lifecycle on startup:
  1. Scans for connected Arduino boards.
  2. If found & auto-flash enabled: automatically compiles and uploads the sensor reader firmware.
  3. Seamlessly initializes real Serial SensorService or falls back to Mock SensorService.
  4. Requires ZERO manual intervention — true plug-and-play.
"""

import time
import logging
from typing import Dict, Any, Tuple

from services.arduino_flasher import ArduinoFlasher
from services.sensor_service import SensorService
import config

logger = logging.getLogger(__name__)


class HardwareManager:
    """
    Unified hardware lifecycle manager for Arduino and environmental sensors.
    """

    def __init__(
        self,
        cli_path: str = config.ARDUINO_CLI_PATH,
        sketch_path: str = config.ARDUINO_SKETCH_PATH,
        auto_flash: bool = config.AUTO_FLASH_ARDUINO,
        default_fqbn: str = config.DEFAULT_FQBN,
        baud_rate: int = config.BAUD_RATE,
    ):
        self.sketch_path = sketch_path
        self.auto_flash = auto_flash
        self.default_fqbn = default_fqbn
        self.baud_rate = baud_rate
        self.flasher = ArduinoFlasher(cli_path=cli_path)
        self.sensor_service: Optional[SensorService] = None
        self.hardware_info: Dict[str, Any] = {}

    def initialize(self) -> SensorService:
        """
        Auto-discover hardware, flash if connected, and start the sensor service.
        Returns the active SensorService instance.
        """
        print("\n" + "-" * 60)
        print("  HARDWARE INITIALIZATION & AUTO-DISCOVERY")
        print("-" * 60)

        # 1. Scan for connected Arduino boards
        detected_boards = self.flasher.detect_boards()
        target_port = None
        target_fqbn = self.default_fqbn

        if detected_boards:
            board = detected_boards[0]
            target_port = board["port"]
            target_fqbn = board.get("fqbn", self.default_fqbn)
            board_name = board.get("name", "Arduino Board")

            print(f"  [+] Detected Hardware: {board_name} on {target_port} ({target_fqbn})")
            self.hardware_info = {
                "connected": True,
                "port": target_port,
                "fqbn": target_fqbn,
                "name": board_name,
            }

            # 2. Auto-flash firmware if enabled
            if self.auto_flash and self.flasher.is_available():
                print(f"  [i] Auto-uploading sensor firmware '{self.sketch_path}' to {target_port}...")
                success = self.flasher.flash_board(
                    port=target_port,
                    sketch_path=self.sketch_path,
                    fqbn=target_fqbn,
                )
                if success:
                    print(f"  [+] Firmware successfully uploaded to {target_port}!")
                    # Give Arduino 2 seconds to reboot after serial flashing
                    time.sleep(2.0)
                else:
                    print(f"  [!] Auto-upload failed. Will attempt direct serial connection...")

            # 3. Start Serial Sensor Service
            print(f"  [+] Connecting to live telemetry on {target_port} @ {self.baud_rate} baud...")
            self.sensor_service = SensorService(
                mock_mode=False,
                port=target_port,
                baud_rate=self.baud_rate,
                timeout=config.SENSOR_TIMEOUT,
                stale_seconds=config.SENSOR_STALE_SECONDS,
            )
            self.sensor_service.start()
            print("  [+] Real Arduino Sensor Service ACTIVE")

        else:
            # No physical Arduino detected
            print("  [i] No physical Arduino detected on USB ports.")
            print("  [+] Starting with realistic MOCK SENSORS.")
            print("      (Tip: Connect your Arduino anytime to auto-detect & flash on next run!)")

            self.hardware_info = {
                "connected": False,
                "port": None,
                "mode": "mock",
            }

            self.sensor_service = SensorService(
                mock_mode=True,
                port="none",
                baud_rate=self.baud_rate,
                timeout=config.SENSOR_TIMEOUT,
                stale_seconds=config.SENSOR_STALE_SECONDS,
            )
            self.sensor_service.start()
            print("  [+] Mock Sensor Service ACTIVE")

        print("-" * 60 + "\n")
        return self.sensor_service

    def shutdown(self):
        """Stop active sensor services cleanly."""
        if self.sensor_service is not None:
            self.sensor_service.stop()
            logger.info("HardwareManager stopped sensor service")
