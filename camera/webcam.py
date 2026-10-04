"""
Webcam source — wraps cv2.VideoCapture for live camera input.
"""

import cv2
import logging

logger = logging.getLogger(__name__)


class WebcamSource:
    """Live webcam/CCTV frame source."""

    def __init__(self, source=0):
        self.source = source
        self.cap = None

    def open(self):
        """Open the camera. Returns True if successful."""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            logger.error("Failed to open camera source: %s", self.source)
            return False
        logger.info("Camera opened: source=%s", self.source)
        return True

    def read_frame(self):
        """Read a single frame. Returns (success, frame)."""
        if self.cap is None or not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def release(self):
        """Release the camera."""
        if self.cap is not None:
            self.cap.release()
            logger.info("Camera released")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
