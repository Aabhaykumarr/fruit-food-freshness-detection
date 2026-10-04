"""
Detection service — Compatibility wrapper forwarding to the Hierarchical Vision Service (SigLIP).
"""

import logging
from typing import List, Dict, Any, Optional
from services.vision_service import VisionService

try:
    from config import FOOD_CLASSES, CONFIDENCE_THRESHOLD
except ImportError:
    FOOD_CLASSES = []
    CONFIDENCE_THRESHOLD = 0.40

logger = logging.getLogger(__name__)


class DetectionService:
    """
    Compatibility wrapper delegating to Hierarchical Vision Service.
    Maintains compatibility with legacy detect(frame) callers.
    """

    def __init__(self, model_path: Optional[str] = None, classes: Optional[List[str]] = None, conf: float = CONFIDENCE_THRESHOLD):
        self.conf = conf
        self.classes = classes if classes is not None else list(FOOD_CLASSES)
        self.vision_service = VisionService()

    def detect(self, frame) -> List[Dict[str, Any]]:
        """
        Detect food and fruit items in a frame.
        Delegates directly to VisionService.
        """
        return self.vision_service.detect(frame)

    def analyze_image(self, frame) -> Dict[str, Any]:
        """
        Full hierarchical analysis on a single image.
        """
        return self.vision_service.analyze_image(frame)

    def set_custom_classes(self, new_classes: List[str]):
        """Dynamically update classes (compatibility stub)."""
        self.classes = new_classes
