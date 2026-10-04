"""
Image loader — loads static images for single-image analysis mode.
"""

import os
import cv2
import logging

logger = logging.getLogger(__name__)


class ImageSource:
    """Static image source for single-image or batch analysis."""

    def load(self, path):
        """
        Load an image from a file path.
        Returns the frame (numpy array) or None on failure.
        """
        if not os.path.exists(path):
            logger.error("Image not found: %s", path)
            return None

        frame = cv2.imread(path)
        if frame is None:
            logger.error("Failed to read image: %s", path)
            return None

        logger.info("Image loaded: %s (%dx%d)", path, frame.shape[1], frame.shape[0])
        return frame

    def load_directory(self, directory):
        """
        Load all images from a directory.
        Yields (filename, frame) tuples.
        """
        if not os.path.isdir(directory):
            logger.error("Directory not found: %s", directory)
            return

        extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        for filename in sorted(os.listdir(directory)):
            if filename.lower().endswith(extensions):
                path = os.path.join(directory, filename)
                frame = self.load(path)
                if frame is not None:
                    yield filename, frame
