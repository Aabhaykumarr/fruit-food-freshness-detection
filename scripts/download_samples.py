"""
Script to create or download sample test images for fruit/food freshness testing.
Generates synthetic sample images if no internet connection is available.
"""

import os
import cv2
import numpy as np

SAMPLE_DIR = "data/samples"


def create_sample_images():
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    # 1. Create a synthetic test image with colored circles representing fruit
    img = np.ones((480, 640, 3), dtype=np.uint8) * 240

    # Draw an apple (red circle with green leaf)
    cv2.circle(img, (200, 240), 90, (30, 30, 220), -1)  # Red body
    cv2.ellipse(img, (200, 140), (20, 10), 45, 0, 360, (50, 180, 50), -1)  # Leaf
    cv2.putText(img, "Sample Apple Test", (110, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)

    # Draw a banana (yellow curve)
    cv2.ellipse(img, (450, 250), (110, 40), 30, 0, 180, (0, 220, 240), -1)  # Yellow
    cv2.putText(img, "Sample Banana Test", (360, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)

    out_path = os.path.join(SAMPLE_DIR, "sample_fruit.jpg")
    cv2.imwrite(out_path, img)
    print(f"Created sample test image: {out_path}")


if __name__ == "__main__":
    create_sample_images()
