"""
Copies and normalizes ground truth images for the real-world validation suite.
"""

import shutil
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation")
RW_DIR = os.path.join(EVAL_DIR, "real_world_validation")

# Copy verified images
shutil.copyfile(
    os.path.join(EVAL_DIR, "difficult", "dal_tadka_01.jpg"),
    os.path.join(RW_DIR, "lentils", "dal_01.jpg")
)

shutil.copyfile(
    os.path.join(EVAL_DIR, "difficult", "indian_thali_01.jpg"),
    os.path.join(RW_DIR, "indian_dishes", "thali_mixed_01.jpg")
)

print("Verified test images copied successfully.")
