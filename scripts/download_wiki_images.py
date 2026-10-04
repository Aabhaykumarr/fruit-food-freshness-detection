"""
Downloads authenticated Wikipedia Commons images for key Indian culinary dishes.
"""

import os
import requests
from PIL import Image

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluation", "real_world_validation")

WIKI_IMAGES = {
    "staples/roti_01.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cb/Chapati_being_made_in_a_pan.jpg/800px-Chapati_being_made_in_a_pan.jpg",
    "staples/paratha_01.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/54/Aloo_Paratha_also_known_as_Batata_Paratha.jpg/800px-Aloo_Paratha_also_known_as_Batata_Paratha.jpg",
    "staples/naan_01.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/52/Butter_Naan.jpg/800px-Butter_Naan.jpg",
    "staples/plain_rice_02.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Rice_01.jpg/800px-Rice_01.jpg",
    "lentils/dal_01.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Dal_Tadka_at_a_Restaurant.jpg/800px-Dal_Tadka_at_a_Restaurant.jpg",
    "lentils/dal_02.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Yellow_dal_tadka_in_a_bowl.jpg/800px-Yellow_dal_tadka_in_a_bowl.jpg",
    "lentils/rajma_01.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Rajma_Masala.jpg/800px-Rajma_Masala.jpg",
    "lentils/chole_01.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Chana_Masala.jpg/800px-Chana_Masala.jpg",
    "indian_dishes/biryani_01.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/Dum_Biryani_Chicken.jpg/800px-Dum_Biryani_Chicken.jpg",
    "indian_dishes/thali_mixed_01.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b3/Gujarati_Thali_at_Toran_Dining_Hall.jpg/800px-Gujarati_Thali_at_Toran_Dining_Hall.jpg"
}

HEADERS = {"User-Agent": "FruitDetectionBot/1.0 (academic research and model evaluation)"}


def download_verified_wiki_images():
    for rel_path, url in WIKI_IMAGES.items():
        dest = os.path.join(DATA_DIR, rel_path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print(f"Downloading verified image for {rel_path}...")
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                with open(dest, "wb") as f:
                    f.write(r.content)
                with Image.open(dest) as img:
                    img.verify()
                print(f"  [OK] Saved and verified: {dest}")
            else:
                print(f"  [ERR] HTTP {r.status_code} for {url}")
        except Exception as e:
            print(f"  [ERR] {e}")


if __name__ == "__main__":
    download_verified_wiki_images()
