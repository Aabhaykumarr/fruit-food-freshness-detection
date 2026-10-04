"""
Queries specific articles for cooked dal and thali images.
"""

import os
import requests
from PIL import Image

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluation", "real_world_validation")

ITEMS = [
    {"title": "Tarka dal", "dest": "lentils/dal_01.jpg"},
    {"title": "Sambar (dish)", "dest": "lentils/dal_02.jpg"},
    {"title": "Indian thali", "dest": "indian_dishes/thali_mixed_01.jpg"},
]

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FruitDetectionResearch/1.0"}


def fetch():
    for item in ITEMS:
        title = item["title"]
        dest = os.path.join(DATA_DIR, item["dest"])
        api_url = f"https://en.wikipedia.org/w/api.php?action=query&titles={title}&prop=pageimages&format=json&pithumbsize=600"
        try:
            r = requests.get(api_url, headers=HEADERS, timeout=10)
            data = r.json()
            pages = data.get("query", {}).get("pages", {})
            img_url = None
            for pid, pdata in pages.items():
                if "thumbnail" in pdata:
                    img_url = pdata["thumbnail"]["source"]
                    break

            if img_url:
                print(f"[{title}] -> {img_url}")
                img_r = requests.get(img_url, headers=HEADERS, timeout=15)
                if img_r.status_code == 200:
                    with open(dest, "wb") as f:
                        f.write(img_r.content)
                    with Image.open(dest) as img:
                        img.verify()
                    print(f"  [SAVED] {dest}")
        except Exception as e:
            print(f"Error {title}: {e}")


if __name__ == "__main__":
    fetch()
